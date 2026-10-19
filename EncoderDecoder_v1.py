#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
LSTM koder -> sažeti vektor -> ponavljanje vektora ->
LSTM dekoder -> izlazi po koracima -> regresija raspodela.

Tri CSV-a obrađuju se nezavisno.
Bez nasumične inicijalizacije, šuma, uzorkovanja i mešanja.
Bez klasifikacione funkcije gubitka.
Bez zaokruživanja predviđenih brojeva.
Bez filtera za paritet, polovine, zone ili uzastopne brojeve.
"""


import argparse
import csv
import hashlib
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "Nedostaje biblioteka. Instalacija:\n"
        "python3 -m pip install numpy torch\n"
        f"Detalj: {exc}"
    )


DEFAULT_FILES = (
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
)

BALLS = 39
PICKS = 7
WINDOW = 8
FEATURES = 12

VALIDATION = 48
TEST = 64
MIN_TRAIN_SAMPLES = 128

MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 128
LEARNING_RATE = 0.0005

DEVICE = torch.device("cpu")


def configure_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def read_csv(path):
    path = Path(path).expanduser()

    if not path.is_file():
        raise ValueError(f"Fajl ne postoji: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()

            if not line:
                continue

            # Podržava CSV sa zarezom, tačka-zarezom ili razmacima.
            if "," in line or ";" in line:
                delimiter = "," if "," in line else ";"
                parts = next(csv.reader([line], delimiter=delimiter))
                parts = [value.strip() for value in parts]
            else:
                parts = re.split(r"\s+", line)

            if len(parts) != PICKS:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 brojeva, pronađeno {len(parts)}."
                )

            if any(re.fullmatch(r"\d+", value) is None for value in parts):
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "dozvoljeno je samo sedam celih brojeva, bez zaglavlja."
                )

            numbers = [int(value) for value in parts]

            if any(number < 1 or number > BALLS for number in numbers):
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "brojevi moraju biti između 1 i 39."
                )

            if len(set(numbers)) != PICKS:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            rows.append(sorted(numbers))

    required = WINDOW + MIN_TRAIN_SAMPLES + VALIDATION + TEST

    if len(rows) < required:
        raise ValueError(
            f"{path.name}: potrebno najmanje {required} izvlačenja; "
            f"pronađeno {len(rows)}."
        )

    return path, np.asarray(rows, dtype=np.int64)


def draw_profiles(draws):
    """
    Sedam sortiranih brojeva i pet opisa raspodele kombinacije.

    Četiri vrednosti opisuju empirijsku raspodelu svih 21
    međusobnih udaljenosti. To nije reprezentacija osam razmaka.
    """
    profile = np.empty((len(draws), FEATURES), dtype=np.float32)
    profile[:, :PICKS] = draws / float(BALLS)

    left, right = np.triu_indices(PICKS, k=1)
    distances = draws[:, right] - draws[:, left]

    for column, threshold in enumerate((2, 5, 10, 20), start=PICKS):
        profile[:, column] = np.mean(
            distances <= threshold, axis=1
        )

    profile[:, 11] = (draws[:, -1] - draws[:, 0]) / 38.0
    return profile


def prepare_dataset(draws):
    profiles = draw_profiles(draws)
    count = len(draws) - WINDOW

    inputs = np.empty(
        (count, WINDOW, FEATURES), dtype=np.float32
    )

    for sample in range(count):
        inputs[sample] = profiles[sample:sample + WINDOW]

    outcomes = draws[WINDOW:].copy()

    grid = np.arange(1, BALLS + 1)
    target_cdf = (
        outcomes[:, :, None] <= grid[None, None, :]
    ).astype(np.float64)

    next_input = profiles[-WINDOW:][None].copy()

    return inputs, target_cdf, outcomes, next_input


def analytic_fill(parameter, phase):
    """
    Analitička, ponovljiva inicijalizacija bez generatora slučajnosti.
    Različite matrice i različite jedinice dobijaju različite vrednosti.
    """
    with torch.no_grad():
        if parameter.ndim == 1:
            parameter.zero_()
            return

        rows, columns = parameter.shape
        row = torch.arange(
            1, rows + 1, dtype=torch.float64,
            device=parameter.device
        )[:, None]

        column = torch.arange(
            1, columns + 1, dtype=torch.float64,
            device=parameter.device
        )[None, :]

        angles = (
            math.sqrt(2.0) * row * column
            + 0.113 * row.square()
            + 0.071 * column.square()
            + phase
        )

        values = torch.sin(angles) * math.sqrt(2.0 / columns)
        parameter.copy_(values.to(parameter.dtype))


class AnalyticLinear(nn.Linear):
    def __init__(self, in_features, out_features, phase):
        self.phase = float(phase)
        super().__init__(in_features, out_features)

    def reset_parameters(self):
        analytic_fill(self.weight, self.phase)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()


class AnalyticLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, phase):
        self.phase = float(phase)

        # reset_parameters is overridden, so the constructor
        # does not perform random initialization.
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

    def reset_parameters(self):
        for index, (_, parameter) in enumerate(
            self.named_parameters(recurse=False)
        ):
            analytic_fill(parameter, self.phase + index * 0.379)

        with torch.no_grad():
            # PyTorch gate order: input, forget, cell, output.
            self.bias_ih_l0[
                self.hidden_size:2 * self.hidden_size
            ].fill_(1.0)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = AnalyticLSTM(FEATURES, 32, phase=0.17)
        self.encoder_2 = AnalyticLSTM(32, 16, phase=0.63)

        self.decoder_1 = AnalyticLSTM(16, 32, phase=1.11)
        self.decoder_2 = AnalyticLSTM(32, 16, phase=1.73)

        # Corresponds to TimeDistributed(Dense(nb_features)).
        self.step_projection = AnalyticLinear(
            16, FEATURES, phase=2.31
        )

        self.hidden = AnalyticLinear(
            WINDOW * FEATURES, 64, phase=2.89
        )

        self.output = AnalyticLinear(
            64, PICKS * BALLS, phase=3.47
        )

        rank = torch.arange(PICKS)[:, None]
        number_index = torch.arange(BALLS)[None, :]

        # Legal support of each sorted position:
        # first number 1..33, second 2..34, ..., seventh 7..39.
        legal = (
            (number_index >= rank)
            & (number_index <= BALLS - PICKS + rank)
        )

        self.register_buffer("legal_support", legal)

    def forward(self, inputs):
        encoded, _ = self.encoder_1(inputs)
        _, (state, _) = self.encoder_2(encoded)

        compressed = state[-1]
        repeated = compressed[:, None, :].expand(
            -1, WINDOW, -1
        ).contiguous()

        decoded, _ = self.decoder_1(repeated)
        decoded, _ = self.decoder_2(decoded)

        projected = self.step_projection(decoded)
        flattened = projected.reshape(len(inputs), -1)
        hidden = F.relu(self.hidden(flattened))

        logits = self.output(hidden).reshape(
            -1, PICKS, BALLS
        ).to(torch.float64)

        # Positive masses learned by CDF regression.
        # These are not classifier scores.
        mass = F.softplus(logits) + 1e-12
        mass = mass * self.legal_support[None]

        return mass / mass.sum(dim=-1, keepdim=True)


def fit_scaler(inputs):
    mean = inputs.mean(axis=(0, 1), dtype=np.float64)
    scale = inputs.std(axis=(0, 1), dtype=np.float64)

    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def scaled_tensor(inputs, scaler):
    mean, scale = scaler
    values = np.ascontiguousarray(
        (inputs - mean) / scale, dtype=np.float32
    )
    return torch.from_numpy(values).to(DEVICE)


def cdf_error(mass, target_cdf):
    return (
        mass.cumsum(dim=-1) - target_cdf
    ).square().mean()


def train_epoch(model, optimizer, inputs, targets):
    model.train()
    total_loss = 0.0
    total_samples = 0

    # Chronological batches, without shuffling.
    for start in range(0, len(inputs), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(inputs))

        optimizer.zero_grad(set_to_none=True)
        mass = model(inputs[start:stop])
        loss = cdf_error(mass, targets[start:stop])

        if not torch.isfinite(loss):
            raise RuntimeError("Nekonačna vrednost greške pri treniranju.")

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if not torch.isfinite(norm):
            raise RuntimeError("Nekonačan gradijent pri treniranju.")

        optimizer.step()

        total_loss += float(loss.detach()) * (stop - start)
        total_samples += stop - start

    return total_loss / total_samples


def predict_mass(model, inputs, scaler):
    model.eval()
    batches = []

    with torch.no_grad():
        for start in range(0, len(inputs), BATCH_SIZE):
            block = scaled_tensor(
                inputs[start:start + BATCH_SIZE], scaler
            )
            batches.append(model(block).cpu().numpy())

    masses = np.concatenate(batches, axis=0)

    if not np.isfinite(masses).all():
        raise RuntimeError("Model je vratio nekonačne vrednosti.")

    return masses


def choose_epochs(train_x, train_y, val_x, val_y):
    scaler = fit_scaler(train_x)
    inputs = scaled_tensor(train_x, scaler)
    targets = torch.from_numpy(
        np.ascontiguousarray(train_y)
    ).to(DEVICE)

    validation_inputs = scaled_tensor(val_x, scaler)
    validation_targets = torch.from_numpy(
        np.ascontiguousarray(val_y)
    ).to(DEVICE)

    model = EncoderDecoder().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        training_loss = train_epoch(
            model, optimizer, inputs, targets
        )

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                cdf_error(
                    model(validation_inputs),
                    validation_targets,
                )
            )

        if not math.isfinite(validation_loss):
            raise RuntimeError("Nekonačna validaciona greška.")

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Izbor epoha {epoch}/{MAX_EPOCHS} | "
                f"trening={training_loss:.6f} | "
                f"validacija={validation_loss:.6f}",
                flush=True,
            )

        if stale >= PATIENCE:
            break

    if best_epoch == 0:
        raise RuntimeError("Nije pronađena važeća epoha.")

    return best_epoch, best_loss


def fit_fixed(inputs, targets, epochs, label):
    scaler = fit_scaler(inputs)
    tensor_x = scaled_tensor(inputs, scaler)
    tensor_y = torch.from_numpy(
        np.ascontiguousarray(targets)
    ).to(DEVICE)

    model = EncoderDecoder().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, optimizer, tensor_x, tensor_y
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  {label}: epoha {epoch}/{epochs} | "
                f"greška={loss:.6f}",
                flush=True,
            )

    return model, scaler


def best_combination(mass):
    """
    Tačno dinamičko programiranje nad proizvodom naučenih
    marginalnih masa sortiranih pozicija.

    Maksimizuje:
        sum_j log(mass[j, broj_j - 1])

    uz jedina ograničenja:
        1 <= broj_1 < ... < broj_7 <= 39.

    Nema zaokruživanja, slučajnog biranja ili naknadnog pomeranja.
    """
    if mass.shape != (PICKS, BALLS):
        raise ValueError("Pogrešan oblik izlaza modela.")

    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("Neispravna raspodela modela.")

    log_mass = np.full_like(mass, -np.inf, dtype=np.float64)
    np.log(mass, out=log_mass, where=mass > 0)

    score = np.full((PICKS, BALLS), -np.inf)
    parent = np.full((PICKS, BALLS), -1, dtype=np.int64)

    score[0, :BALLS - PICKS + 1] = (
        log_mass[0, :BALLS - PICKS + 1]
    )

    for rank in range(1, PICKS):
        best_previous_score = -np.inf
        best_previous_index = -1

        for number_index in range(BALLS):
            previous_index = number_index - 1

            if (
                previous_index >= 0
                and score[rank - 1, previous_index]
                > best_previous_score
            ):
                best_previous_score = score[
                    rank - 1, previous_index
                ]
                best_previous_index = previous_index

            if not (
                rank <= number_index <= BALLS - PICKS + rank
            ):
                continue

            if best_previous_index >= 0:
                score[rank, number_index] = (
                    best_previous_score
                    + log_mass[rank, number_index]
                )
                parent[rank, number_index] = best_previous_index

    last = int(np.argmax(score[-1]))

    if not np.isfinite(score[-1, last]):
        raise RuntimeError("Nema važeće kombinacije iz izlaza modela.")

    selected = np.empty(PICKS, dtype=np.int64)
    selected[-1] = last + 1

    for rank in range(PICKS - 1, 0, -1):
        last = int(parent[rank, last])

        if last < 0:
            raise RuntimeError("Neispravna putanja izbora kombinacije.")

        selected[rank - 1] = last + 1

    result = tuple(int(value) for value in selected)

    if (
        len(set(result)) != PICKS
        or result[0] < 1
        or result[-1] > BALLS
        or tuple(sorted(result)) != result
    ):
        raise RuntimeError("Izabrana kombinacija nije važeća.")

    return result


def format_numbers(numbers):
    return ", ".join(str(int(number)) for number in numbers)


def longest_run(numbers):
    best = current = 1

    for previous, following in zip(numbers[:-1], numbers[1:]):
        current = current + 1 if following == previous + 1 else 1
        best = max(best, current)

    return best


def evaluate_test(masses, target_cdf, actual_draws, prior_cdf):
    predicted_cdf = masses.cumsum(axis=-1)

    model_error = float(
        np.mean((predicted_cdf - target_cdf) ** 2)
    )
    reference_error = float(
        np.mean((prior_cdf[None] - target_cdf) ** 2)
    )

    hits = []

    for mass, actual in zip(masses, actual_draws):
        prediction = best_combination(mass)
        hits.append(len(set(prediction) & set(actual.tolist())))

    distribution = np.bincount(hits, minlength=PICKS + 1)

    print(f"  Završna provera: {len(hits)} kola")
    print(f"  Greška regresije CDF: {model_error:.6f}")
    print(
        "  Greška istorijskog proseka CDF "
        f"(poređenje): {reference_error:.6f}"
    )
    print(f"  Prosečan broj pogodaka: {np.mean(hits):.4f}")
    print(
        "  Pogoci 0–7: "
        + " | ".join(
            f"{count}: {int(distribution[count])}"
            for count in range(PICKS + 1)
        )
    )


def process_file(path):
    path, draws = read_csv(path)
    inputs, targets, outcomes, next_input = prepare_dataset(draws)

    train_end = len(inputs) - VALIDATION - TEST
    validation_end = len(inputs) - TEST

    fingerprint = hashlib.sha256(
        draws.astype("<i8", copy=False).tobytes()
    ).hexdigest()[:16]

    print("\n" + "=" * 72)
    print(path.name)
    print(
        f"Izvlačenja: {len(draws)} | "
        f"istorijski prelazi: {len(inputs)} | "
        f"oznaka sadržaja: {fingerprint}",
        flush=True,
    )

    print("\n1. Izbor broja epoha na hronološkoj validaciji")

    epochs, validation_loss = choose_epochs(
        inputs[:train_end],
        targets[:train_end],
        inputs[train_end:validation_end],
        targets[train_end:validation_end],
    )

    print(
        f"  Izabrano epoha: {epochs} | "
        f"validaciona greška: {validation_loss:.6f}"
    )

    print("\n2. Nezavisna završna provera")

    test_model, test_scaler = fit_fixed(
        inputs[:validation_end],
        targets[:validation_end],
        epochs,
        label="Model za proveru",
    )

    test_masses = predict_mass(
        test_model,
        inputs[validation_end:],
        test_scaler,
    )

    # Reference uses only the history preceding the test period.
    prior_cdf = targets[:validation_end].mean(axis=0)

    evaluate_test(
        test_masses,
        targets[validation_end:],
        outcomes[validation_end:],
        prior_cdf,
    )

    del test_model, test_scaler, test_masses

    print(
        "\n3. Završno treniranje na celoj poznatoj istoriji",
        flush=True,
    )

    final_model, final_scaler = fit_fixed(
        inputs,
        targets,
        epochs,
        label="Završni model",
    )

    mass = predict_mass(
        final_model, next_input, final_scaler
    )[0]

    next_combination = best_combination(mass)

    print("\nKontrola: prethodno izvuceno kolo")
    print(format_numbers(draws[-1]))

    print("\nNEXT:")
    print(format_numbers(next_combination))

    low_count = sum(number <= 19 for number in next_combination)
    history_matches = int(
        np.sum(
            np.all(
                draws == np.asarray(next_combination)[None],
                axis=1,
            )
        )
    )

    print(
        "\nOpis izlaza: "
        f"najduži uzastopni niz={longest_run(next_combination)}; "
        f"1–19: {low_count}; 20–39: {PICKS - low_count}; "
        f"ista kombinacija u istoriji: {history_matches}."
    )
    print("Opis izlaza ne utiče na izbor brojeva.")

    return path.name, next_combination


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Loto 7/39 — LSTM koder i dekoder, V1."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Tri CSV putanje. Bez argumenata koristi DEFAULT_FILES.",
    )

    args = parser.parse_args()

    if args.csv_files and len(args.csv_files) != 3:
        parser.error("Prosledi tačno tri CSV putanje.")

    return args.csv_files or list(DEFAULT_FILES)


def main():
    paths = parse_arguments()
    configure_torch()

    print("LOTO 7/39 — LSTM KODER I DEKODER V1")
    print("Odvojeni modeli; regresija raspodela; bez random koraka.")

    results = []
    failures = []

    for path in paths:
        try:
            results.append(process_file(path))
        except (ValueError, OSError, RuntimeError, csv.Error) as exc:
            failures.append((str(path), str(exc)))
            print(f"\nGREŠKA za {Path(path).name}: {exc}", flush=True)

    if results:
        print("\n" + "=" * 72)
        print("KONAČNI NEXT REZULTATI")

        for name, combination in results:
            print(f"{name}: ({format_numbers(combination)})")

        for index, (first_name, first_combo) in enumerate(results):
            for second_name, second_combo in results[index + 1:]:
                if first_combo == second_combo:
                    print(
                        "\nIsti NEXT: "
                        f"{first_name} / {second_name}: "
                        f"({format_numbers(first_combo)}). "
                        "Rezultat nije naknadno menjan."
                    )

    if failures:
        print(
            "\nZa neuspešno obrađene fajlove nema zamenskih kombinacija."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



"""
LOTO 7/39 — LSTM KODER I DEKODER V1
Odvojeni modeli; regresija raspodela; bez random koraka.



========================================================================
loto7_4704_k83.csv
Izvlačenja: 4704 | istorijski prelazi: 4696 | oznaka sadržaja: 1badcf2632a9f966

1. Izbor broja epoha na hronološkoj validaciji
  Izbor epoha 1/100 | trening=0.127921 | validacija=0.095038
  Izbor epoha 10/100 | trening=0.073241 | validacija=0.069545
  Izbor epoha 20/100 | trening=0.073190 | validacija=0.069470
  Izbor epoha 30/100 | trening=0.073174 | validacija=0.069447
  Izbor epoha 40/100 | trening=0.073153 | validacija=0.069423
  Izbor epoha 50/100 | trening=0.073061 | validacija=0.069366
  Izbor epoha 60/100 | trening=0.072501 | validacija=0.069257
  Izbor epoha 70/100 | trening=0.071014 | validacija=0.070827
  Izabrano epoha: 59 | validaciona greška: 0.069046

2. Nezavisna završna provera
  Model za proveru: epoha 1/59 | greška=0.127592
  Model za proveru: epoha 10/59 | greška=0.073200
  Model za proveru: epoha 20/59 | greška=0.073149
  Model za proveru: epoha 30/59 | greška=0.073133
  Model za proveru: epoha 40/59 | greška=0.073111
  Model za proveru: epoha 50/59 | greška=0.073002
  Model za proveru: epoha 59/59 | greška=0.072393
  Završna provera: 64 kola
  Greška regresije CDF: 0.073602
  Greška istorijskog proseka CDF (poređenje): 0.073868
  Prosečan broj pogodaka: 1.3906
  Pogoci 0-7: 0: 15 | 1: 19 | 2: 21 | 3: 8 | 4: 1 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: epoha 1/59 | greška=0.127149
  Završni model: epoha 10/59 | greška=0.073207
  Završni model: epoha 20/59 | greška=0.073155
  Završni model: epoha 30/59 | greška=0.073140
  Završni model: epoha 40/59 | greška=0.073120
  Završni model: epoha 50/59 | greška=0.073023
  Završni model: epoha 59/59 | greška=0.072288

Kontrola: prethodno izvuceno kolo
7, 10, 13, 27, 28, 31, 38

NEXT:
1, 8, 12, 18, 29, 33, 39

Opis izlaza: najduži uzastopni niz=1; 1-19: 4; 20-39: 3; ista kombinacija u istoriji: 0.
Opis izlaza ne utiče na izbor brojeva.



========================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974 | istorijski prelazi: 2966 | oznaka sadržaja: 992bc400ed1af4f4

1. Izbor broja epoha na hronološkoj validaciji
  Izbor epoha 1/100 | trening=0.136248 | validacija=0.124645
  Izbor epoha 10/100 | trening=0.073285 | validacija=0.069875
  Izbor epoha 20/100 | trening=0.073177 | validacija=0.069847
  Izbor epoha 30/100 | trening=0.073146 | validacija=0.069829
  Izbor epoha 40/100 | trening=0.073092 | validacija=0.069796
  Izbor epoha 50/100 | trening=0.073027 | validacija=0.069873
  Izabrano epoha: 39 | validaciona greška: 0.069788

2. Nezavisna završna provera
  Model za proveru: epoha 1/39 | greška=0.136077
  Model za proveru: epoha 10/39 | greška=0.073217
  Model za proveru: epoha 20/39 | greška=0.073115
  Model za proveru: epoha 30/39 | greška=0.073087
  Model za proveru: epoha 39/39 | greška=0.073066
  Završna provera: 64 kola
  Greška regresije CDF: 0.072476
  Greška istorijskog proseka CDF (poređenje): 0.072466
  Prosečan broj pogodaka: 1.2656
  Pogoci 0-7: 0: 14 | 1: 27 | 2: 17 | 3: 4 | 4: 2 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: epoha 1/39 | greška=0.135864
  Završni model: epoha 10/39 | greška=0.073202
  Završni model: epoha 20/39 | greška=0.073104
  Završni model: epoha 30/39 | greška=0.073064
  Završni model: epoha 39/39 | greška=0.073063

Kontrola: prethodno izvuceno kolo
9, 14, 17, 19, 24, 28, 35

NEXT:
1, 9, 14, 21, 29, 33, 39

Opis izlaza: najduži uzastopni niz=1; 1-19: 3; 20-39: 4; ista kombinacija u istoriji: 0.
Opis izlaza ne utiče na izbor brojeva.



========================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730 | istorijski prelazi: 1722 | oznaka sadržaja: fc57041695a80b3c

1. Izbor broja epoha na hronološkoj validaciji
  Izbor epoha 1/100 | trening=0.140184 | validacija=0.141530
  Izbor epoha 10/100 | trening=0.073481 | validacija=0.076148
  Izbor epoha 20/100 | trening=0.073247 | validacija=0.075905
  Izbor epoha 30/100 | trening=0.073183 | validacija=0.075836
  Izbor epoha 40/100 | trening=0.073156 | validacija=0.075811
  Izbor epoha 50/100 | trening=0.073138 | validacija=0.075799
  Izbor epoha 60/100 | trening=0.073118 | validacija=0.075793
  Izbor epoha 70/100 | trening=0.073051 | validacija=0.075744
  Izbor epoha 80/100 | trening=0.072838 | validacija=0.075554
  Izbor epoha 90/100 | trening=0.071009 | validacija=0.075715
  Izbor epoha 100/100 | trening=0.068678 | validacija=0.075980
  Izabrano epoha: 89 | validaciona greška: 0.075177

2. Nezavisna završna provera
  Model za proveru: epoha 1/89 | greška=0.140242
  Model za proveru: epoha 10/89 | greška=0.073556
  Model za proveru: epoha 20/89 | greška=0.073321
  Model za proveru: epoha 30/89 | greška=0.073257
  Model za proveru: epoha 40/89 | greška=0.073230
  Model za proveru: epoha 50/89 | greška=0.073214
  Model za proveru: epoha 60/89 | greška=0.073199
  Model za proveru: epoha 70/89 | greška=0.073176
  Model za proveru: epoha 80/89 | greška=0.073071
  Model za proveru: epoha 89/89 | greška=0.072688
  Završna provera: 64 kola
  Greška regresije CDF: 0.072770
  Greška istorijskog proseka CDF (poređenje): 0.072645
  Prosečan broj pogodaka: 1.3594
  Pogoci 0–7: 0: 16 | 1: 23 | 2: 15 | 3: 7 | 4: 2 | 5: 1 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: epoha 1/89 | greška=0.140093
  Završni model: epoha 10/89 | greška=0.073512
  Završni model: epoha 20/89 | greška=0.073297
  Završni model: epoha 30/89 | greška=0.073236
  Završni model: epoha 40/89 | greška=0.073210
  Završni model: epoha 50/89 | greška=0.073192
  Završni model: epoha 60/89 | greška=0.073178
  Završni model: epoha 70/89 | greška=0.073150
  Završni model: epoha 80/89 | greška=0.073034
  Završni model: epoha 89/89 | greška=0.072295

Kontrola: prethodno izvuceno kolo
7, 10, 13, 27, 28, 31, 38

NEXT:
1, 8, 12, 18, 29, 34, 39

Opis izlaza: najduži uzastopni niz=1; 1-19: 4; 20-39: 3; ista kombinacija u istoriji: 0.
Opis izlaza ne utiče na izbor brojeva.



========================================================================
KONAČNI NEXT REZULTATI
loto7_4704_k83.csv: (1, 8, 12, 18, 29, 33, 39)
loto7_4704_k83_loto_2974.csv: (1, 9, 14, 21, 29, 33, 39)
loto7_4704_k83_loto_plus_1730.csv: (1, 8, 12, 18, 29, 34, 39)
"""



"""
LSTM mreža sa koderom i dekoderom; izlaz je regresija raspodela. 
Model nazvan „autoenkoder“ sažima niz prethodnih izvlačenja u unutrašnji vektor, zatim ga razvija kroz dekoder i predviđa sledeće izvlačenje. 
Iako je nazvana autoenkoderom, trenira se za predviđanje sledećeg kola, bez posebnog zadatka rekonstrukcije ulaza.
Sačuvati taj postupak, koristiti raspodele kombinacija i regresioni izlaz za 7/39. 
Uklonjen šum, zaokruživanje i ručno bodovanje kombinacija, a skaliranje računati samo na delu za treniranje — sada uključuje i validacione podatke.
"""
