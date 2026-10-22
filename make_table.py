import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


RESULTS_DIRECTORY = Path("results")
RAW_RESULTS_FILE = RESULTS_DIRECTORY / "model_comparison_raw.csv"
BASELINE_FILE = RESULTS_DIRECTORY / "observed_baselines.csv"

OUTPUT_FILE = Path("tables") / "table.csv"

MODEL_NAMES = ["vae", "vae_u", "ivae"]
MODEL_TITLES = {"vae": "VAE", "vae_u": "VAE + u", "ivae": "iVAE"}
OBSERVED_VARIABLES = ["x2", "x3", "x4", "x5"]

SAMPLE_SIZES = None
UNCERTAINTY = "sd"

MEAN_DECIMALS = 3
SPREAD_DECIMALS = 3 if UNCERTAINTY == "sd" else 4

PLUS_MINUS = "\u00b1"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def condition_key(row):
    return (
        int(row["n_per_environment"]),
        float(row["child_noise_std"]),
        row["weight_label"],
    )


def load_model_scores():
    scores = defaultdict(lambda: defaultdict(list))
    for row in read_rows(RAW_RESULTS_FILE):
        scores[condition_key(row)][row["model"]].append(abs(float(row["mcc"])))
    return scores


def load_baseline_scores():
    scores = defaultdict(lambda: defaultdict(list))
    for row in read_rows(BASELINE_FILE):
        for variable in OBSERVED_VARIABLES:
            value = abs(float(row[variable + "_absolute_pearson"]))
            scores[condition_key(row)][variable].append(value)
    return scores


def format_estimate(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.{MEAN_DECIMALS}f}"
    spread = statistics.stdev(values)
    if UNCERTAINTY == "se":
        spread = spread / math.sqrt(len(values))
    return f"{mean:.{MEAN_DECIMALS}f} {PLUS_MINUS} {spread:.{SPREAD_DECIMALS}f}"


def best_observed_variable(baseline_scores, condition):
    means = {
        variable: statistics.mean(values)
        for variable, values in baseline_scores[condition].items()
    }
    variable = max(means, key=means.get)
    return variable, means[variable]


def selected_conditions(model_scores):
    conditions = list(model_scores)
    if SAMPLE_SIZES is not None:
        conditions = [key for key in conditions if key[0] in SAMPLE_SIZES]
    if not conditions:
        raise SystemExit("No results match the requested sample sizes.")
    conditions.sort(key=lambda key: (key[2], key[1], key[0]))
    return conditions


def build_table(model_scores, baseline_scores):
    header = [
        "Setting",
        "Noise",
        "Samples per environment",
        MODEL_TITLES["vae"],
        MODEL_TITLES["vae_u"],
        MODEL_TITLES["ivae"],
        "Best single variable",
        "iVAE minus baseline",
    ]

    rows = []
    for condition in selected_conditions(model_scores):
        sample_size, noise, weight_label = condition

        model_cells = [
            format_estimate(model_scores[condition][model_name])
            for model_name in MODEL_NAMES
        ]

        variable, baseline_mean = best_observed_variable(
            baseline_scores, condition
        )
        baseline_cell = (
            format_estimate(baseline_scores[condition][variable])
            + f" ({variable})"
        )
        difference = (
            statistics.mean(model_scores[condition]["ivae"]) - baseline_mean
        )

        rows.append(
            [
                weight_label,
                f"{noise:.2f}",
                sample_size,
                *model_cells,
                baseline_cell,
                f"{difference:+.{MEAN_DECIMALS}f}",
            ]
        )
    return header, rows


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    if UNCERTAINTY not in {"sd", "se"}:
        raise SystemExit('UNCERTAINTY must be either "sd" or "se".')
    for path in (RAW_RESULTS_FILE, BASELINE_FILE):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    model_scores = load_model_scores()
    baseline_scores = load_baseline_scores()
    header, rows = build_table(model_scores, baseline_scores)
    write_csv(OUTPUT_FILE, header, rows)

    sample_sizes = sorted({key[0] for key in selected_conditions(model_scores)})
    seed_counts = sorted({
        len(model_scores[condition][model_name])
        for condition in model_scores
        for model_name in MODEL_NAMES
    })
    print(f"{len(rows)} conditions")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Seeds per cell: {seed_counts}")
    print(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
