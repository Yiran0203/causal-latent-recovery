from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dgp import generate_dataset
from model import GaussianLatentModel, VAE, VAEWithU, iVAE


# ============================================================
# Experiment configuration
# ============================================================

SAMPLE_SIZES_PER_ENVIRONMENT = [100, 500, 1000, 2000]
SEEDS = list(range(10))
MODEL_NAMES = ["vae", "vae_u", "ivae"]

CHILD_NOISE_STD = [0.35, 0.75, 1.60, 3.50, 5.00]
CHILD_WEIGHT = {
    "A": (0.50, -0.30, 0.80),
    "B": (0.50, -0.30, 0.20),
    "C": (0.50,  0.30, 0.20),
    "D": (0.50,  0.50, 0.50),
}

NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

X_DIM = 4       # [x2, x3, x4, x5]
U_DIM = 3       # three environments
Z_DIM = 1       # hidden x1
HIDDEN_DIM = 50
DECODER_STD = 0.1

RESULTS_DIRECTORY = Path("runs/full")
RAW_RESULTS_FILE = RESULTS_DIRECTORY / "g2_model_comparison_raw.csv"
SUMMARY_FILE = RESULTS_DIRECTORY / "g2_model_comparison_summary.csv"
BASELINE_FILE = RESULTS_DIRECTORY / "g2_observed_baselines.csv"


# ============================================================
# Device and reproducibility
# ============================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch model randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data preparation
# ============================================================


def standardize_observations(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize each observed variable to mean 0 and SD 1."""
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized_x = (x - x_mean) / x_std
    return standardized_x, x_mean, x_std


def make_dataloader(
    x: torch.Tensor,
    u: torch.Tensor,
    batch_size: int,
    loader_seed: int,
) -> DataLoader:
    """
    Create a new DataLoader for one model.

    Reusing the same loader_seed gives all three models the same
    initial shuffled sample order for a given dataset and simulation seed.
    """
    dataset = TensorDataset(x, u)
    generator = torch.Generator().manual_seed(loader_seed)

    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )


# ============================================================
# Models
# ============================================================


def create_model(model_name: str) -> GaussianLatentModel:
    common_arguments = {
        "x_dim": X_DIM,
        "u_dim": U_DIM,
        "z_dim": Z_DIM,
        "hidden_dim": HIDDEN_DIM,
        "decoder_std": DECODER_STD,
    }

    if model_name == "vae":
        return VAE(**common_arguments)
    if model_name == "vae_u":
        return VAEWithU(**common_arguments)
    if model_name == "ivae":
        return iVAE(**common_arguments)

    raise ValueError(f"Unknown model name: {model_name!r}")


# ============================================================
# Recovery metrics
# ============================================================


def pearson_correlation(
    estimated: torch.Tensor,
    truth: torch.Tensor,
) -> float:
    """Calculate signed Pearson correlation on CPU."""
    estimated = estimated.detach().cpu().flatten()
    truth = truth.detach().cpu().flatten()

    if estimated.numel() != truth.numel():
        raise ValueError("estimated and truth must have the same length.")
    if estimated.numel() < 2:
        return float("nan")
    if estimated.std() < 1e-8 or truth.std() < 1e-8:
        return float("nan")

    correlation = torch.corrcoef(
        torch.stack([estimated, truth])
    )[0, 1]
    return correlation.item()


def evaluate_recovery(
    model: GaussianLatentModel,
    x: torch.Tensor,
    u: torch.Tensor,
    true_x1: torch.Tensor,
    environment_labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """
    Recover x1 using the encoder posterior mean.

    Since z_dim=1, MCC equals absolute Pearson correlation.
    """
    model.eval()

    with torch.no_grad():
        posterior_mean, _ = model.encoder_params(
            x.to(device),
            u.to(device),
        )

    estimated_x1 = posterior_mean[:, 0].cpu()
    pooled_pearson = pearson_correlation(
        estimated_x1,
        true_x1,
    )

    results: dict[str, float] = {
        "pearson": pooled_pearson,
        "mcc": abs(pooled_pearson),
    }

    for environment in range(U_DIM):
        mask = environment_labels == environment
        environment_pearson = pearson_correlation(
            estimated_x1[mask],
            true_x1[mask],
        )
        results[f"env_{environment}_pearson"] = environment_pearson
        results[f"env_{environment}_mcc"] = abs(environment_pearson)

    return results


def observed_variable_baselines(
    x: torch.Tensor,
    true_x1: torch.Tensor,
) -> dict[str, float]:
    """Measure how strongly each observed variable alone reveals x1."""
    names = ["x2", "x3", "x4", "x5"]
    results: dict[str, float] = {}

    for column, name in enumerate(names):
        correlation = pearson_correlation(
            x[:, column],
            true_x1,
        )
        results[f"{name}_pearson"] = correlation
        results[f"{name}_absolute_pearson"] = abs(correlation)

    return results


# ============================================================
# Training
# ============================================================


def train_model(
    model: GaussianLatentModel,
    dataloader: DataLoader,
    device: torch.device,
    number_of_epochs: int,
    learning_rate: float,
) -> float:
    """Train one model and return the final epoch's mean loss."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    final_average_loss = float("nan")

    for _epoch in range(1, number_of_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x_batch, u_batch in dataloader:
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)

            # elbo() returns one value per sample; average it before backward().
            elbo = model.elbo(x_batch, u_batch).mean()
            loss = -elbo

            if not torch.isfinite(loss).item():
                raise RuntimeError("The loss became NaN or infinite.")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            current_batch_size = x_batch.shape[0]
            total_loss += loss.detach().item() * current_batch_size
            total_samples += current_batch_size

        final_average_loss = total_loss / total_samples

    return final_average_loss


# ============================================================
# One paired simulation condition
# ============================================================


def run_paired_simulation(
    n_per_environment: int,
    seed: int,
    device: torch.device,
    child_noise_std: float = 0.35,
    weight_label: str = "A",
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, float | int],
]:
    """
    Generate one dataset, then train all three models on that same dataset.

    This paired design prevents model comparisons from being confounded by
    different simulated datasets.
    """
    data_seed = seed
    model_seed = 10_000 + seed
    loader_seed = 20_000 + seed
    child_weights = CHILD_WEIGHT[weight_label]

    x_raw, u, true_x1, environment_labels = generate_dataset(
        n_per_environment=n_per_environment,
        seed=data_seed,
        child_noise_std=child_noise_std,
        child_weights=child_weights,
    )

    # The same standardised x is supplied to every model.
    x, _, _ = standardize_observations(x_raw)

    baseline_row: dict[str, float | int | str] = {
        "n_per_environment": n_per_environment,
        "child_noise_std": child_noise_std,
        "weight_label": weight_label,
        "child_weights": str(child_weights),
        "total_samples": len(x),
        "seed": seed,
        **observed_variable_baselines(x, true_x1),
    }

    model_results: list[dict[str, float | int | str]] = []

    for model_name in MODEL_NAMES:
        # Reset to the same base model seed before each model is constructed.
        # Architectures differ, so parameters are not identical, but each
        # condition remains deterministic and paired by simulation seed.
        set_random_seed(model_seed)

        model = create_model(model_name).to(device)

        # Construct a fresh DataLoader with the same shuffle seed for each model.
        dataloader = make_dataloader(
            x=x,
            u=u,
            batch_size=BATCH_SIZE,
            loader_seed=loader_seed,
        )

        final_loss = train_model(
            model=model,
            dataloader=dataloader,
            device=device,
            number_of_epochs=NUMBER_OF_EPOCHS,
            learning_rate=LEARNING_RATE,
        )

        recovery = evaluate_recovery(
            model=model,
            x=x,
            u=u,
            true_x1=true_x1,
            environment_labels=environment_labels,
            device=device,
        )

        model_results.append(
            {
                "model": model_name,
                "n_per_environment": n_per_environment,
                "child_noise_std": child_noise_std,
                "weight_label": weight_label,
                "child_weights": str(child_weights),
                "total_samples": len(x),
                "seed": seed,
                "data_seed": data_seed,
                "model_seed": model_seed,
                "loader_seed": loader_seed,
                "batch_size": min(BATCH_SIZE, len(x)),
                "epochs": NUMBER_OF_EPOCHS,
                "final_loss": final_loss,
                **recovery,
            }
        )

        # Helpful when many models are trained sequentially on Apple Silicon.
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    return model_results, baseline_row


# ============================================================
# Saving and summarising
# ============================================================


def save_rows(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
    append: bool = False,
) -> None:
    """Write rows to CSV. With append=True, add to an existing file and
    write the header only when creating it."""
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not (append and output_path.exists())
    mode = "a" if append else "w"

    with output_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def completed_conditions() -> set[tuple[int, float, str]]:
    """Conditions already present in the raw results file, so that an
    interrupted run can be restarted without repeating work."""
    if not RAW_RESULTS_FILE.exists():
        return set()

    done: set[tuple[int, float, str]] = set()
    with RAW_RESULTS_FILE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            done.add(
                (
                    int(row["n_per_environment"]),
                    float(row["child_noise_std"]),
                    row["weight_label"],
                )
            )
    return done


def load_all_results() -> list[dict[str, float | int | str]]:
    """Read back every raw result row, for the end-of-run summary."""
    if not RAW_RESULTS_FILE.exists():
        return []

    numeric = {
        "n_per_environment", "child_noise_std", "seed", "total_samples",
        "mcc", "pearson", "final_loss",
    }
    rows: list[dict[str, float | int | str]] = []
    with RAW_RESULTS_FILE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            parsed: dict[str, float | int | str] = {}
            for key, value in row.items():
                if key in numeric or key.startswith("env_"):
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
                else:
                    parsed[key] = value
            parsed["n_per_environment"] = int(float(parsed["n_per_environment"]))
            rows.append(parsed)
    return rows


def finite_values(values: list[float]) -> list[float]:
    """Remove NaN and infinite values before summary statistics."""
    return [value for value in values if math.isfinite(value)]


def sweep_grid() -> list[tuple[int, float, str]]:
    """Every condition in the sweep, in the order they are run."""
    return [
        (n_per_environment, child_noise_std, weight_label)
        for n_per_environment in SAMPLE_SIZES_PER_ENVIRONMENT
        for child_noise_std in CHILD_NOISE_STD
        for weight_label in CHILD_WEIGHT
    ]


def summarize_results(
    results: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []

    for n_per_environment, child_noise_std, weight_label in sweep_grid():
        for model_name in MODEL_NAMES:
            condition_results = [
                result
                for result in results
                if result["n_per_environment"] == n_per_environment
                and result["child_noise_std"] == child_noise_std
                and result["weight_label"] == weight_label
                and result["model"] == model_name
            ]

            mcc_values = finite_values(
                [float(result["mcc"]) for result in condition_results]
            )

            if not mcc_values:
                raise RuntimeError(
                    f"No finite MCC values for {model_name}, "
                    f"n={n_per_environment}, noise={child_noise_std}, "
                    f"weights={weight_label}."
                )

            summary: dict[str, float | int | str] = {
                "model": model_name,
                "n_per_environment": n_per_environment,
                "child_noise_std": child_noise_std,
                "weight_label": weight_label,
                "total_samples": n_per_environment * U_DIM,
                "number_of_seeds": len(mcc_values),
                "mean_mcc": statistics.mean(mcc_values),
                "sd_mcc": (
                    statistics.stdev(mcc_values)
                    if len(mcc_values) > 1
                    else 0.0
                ),
                "min_mcc": min(mcc_values),
                "max_mcc": max(mcc_values),
            }

            for environment in range(U_DIM):
                environment_values = finite_values(
                    [
                        float(result[f"env_{environment}_mcc"])
                        for result in condition_results
                    ]
                )
                summary[f"env_{environment}_mean_mcc"] = (
                    statistics.mean(environment_values)
                )
                summary[f"env_{environment}_sd_mcc"] = (
                    statistics.stdev(environment_values)
                    if len(environment_values) > 1
                    else 0.0
                )

            summaries.append(summary)

    return summaries


# ============================================================
# Main experiment
# ============================================================


def main() -> None:
    device = get_device()

    print("Using device:", device)
    print("Models:", MODEL_NAMES)
    print("Sample sizes per environment:", SAMPLE_SIZES_PER_ENVIRONMENT)
    print("Child noise levels:", CHILD_NOISE_STD)
    print("Child weight:", dict(CHILD_WEIGHT))
    print("Seeds:", SEEDS)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUMBER_OF_EPOCHS)
    print()

    all_results: list[dict[str, float | int | str]] = []
    all_baselines: list[dict[str, float | int | str]] = []

    grid = sweep_grid()
    done = completed_conditions()
    if done:
        print(f"Resuming: {len(done)} of {len(grid)} conditions already in "
              f"{RAW_RESULTS_FILE}")
        print()
    print(f"Conditions: {len(grid)}")
    print(f"Training runs: {len(grid) * len(SEEDS) * len(MODEL_NAMES)}")
    print()

    for n_per_environment, child_noise_std, weight_label in grid:
        if (n_per_environment, child_noise_std, weight_label) in done:
            continue

        condition_results: list[dict[str, float | int | str]] = []
        condition_baselines: list[dict[str, float | int | str]] = []

        print(
            f"N={n_per_environment} | "
            f"noise={child_noise_std:.2f} | "
            f"weights={weight_label} "
            f"{CHILD_WEIGHT[weight_label]}"
        )

        for seed in SEEDS:
            paired_results, baseline_row = run_paired_simulation(
                n_per_environment=n_per_environment,
                seed=seed,
                device=device,
                child_noise_std=child_noise_std,
                weight_label=weight_label,
            )

            all_results.extend(paired_results)
            all_baselines.append(baseline_row)
            condition_results.extend(paired_results)
            condition_baselines.append(baseline_row)

            for result in paired_results:
                print(
                    f"  Seed {seed:02d} | "
                    f"{str(result['model']):6s} | "
                    f"loss={float(result['final_loss']):8.4f} | "
                    f"Pearson={float(result['pearson']):+.4f} | "
                    f"MCC={float(result['mcc']):.4f}"
                )

        # Written after every condition, so an interrupted run keeps
        # everything finished so far and can be restarted.
        save_rows(condition_results, RAW_RESULTS_FILE, append=True)
        save_rows(condition_baselines, BASELINE_FILE, append=True)

        print("  Condition summary")
        for model_name in MODEL_NAMES:
            values = [
                float(result["mcc"])
                for result in all_results
                if result["n_per_environment"] == n_per_environment
                and result["child_noise_std"] == child_noise_std
                and result["weight_label"] == weight_label
                and result["model"] == model_name
            ]
            sd_value = (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            )
            print(
                f"    {model_name:6s}: "
                f"MCC={statistics.mean(values):.4f} "
                f"± {sd_value:.4f}"
            )
        print()

    # Read back from disk so that a resumed run summarises every condition,
    # not just the ones completed in this session.
    summaries = summarize_results(load_all_results())
    save_rows(summaries, SUMMARY_FILE)

    print("Final summary")
    print("-------------")
    for summary in summaries:
        print(
            f"N={int(summary['n_per_environment']):5d} | "
            f"noise={float(summary['child_noise_std']):.2f} | "
            f"{str(summary['weight_label']):9s} | "
            f"{str(summary['model']):6s} | "
            f"MCC={float(summary['mean_mcc']):.4f} "
            f"± {float(summary['sd_mcc']):.4f}"
        )

    print()
    print("Raw model results:", RAW_RESULTS_FILE)
    print("Summary results:", SUMMARY_FILE)
    print("Observed baselines:", BASELINE_FILE)


if __name__ == "__main__":
    main()
