from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_dataset(
    n_per_environment=1000,
    seed=0,
    means=(-2.0, 0.0, 2.0),
    stds=(0.6, 1.0, 1.4),
    child_noise_std=0.35,
    outcome_noise_std=0.25,
    child_weights=(0.5, -0.3, 0.8),
    shuffle=True,
):
    """
        u -> x1 -> (x2, x3, x4) -> x5

    The model observes x = [x2, x3, x4, x5].
    The hidden ground-truth variable x1 is returned only for evaluation.

    Returns
    -------
    x:
        Observed variables, shape [N, 4].
    u:
        One-hot environment indicators, shape [N, num_environments].
    true_x1:
        Hidden ground-truth x1, shape [N].
    environment_labels:
        Integer environment labels, shape [N].
    """
    if n_per_environment <= 0:
        raise ValueError("n_per_environment must be positive.")
    if len(means) != len(stds):
        raise ValueError("means and stds must have the same length.")
    if len(means) < 2:
        raise ValueError("At least two environments are required.")
    if any(std <= 0 for std in stds):
        raise ValueError("All latent standard deviations must be positive.")
    if child_noise_std <= 0:
        raise ValueError("child_noise_std must be positive.")
    if outcome_noise_std <= 0:
        raise ValueError("outcome_noise_std must be positive.")
    if len(child_weights) != 3:
        raise ValueError("child_weights must have exactly three entries.")

    # A local generator means that this seed changes only this dataset.
    generator = torch.Generator().manual_seed(seed)

    means_tensor = torch.tensor(means, dtype=torch.float32)
    stds_tensor = torch.tensor(stds, dtype=torch.float32)
    num_environments = len(means)

    weight_x2, weight_x3, weight_x4 = (float(w) for w in child_weights)

    all_x = []
    all_x1 = []
    all_labels = []

    for environment in range(num_environments):
        # u -> x1
        # x1 | u=k ~ Normal(mean_k, std_k^2)
        x1_noise = torch.randn(
            n_per_environment,
            generator=generator,
        )
        x1 = (
            means_tensor[environment]
            + stds_tensor[environment] * x1_noise
        )

        # x1 -> x2, x3, x4
        noise_x2 = child_noise_std * torch.randn(
            n_per_environment,
            generator=generator,
        )
        noise_x3 = child_noise_std * torch.randn(
            n_per_environment,
            generator=generator,
        )
        noise_x4 = child_noise_std * torch.randn(
            n_per_environment,
            generator=generator,
        )

        x2 = weight_x2 * x1 + noise_x2
        x3 = weight_x3 * x1 + noise_x3
        x4 = weight_x4 * x1 + noise_x4

        # (x2, x3, x4) -> x5
        noise_x5 = outcome_noise_std * torch.randn(
            n_per_environment,
            generator=generator,
        )
        x5 = 0.8 * x2 - 0.6 * x3 + 0.5 * x4 + noise_x5

        # Every variable above has shape [n_per_environment].
        # stack(dim=1) produces [n_per_environment, 4].
        x_environment = torch.stack(
            [x2, x3, x4, x5],
            dim=1,
        )

        labels_environment = torch.full(
            (n_per_environment,),
            environment,
            dtype=torch.long,
        )

        all_x.append(x_environment)
        all_x1.append(x1)
        all_labels.append(labels_environment)

    x = torch.cat(all_x, dim=0)
    true_x1 = torch.cat(all_x1, dim=0)
    environment_labels = torch.cat(all_labels, dim=0)

    if shuffle:
        permutation = torch.randperm(
            len(x),
            generator=generator,
        )
        x = x[permutation]
        true_x1 = true_x1[permutation]
        environment_labels = environment_labels[permutation]

    u = F.one_hot(
        environment_labels,
        num_classes=num_environments,
    ).float()

    return x, u, true_x1, environment_labels
