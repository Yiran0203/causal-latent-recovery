"""Short CPU run to check all three training paths; not a benchmark."""
import torch
import train

def main():
    train.NUMBER_OF_EPOCHS = 2
    train.BATCH_SIZE = 32
    rows, baseline = train.run_paired_simulation(
        n_per_environment=32, seed=0, device=torch.device("cpu"),
        child_noise_std=0.75, weight_label="A",
    )
    assert {r["model"] for r in rows} == {"vae", "vae_u", "ivae"}
    assert all(0 <= r["mcc"] <= 1 and torch.isfinite(torch.tensor(r["final_loss"])) for r in rows)
    for row in rows:
        print(f"{row['model']:6} loss={row['final_loss']:.4f} MCC={row['mcc']:.4f}")
    print("Passed: three models trained and produced finite recovery metrics.")

if __name__ == "__main__":
    main()
