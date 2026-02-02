import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from causdiff import DEVICE, SAVE_PATH
from causdiff.models.unetflow import FlowUNet
from causdiff.utils.flow_utils import train_loop
from causdiff.data.dataloaders import get_celeba_loader


def main(args):
    if args.wandb:
        import wandb

        wandb.init(
            project="diffusion_experiment",
            name="celeba_unet_attention",
            config=args.__dict__,
        )

    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving results to {SAVE_PATH}")

    # (Hyper)Parameters
    img_size = 64
    training_loader = get_celeba_loader(
        img_size, batch_size=args.batch_size, train=True
    )

    print(f"Length of training loader: {len(training_loader)}")

    if args.compute_attr_corr:
        all_attrs = []
        pbar = tqdm(total=len(training_loader), desc="Loading attributes")
        for _, attrs in training_loader:
            all_attrs.append(attrs.cpu().numpy())
            pbar.update(1)
        pbar.close()

        all_attrs = np.concatenate(all_attrs, axis=0)
        attr_corr = np.corrcoef(all_attrs, rowvar=False)
        print(f"Attribute correlation matrix shape: {attr_corr.shape}")
        print(attr_corr)
        np.savetxt(f"{SAVE_PATH}/attr_corr.csv", attr_corr, delimiter=",")

        positions = np.arange(0, 40, 3)
        labels = positions + 1

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(attr_corr, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title("Ground truth attribute correlation matrix")

        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation coefficient", rotation=270, labelpad=15)

        fig.tight_layout()
        fig.savefig(f"{SAVE_PATH}/attr_corr.png", dpi=300)
        plt.close(fig)
        return

    model = FlowUNet(channels=3, n_classes=1, time_emb_dim=256).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loop(model, optim, training_loader, args)

    torch.save(
        model.state_dict(), f"{SAVE_PATH}/unet_attention_celeba_weights_final.pt"
    )
    print(f"Model saved at {SAVE_PATH}/unet_attention_celeba_weights_final.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400_001)
    parser.add_argument("--on_SUPR", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--compute_attr_corr", type=bool, default=False)
    args = parser.parse_args()

    main(args)
