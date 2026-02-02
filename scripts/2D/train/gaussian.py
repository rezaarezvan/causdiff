import os
import torch

from causdiff import DEVICE, SAVE_PATH
from causdiff.models.MLPflow import MLP
from causdiff.utils.flow_utils import train_loop_cond
from causdiff.data.dataloaders import get_2D_gaussian_loader


def main(args):
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving results to {SAVE_PATH}")

    # (Hyper)Parameters
    N = 1000
    rho = 0.55

    # Target/True Distribution - p_t(x_t) ~ N(0, [1, rho; rho, 1])
    training_loader = get_2D_gaussian_loader(args.batch_size, N, rho, train=True)

    model = MLP(layers=10, inter_dim=1024).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters())

    train_loop_cond(model, optim, training_loader, args)

    torch.save(model.state_dict(), f"{SAVE_PATH}/mlp_{rho}_gaussian_weights_final.pt")
    print(f"Model saved at {SAVE_PATH}/mlp_{rho}_gaussian_weights_final.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--on_SUPR", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", type=bool, default=False)
    args = parser.parse_args()

    main(args)
