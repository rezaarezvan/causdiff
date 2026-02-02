import os
import torch

from causdiff import DEVICE, SAVE_PATH
from causdiff.models.unetflow import FlowUNet
from causdiff.utils.flow_utils import train_loop
from causdiff.data.dataloaders import get_mnist_loader_digit


def main(args):
    if args.wandb:
        import wandb

        wandb.init(
            project="diffusion_experiment",
            name="mnist_digit_unet",
            config=args.__dict__,
        )

    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving results to {SAVE_PATH}")

    # (Hyper)Parameters
    digit = 5
    training_loader = get_mnist_loader_digit(
        batch_size=args.batch_size, digit=digit, train=True, SUPR=args.on_SUPR
    )

    model = FlowUNet(channels=1, n_classes=1, time_emb_dim=256).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loop(model, optim, training_loader, args, checkpoint_interval=2000)

    torch.save(
        model.state_dict(),
        f"{SAVE_PATH}/unet_mnist_digit_weights_final.pt",
    )
    print(f"Model saved at {SAVE_PATH}/unet_mnist_digit_weights_final.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_001)
    parser.add_argument("--on_SUPR", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", type=bool, default=False)
    args = parser.parse_args()

    main(args)
