import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from causdiff import DEVICE, SAVE_PATH, SEED
from causdiff.models.unetflow import FlowUNet
from causdiff.utils.flow_utils import integrate_flow, fd_dx, ad_dx


def main(args):
    """
    Compare the accuracy and performance of different integration methods
    and directional derivative computation techniques.
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving results to {SAVE_PATH}")

    model = FlowUNet(channels=args.channels, n_classes=1, time_emb_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    torch.manual_seed(SEED)
    z = torch.randn(
        args.batch_size, args.channels, args.img_size, args.img_size, device=DEVICE
    )

    t_steps = torch.linspace(0, 1, args.steps, device=DEVICE)

    print("\n=== Integration Method Comparison ===")
    start_time = time.time()
    x_euler = integrate_flow(model, z, t_steps, label=args.label, method="euler")
    euler_time = time.time() - start_time

    start_time = time.time()
    x_rk4 = integrate_flow(model, z, t_steps, label=args.label, method="rk4")
    rk4_time = time.time() - start_time

    integration_diff = torch.abs(x_euler - x_rk4).mean().item()

    print("Integration results:")
    print(f"  Euler integration time: {euler_time:.4f} seconds")
    print(f"  RK4 integration time: {rk4_time:.4f} seconds")
    print(f"  Mean absolute difference: {integration_diff:.6e}")
    print(f"  RK4 slowdown factor: {rk4_time / euler_time:.2f}x")

    print("\n=== Directional Derivative Method Comparison ===")
    torch.manual_seed(SEED)
    start_time = time.time()
    dx_fd, _, _, _ = fd_dx(model, z, t_steps, args.epsilon, label=args.label)
    fd_time = time.time() - start_time

    torch.manual_seed(SEED)
    start_time = time.time()
    dx_auto, _, _, _ = ad_dx(model, z, t_steps, args.epsilon, label=args.label)
    auto_time = time.time() - start_time

    derivative_diff = torch.abs(dx_fd - dx_auto).mean().item()

    print("Directional derivative results:")
    print(f"  Finite differences time: {fd_time:.4f} seconds")
    print(f"  Efficient autograd time: {auto_time:.4f} seconds")
    print(f"  Mean absolute difference: {derivative_diff:.6e}")
    print(
        f"  Speed comparison: {'Faster' if auto_time < fd_time else 'Slower'} by {abs(fd_time / auto_time - 1) * 100:.1f}%"
    )

    if args.batch_size > 0 and args.channels > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        sample_idx = 0
        channel_idx = 0

        im0 = axes[0, 0].imshow(
            dx_fd[sample_idx, channel_idx].detach().cpu(), cmap="viridis"
        )
        axes[0, 0].set_title("Finite Difference (dx/dz)")
        fig.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(
            dx_auto[sample_idx, channel_idx].detach().cpu(), cmap="viridis"
        )
        axes[0, 1].set_title("Autograd Method (dx/dz)")
        fig.colorbar(im1, ax=axes[0, 1])

        diff = dx_fd[sample_idx, channel_idx] - dx_auto[sample_idx, channel_idx]
        im2 = axes[1, 0].imshow(diff.detach().cpu(), cmap="coolwarm")
        axes[1, 0].set_title("Difference (FD - Autograd)")
        fig.colorbar(im2, ax=axes[1, 0])

        rel_diff = torch.zeros_like(diff)
        mask = torch.abs(dx_auto[sample_idx, channel_idx]) > 1e-6
        rel_diff[mask] = diff[mask] / dx_auto[sample_idx, channel_idx][mask]
        im3 = axes[1, 1].imshow(
            rel_diff.detach().cpu(), cmap="coolwarm", vmin=-0.1, vmax=0.1
        )
        axes[1, 1].set_title("Relative Difference")
        fig.colorbar(im3, ax=axes[1, 1])

        plt.suptitle(
            f"Comparison of Directional Derivative Methods (ε={args.epsilon})",
            fontsize=14,
        )
        plt.tight_layout()

        save_path = os.path.join(SAVE_PATH, "derivative_method_comparison.png")
        plt.savefig(save_path)
        print(f"\nVisual comparison saved to {save_path}")

        if not args.on_SUPR:
            plt.show()

    if args.epsilon_analysis:
        print("\n=== Epsilon Sensitivity Analysis ===")
        epsilon_values = np.logspace(-8, -2, 7)
        fd_errors = []
        auto_errors = []

        for eps in epsilon_values:
            torch.manual_seed(SEED)
            dx_fd, _, _, _ = fd_dx(model, z, t_steps, eps, label=args.label)

            torch.manual_seed(SEED)
            dx_auto, _, _, _ = ad_dx(model, z, t_steps, eps, label=args.label)

            fd_error = torch.abs(dx_fd - dx_auto).mean().item()
            fd_errors.append(fd_error)

            auto_errors.append(0)

        plt.figure(figsize=(10, 6))
        plt.loglog(epsilon_values, fd_errors, "o-", label="FD vs Autograd Difference")
        plt.grid(True, which="both", ls="--")
        plt.xlabel("Epsilon (ε)")
        plt.ylabel("Mean Absolute Difference")
        plt.title("Sensitivity of Finite Differences to Epsilon Value")
        plt.legend()

        plt.loglog(
            epsilon_values,
            [eps for eps in epsilon_values],
            "k--",
            alpha=0.5,
            label="O(ε)",
        )

        save_path = os.path.join(SAVE_PATH, "epsilon_sensitivity.png")
        plt.savefig(save_path)
        print(f"Epsilon sensitivity analysis saved to {save_path}")

        if not args.on_SUPR:
            plt.show()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare integration and directional derivative methods"
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of integration steps"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="result/mnist_digit/unet_mnist_digit_weights_final.pt",
        help="Path to pretrained FlowUNet model",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Perturbation scale for finite differences",
    )
    parser.add_argument(
        "--on_SUPR", type=bool, default=False, help="Whether running on SUPR cluster"
    )
    parser.add_argument(
        "--img_size", type=int, default=28, help="Image size (assuming square images)"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels (1 for grayscale, 3 for RGB)",
    )
    parser.add_argument(
        "--label", type=int, default=None, help="Label for conditional models"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for testing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--epsilon_analysis",
        type=bool,
        default=True,
        help="Run epsilon sensitivity analysis",
    )

    args = parser.parse_args()
    main(args)
