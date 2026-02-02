import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from causdiff import DEVICE, SAVE_PATH
from causdiff.models.unetflow import FlowUNet
from causdiff.data.dataloaders import get_mnist_loader_digit
from causdiff.utils.image_utils import plot_scatter
from causdiff.utils.flow_utils import sample_batch_generation


def main(args):
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving results to {SAVE_PATH}")

    dataset = get_mnist_loader_digit(batch_size=args.batch_size, digit=5, train=False)
    target, _ = next(iter(dataset))
    target = target.to(DEVICE)

    model = FlowUNet(channels=1, n_classes=1, time_emb_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval().requires_grad_(False)

    x1, y1 = 14, 8
    x2, y2 = 15, 8

    # Compute dx/dz over iterations
    directions = []
    for _ in tqdm(
        range(0, args.iterations, args.batch_size), desc="Integration Iterations"
    ):
        xt, xt_perturbed, dx = sample_batch_generation(
            model,
            args,
            28,
            1,
            args.batch_size,
            label=0,
            plot=False,
            compute_perturbations=True,
        )
        p1 = dx[:, 0, y1, x1].cpu()
        p2 = dx[:, 0, y2, x2].cpu()
        directions.append(torch.stack((p1, p2), dim=1))

    directions_tensor = torch.cat(directions, dim=0)
    plot_scatter(
        directions_tensor[:, 0],
        directions_tensor[:, 1],
        f"dx/dz at pixel ({x1},{y1})",
        f"dx/dz at pixel ({x2},{y2})",
        "Finite Differences at Two Pixels",
        plot=False,
    )
    plt.savefig(f"{SAVE_PATH}/dx_dz_scatter.png", dpi=300)
    print(f"Saved dx/dz scatter plot to {SAVE_PATH}/dx_dz_scatter.png")

    # Plot last dx image
    last_dx = dx[-1, 0].detach().cpu()
    plt.figure(figsize=(4, 4))
    plt.imshow(last_dx, cmap="gray")
    plt.axis("off")
    plt.title("dx/dz Derivative Image (Last Iteration)")
    plt.savefig(f"{SAVE_PATH}/dx_dz_last_iteration.png", dpi=300)
    print(f"Saved dx/dz last iteration image to {SAVE_PATH}/dx_dz_last_iteration.png")

    # Statistics
    mean_vals = directions_tensor.mean(dim=0)
    print("Mean of directions:", mean_vals)
    corrcoef = torch.corrcoef(directions_tensor.T)
    print("Correlation coefficient matrix:\n", corrcoef)

    # True correlations from dataset
    all_pixel_values = []
    for data, _ in tqdm(dataset, desc="Computing True Pixel Correlations"):
        data = data.to(DEVICE)
        p1 = data[:, 0, y1, x1].cpu().tolist()
        p2 = data[:, 0, y2, x2].cpu().tolist()
        all_pixel_values.extend(zip(p1, p2))

    all_pixels_tensor = torch.tensor(all_pixel_values, dtype=torch.float)
    true_corrcoef = torch.corrcoef(all_pixels_tensor.T)
    print(
        "True correlation coefficient matrix for MNIST fives images:\n", true_corrcoef
    )
    plot_scatter(
        all_pixels_tensor[:, 0],
        all_pixels_tensor[:, 1],
        f"Pixel intensity at ({x1},{y1})",
        f"Pixel intensity at ({x2},{y2})",
        "Joint Distribution of Pixel Intensities in MNIST fives",
        plot=False,
    )
    plt.savefig(f"{SAVE_PATH}/joint_distribution_mnist_fives.png", dpi=300)
    print(
        f"Saved joint distribution plot to {
            SAVE_PATH
        }/joint_distribution_mnist_fives.png"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="result/mnist_digit/unet_mnist_digit_weights_final.pt",
    )
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--on_SUPR", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=32 * 10)
    args = parser.parse_args()
    main(args)
