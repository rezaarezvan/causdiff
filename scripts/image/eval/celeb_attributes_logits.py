import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import models
from causdiff import DEVICE, SAVE_PATH
from causdiff.models.unetflow import FlowUNet
from causdiff.utils.flow_utils import sample_batch_generation


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving results to {SAVE_PATH}")
    torch.manual_seed(42)
    np.random.seed(42)
    # (Hyper)Parameters
    img_size = 64
    flow_model = FlowUNet(channels=3, n_classes=1, time_emb_dim=256).to(DEVICE)

    # Load model (final saved model)
    flow_model.load_state_dict(
        torch.load(args.flow_model, map_location=DEVICE)
    )
    print(f"Flow model loaded from {args.flow_model}")

    # Load discriminator (classifier) from checkpoint
    checkpoint = torch.load(args.classifier_ckpt, map_location=DEVICE)
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = torch.nn.Linear(num_features, 40)
    classifier.load_state_dict(checkpoint["model"])
    classifier.to(DEVICE)
    print(f"Classifier loaded from {args.classifier_ckpt}")

    all_probs = []
    all_probs_pert = []
    classifier.eval()
    with torch.no_grad():
        for iteration in tqdm(range(args.iterations)):
            xt, xt_perturbed, dx = sample_batch_generation(
                flow_model,
                args,
                img_size,
                3,
                args.batch_size,
                plot=False,
                compute_perturbations=True,
            )
            y = classifier(xt)
            y_perturbed = classifier(xt_perturbed)

            all_probs.append(y.cpu().numpy())
            all_probs_pert.append(y_perturbed.cpu().numpy())
            # Convert logits to probabilities
            # sigmoid_y = torch.sigmoid(y)
            # sigmoid_y_pert = torch.sigmoid(y_perturbed)

            # all_probs.append(sigmoid_y.cpu().numpy())
            # all_probs_pert.append(sigmoid_y_pert.cpu().numpy())

    probs_original = np.concatenate(all_probs, axis=0)
    probs_perturbed = np.concatenate(all_probs_pert, axis=0)
    diff_probs = probs_perturbed - probs_original

    # Compute correlation matrices
    corr_original = np.corrcoef(probs_original, rowvar=False)
    corr_perturbed = np.corrcoef(probs_perturbed, rowvar=False)
    corr_diff = np.corrcoef(diff_probs, rowvar=False)

    # Save correlation matrices to CSV
    np.savetxt(f"{SAVE_PATH}/corr_original.csv", corr_original, delimiter=",")
    np.savetxt(f"{SAVE_PATH}/corr_perturbed.csv", corr_perturbed, delimiter=",")
    np.savetxt(f"{SAVE_PATH}/corr_diff.csv", corr_diff, delimiter=",")
    print(f"Correlation matrices saved to {SAVE_PATH}")

    frobenius_norm = np.linalg.norm(diff_probs, ord="fro")
    print(f"Frobenius norm of difference: {frobenius_norm:.6f}")

    # Plot the correlation matrices
    _, ax = plt.subplots(1, 3, figsize=(15, 15))
    xs = np.arange(1, 41, 5)
    ys = np.arange(1, 41, 5)
    ax[0].imshow(corr_original, cmap="viridis", vmin=-1, vmax=1)
    ax[0].set_title("Original correlation matrix")
    ax[0].set_xticks(xs)
    ax[0].set_yticks(ys)
    ax[1].imshow(corr_perturbed, cmap="viridis", vmin=-1, vmax=1)
    ax[1].set_title("Perturbed correlation matrix")
    ax[1].set_xticks(xs)
    ax[1].set_yticks(ys)
    ax[2].imshow(corr_diff, cmap="viridis", vmin=-1, vmax=1)
    ax[2].set_title("Difference correlation matrix")
    ax[2].set_xticks(xs)
    ax[2].set_yticks(ys)
    plt.suptitle(
        f"Correlation matrices with epsilon={args.epsilon} and Frobenius norm={
            frobenius_norm:.6f
        }"
    )
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}/correlation_matrices.png")
    print(f"Correlation matrices plot saved to {SAVE_PATH}")

    # Now we condition on 'Heavy Makeup' (idx=18), only select predictions where
    # the difference in 'Heavy Makeup' is small (meaning that the attribute is not
    # changed much by the perturbation)
    idx_heavy_makeup = 18
    threshold = 1e-4
    diff_probs_heavy_makeup = np.where(
        np.abs(diff_probs[:, idx_heavy_makeup]) < threshold
    )
    print(
        f"Number of samples with |diff| < {threshold} for 'Heavy Makeup': {
            len(diff_probs_heavy_makeup[0])
        }"
    )

    # Recompute correlation matrices for conditioned subset
    corr_original_heavy_makeup = np.corrcoef(
        probs_original[diff_probs_heavy_makeup], rowvar=False
    )
    corr_perturbed_heavy_makeup = np.corrcoef(
        probs_perturbed[diff_probs_heavy_makeup], rowvar=False
    )
    corr_diff_heavy_makeup = np.corrcoef(
        diff_probs[diff_probs_heavy_makeup], rowvar=False
    )

    # Save conditioned correlation matrices
    np.savetxt(
        f"{SAVE_PATH}/corr_original_heavy_makeup.csv",
        corr_original_heavy_makeup,
        delimiter=",",
    )
    np.savetxt(
        f"{SAVE_PATH}/corr_perturbed_heavy_makeup.csv",
        corr_perturbed_heavy_makeup,
        delimiter=",",
    )
    np.savetxt(
        f"{SAVE_PATH}/corr_diff_heavy_makeup.csv", corr_diff_heavy_makeup, delimiter=","
    )
    print(f"Correlation matrices for 'Heavy Makeup' saved to {SAVE_PATH}")

    # Plot conditioned correlation matrices
    _, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(corr_original_heavy_makeup, cmap="viridis", vmin=-1, vmax=1)
    ax[0].set_title("Original corr. ('Heavy Makeup')")
    ax[0].set_xticks(xs)
    ax[0].set_yticks(ys)
    ax[1].imshow(corr_perturbed_heavy_makeup, cmap="viridis", vmin=-1, vmax=1)
    ax[1].set_title("Perturbed corr. ('Heavy Makeup')")
    ax[1].set_xticks(xs)
    ax[1].set_yticks(ys)
    ax[2].imshow(corr_diff_heavy_makeup, cmap="viridis", vmin=-1, vmax=1)
    ax[2].set_title("Difference corr. ('Heavy Makeup')")
    ax[2].set_xticks(xs)
    ax[2].set_yticks(ys)
    plt.suptitle(
        f"Conditioned corr. matrices for 'Heavy Makeup' (epsilon={args.epsilon:.6f})"
    )
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}/correlation_matrices_heavy_makeup.png")
    print(f"Correlation matrices plot for 'Heavy Makeup' saved to {SAVE_PATH}")

    # Compute inverse (precision) matrices for the full correlation matrices
    precision_original = np.linalg.inv(corr_original)
    precision_dz = np.linalg.inv(corr_diff)
    precision_controlled_wiggle = np.linalg.inv(corr_diff_heavy_makeup)

    # NOTE: Adjust indices x-1 by 0-index
    idx_rosyCheeks = 29  # if id 30 corresponds to rosyCheeks
    idx_pointNose = 27  # -//-

    # Extract values from the correlation matrices
    corr_orig_val = corr_original[idx_rosyCheeks, idx_pointNose]
    corr_diff_val = corr_diff[idx_rosyCheeks, idx_pointNose]
    corr_diff_control_val = corr_diff_heavy_makeup[idx_rosyCheeks, idx_pointNose]

    # Extract values from the precision matrices
    precision_orig_val = precision_original[idx_rosyCheeks, idx_pointNose]
    precision_wiggle_val = precision_dz[idx_rosyCheeks, idx_pointNose]
    precision_controlled_wiggle_val = precision_controlled_wiggle[
        idx_rosyCheeks, idx_pointNose
    ]
    print("\n=== Correlation Values ===")
    print(f"Original correlation (rosyCheeks, pointNose):  {corr_orig_val: .6f}")
    print(f"Wiggle correlation (rosyCheeks, pointNose):  {corr_diff_val: .6f}")
    print(
        f"Controlled wiggle correlation (rosyCheeks, pointNose): {
            corr_diff_control_val: .6f
        }"
    )

    print("\n=== Precision Values ===")
    print(f"Original precision (rosyCheeks, pointNose):  {precision_orig_val: .6f}")
    print(f"Wiggle precision (rosyCheeks, pointNose):  {precision_wiggle_val: .6f}")
    print(
        f"Controlled wiggle precision (rosyCheeks, pointNose): {
            precision_controlled_wiggle_val: .6f
        }"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument(
        "--epsilon", type=float, default=1e-2, help="Perturbation scale"
    )
    parser.add_argument("--on_SUPR", type=bool, default=False)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for generation"
    )
    parser.add_argument(
        "--flow_model",
        type=str,
        default="result/celeba/unet_celeba_weights_final.pt",
        help="Path to FlowUNet weights",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default="result/ResNet_Attributes_classifier_celebA/ckpt_step_40.pt",
        help="Path to attribute classifier checkpoint",
    )
    args = parser.parse_args()

    main(args)
