import os
import torch
import imageio
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.distributions as dists

from PIL import Image
from causdiff import DEVICE
from torchvision.utils import make_grid


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat([torch.cat([i for i in images.cpu()], dim=-1)], dim=-2)
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def create_gif(
    images_path,
    gif_name,
    start,
    end,
    prefix,
    gif_path,
    file_extension="png",
    duration=0.01,
):
    """
    Generates GIF {gif_name}.gif from images at {images_path} namned '{prefix}{start}_of_{end}.
    GIF is saved in gif_path.
    """
    images = []
    for i in range(start, end + 1, 3):
        filename = f"{prefix}{i}.{file_extension}"
        file_path = os.path.join(images_path, filename)
        images.append(imageio.imread(file_path))

    output_path = os.path.join(gif_path, f"{gif_name}.gif")

    imageio.mimsave(output_path, images, duration=duration)


def plot_scatter(x_data, y_data, xlabel, ylabel, title, plot=True):
    """
    Generic scatter plot function.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(x_data, y_data, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show() if plot else None


def plot_regions(image, region1, region2, title="Marked Regions"):
    """Plot an image with highlighted regions."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    rect1_params = (
        region1[1].start,
        region1[0].start,
        region1[1].stop - region1[1].start,
        region1[0].stop - region1[0].start,
    )
    rect2_params = (
        region2[1].start,
        region2[0].start,
        region2[1].stop - region2[1].start,
        region2[0].stop - region2[0].start,
    )
    rect1 = patches.Rectangle(
        (rect1_params[0], rect1_params[1]),
        rect1_params[2],
        rect1_params[3],
        edgecolor="red",
        facecolor="none",
        linewidth=1,
    )
    rect2 = patches.Rectangle(
        (rect2_params[0], rect2_params[1]),
        rect2_params[2],
        rect2_params[3],
        edgecolor="blue",
        facecolor="none",
        linewidth=1,
    )
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.axis("off")
    ax.set_title(title)
    plt.show()


def plot_side_by_side(images, images_pert, images_dz, steps, images_to_generate):
    """Plot original, perturbed, and dz images side by side. Works for both grayscale and RGB images."""
    cols = images_to_generate
    fig, axes = plt.subplots(3, cols, figsize=(cols * 2, 4))
    if cols == 1:
        axes = axes.reshape(3, 1)

    for j in range(cols):
        img = images[j][0] if isinstance(images[j], list) else images[j]
        img_pert = (
            images_pert[j][0] if isinstance(images_pert[j], list) else images_pert[j]
        )
        img_dz = images_dz[j][0] if isinstance(images_dz[j], list) else images_dz[j]

        img_set = [img, img_pert, img_dz]
        titles = [
            f"Step {int(steps - (images_to_generate - (j + 1)))}/{steps}",
            "Perturbed image",
            "Perturbation (dx)",
        ]
        for i, (img_data, title) in enumerate(zip(img_set, titles)):
            ax = axes[i, j]
            # Check if the image is grayscale (2D) or RGB (3 channels).
            if img_data.ndim == 2 or (img_data.ndim == 3 and img_data.shape[-1] == 1):
                ax.imshow(img_data, cmap="gray")
            else:
                if img_data.ndim == 3 and img_data.shape[0] == 3:  # (C, H, W)
                    img_data = img_data.permute(1, 2, 0)
                elif img_data.ndim == 3 and img_data.shape[-1] == 3:
                    img_data = img_data
                else:
                    raise ValueError(
                        f"Unexpected shape for image data: {img_data.shape}"
                    )
                ax.imshow(img_data)
            ax.axis("off")
            ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_image_grid(images, nrow, save_path=None, filename=None, channels=1):
    """Plot or save a grid of images."""
    grid = make_grid(images, nrow=nrow, normalize=True)
    plt.figure()
    if channels == 1:
        plt.imshow(grid.cpu().permute(1, 2, 0).squeeze(2), cmap="gray")
    else:
        plt.imshow(grid.cpu().permute(1, 2, 0).clamp(0, 1))
    plt.axis("off")
    if save_path and filename:
        plt.savefig(os.path.join(save_path, filename))
    plt.show()
    plt.close()


def plot_batch_of_images_with_prediction_and_label(
    images, amount_of_images, predictions, targets, epoch, save_dir
):
    """Send images as (B, C, H, W), images need to be a factor of two."""
    images = images.detach().cpu()
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()

    num_images = min(amount_of_images, images.shape[0])
    _, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()

        # Undo normalization from
        img = img * 0.5 + 0.5
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"P: {predictions[i].item()}\nT: {targets[i].item()}", fontsize=8
        )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def plot_image_with_prediction_vector(images, predictions, targets, epoch, save_dir):
    celeba_attributes = [
        "5 o'clock Shadow",
        "Arched Eyebrows",
        "Attractive",
        "Bags Under Eyes",
        "Bald",
        "Bangs",
        "Big Lips",
        "Big Nose",
        "Black Hair",
        "Blond Hair",
        "Blurry",
        "Brown Hair",
        "Bushy Eyebrows",
        "Chubby",
        "Double Chin",
        "Eyeglasses",
        "Goatee",
        "Gray Hair",
        "Heavy Makeup",
        "High Cheekbones",
        "Male",
        "Mouth Slightly Open",
        "Mustache",
        "Narrow Eyes",
        "No Beard",
        "Oval Face",
        "Pale Skin",
        "Pointy Nose",
        "Receding Hairline",
        "Rosy Cheeks",
        "Sideburns",
        "Smiling",
        "Straight Hair",
        "Wavy Hair",
        "Wearing Earrings",
        "Wearing Hat",
        "Wearing Lipstick",
        "Wearing Necklace",
        "Wearing Necktie",
        "Young",
    ]
    """
    Plots first image from batch and displays table-like
    text of predicted probabilities for each attribute.
    """
    images = images.detach().cpu()
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()

    # Plot only the first image from the batch
    i = 0
    _, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(12, 5))

    img = images[i].permute(1, 2, 0).numpy()

    img = img * 0.5 + 0.5
    img = img.clip(0, 1)  # Ensure pixel values are within [0,1]

    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"Epoch {epoch}", fontsize=10)

    pred_vals = predictions[i].numpy()
    target_vals = targets[i].numpy()

    lines = []
    for attr_name, pred_val, true_val in zip(celeba_attributes, pred_vals, target_vals):
        lines.append(f"{attr_name:20s}: {pred_val:.2f} (T: {int(true_val)})")
    text_str = "\n".join(lines)

    ax_text.axis("off")
    ax_text.text(0, 1, text_str, fontsize=8, va="top", family="monospace")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def sample_trajectory(model, x_init, steps=100):
    """
    Numerically integrates the learned flow from t=0 to t=1 using Euler’s method.

    Args:
        model (torch.nn.Module): the trained flow-matching network.
        x_init (torch.Tensor): initial sample(s) at t=0 with shape (B,2).
        steps (int): number of integration steps.

    Returns:
        traj (torch.Tensor): trajectory of shape (steps+1, B, 2).
    """
    dt = 1.0 / steps
    traj = [x_init.to(DEVICE).clone()]
    t_values = torch.linspace(0, 1, steps, device=DEVICE)
    x = x_init.to(DEVICE).clone()

    for t in t_values:
        t_batch = t.expand(x.size(0))
        v = model(x, t_batch)
        x = x + dt * v
        traj.append(x.clone())

    return torch.stack(traj, dim=0)


def sample_trajectory_crank_nicolson(model, x_init, steps=100):
    """
    Numerically integrates the learned flow from t=0 to t=1 using the Crank-Nicolson method.

    Args:
        model (torch.nn.Module): the trained flow-matching network.
        x_init (torch.Tensor): initial sample(s) at t=0 with shape (B,2).
        steps (int): number of integration steps.

    Returns:
        traj (torch.Tensor): trajectory of shape (steps+1, B, 2).
    """
    dt = 1.0 / steps
    traj = [x_init.to(DEVICE).clone()]
    t_values = torch.linspace(0, 1 - dt, steps, device=DEVICE)
    x = x_init.to(DEVICE).clone()

    for t in t_values:
        t_batch = t.expand(x.size(0))
        v_current = model(x, t_batch)
        x_pred = x + dt * v_current  # Predictor step

        t_next = t + dt
        t_next_batch = t_next.expand(x.size(0))
        v_next = model(x_pred, t_next_batch)

        # Corrector step: average velocity
        x = x + (dt / 2) * (v_current + v_next)
        traj.append(x.clone())

    return torch.stack(traj, dim=0)


def analytical_trajectory(x0, x1, steps=100):
    """
    Computes the ideal (straight-line) trajectory:
      x(t) = (1-t) * x0 + t * x1.

    Args:
        x0 (torch.Tensor): initial sample with shape (1,2).
        x1 (torch.Tensor): target sample with shape (1,2).
        steps (int): number of time steps.

    Returns:
        traj (torch.Tensor): trajectory of shape (steps+1, 2).
    """
    t_values = torch.linspace(0, 1, steps + 1).unsqueeze(1)
    traj = (1 - t_values) * x0 + t_values * x1
    return traj


def plot_trajectories(traj_mlp, traj_ana):
    """
    Plots the trajectories (for a single sample) as computed by the
    MLP and the analytical (straight-line) interpolation.

    Args:
        traj_mlp (torch.Tensor): trajectory from the model of shape (steps+1, 1, 2).
        traj_ana (torch.Tensor): analytical trajectory of shape (steps+1, 2).
    """
    mlp_path = traj_mlp[:, 0, :].detach().cpu()
    ana_path = traj_ana.detach().cpu()

    plt.figure(figsize=(6, 6))
    plt.plot(mlp_path[:, 0], mlp_path[:, 1], "o-", label="MLP Flow Trajectory")
    plt.plot(ana_path[:, 0], ana_path[:, 1], "x--", label="Analytical (Straight Line)")
    plt.scatter(
        mlp_path[0, 0], mlp_path[0, 1], color="green", s=50, label="Start (t=0)"
    )
    plt.scatter(mlp_path[-1, 0], mlp_path[-1, 1], color="red", s=50, label="End (t=1)")
    plt.title("Trajectory Comparison")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_flow(model, rho=0.55, steps=100, sample_size=1000):
    """
    Evaluates the learned flow by comparing the final
    distribution of samples with the target distribution.

    Args:
        model (torch.nn.Module): The trained flow model.
        steps (int): Number of integration steps.
        final_batch_size (int): Number of samples for final distribution evaluation.
    """

    rho = 0.55
    mean = torch.tensor([0.0, 0.0]).to(DEVICE)
    cov = torch.tensor([[1.0, rho], [rho, 1.0]]).to(DEVICE)
    target_distribution = dists.MultivariateNormal(loc=mean, covariance_matrix=cov)

    # Evaluate the final distribution using a larger batch of samples
    big_batch = torch.randn(sample_size, 2).to(DEVICE)
    traj_big = sample_trajectory(model, big_batch, steps=steps)
    x_final = traj_big[-1]
    print(f"Estimated mean of final samples: {x_final.mean(dim=0)}")
    print(f"Estimated correlation matrix:\n{torch.corrcoef(x_final.T)}")

    # Plot final samples versus target samples
    target_samples = target_distribution.sample((sample_size,)).to(DEVICE)
    plt.figure(figsize=(6, 6))
    plt.scatter(
        target_samples[:, 0].detach().cpu(),
        target_samples[:, 1].detach().cpu(),
        color="red",
        alpha=0.3,
        label="Target Samples",
    )
    plt.scatter(
        x_final[:, 0].detach().cpu(),
        x_final[:, 1].detach().cpu(),
        color="blue",
        alpha=0.3,
        label="MLP Flow Samples",
    )
    plt.title("Final MLP Flow Samples vs. Target Distribution")
    plt.legend()
    plt.show()
