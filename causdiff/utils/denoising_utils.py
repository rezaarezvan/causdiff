"""
Utilities for Denoising Diffusion Probabilistic Models (DDPM).

This module provides functions for implementing the DDPM algorithm from Ho et al. (2020),
including noise scheduling, forward diffusion, training, and sampling.

References:
    - Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models.
      arXiv preprint arXiv:2006.11239.
    - Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models.
      arXiv preprint arXiv:2102.09672.
"""

import os
import torch
import wandb
import torch.nn.functional as F

from tqdm import tqdm
from causdiff import DEVICE, SAVE_PATH
from causdiff.utils.misc_utils import save_checkpoint
from causdiff.utils.image_utils import plot_image_grid

# NOTE: We can use the existing UNet model from causdiff/models, but the model
# predicts noise instead of velocity fields. We normalize the timesteps
# from integers in [0, noise_steps-1] to floats in [0, 1] for the UNet.


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear noise schedule as in the original DDPM paper.

    Args:
        timesteps: Number of timesteps in the diffusion process
        beta_start: Starting value for beta
        beta_end: Ending value for beta

    Returns:
        beta schedule as a 1D tensor
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule from the improved DDPM paper.

    Args:
        timesteps: Number of timesteps in the diffusion process
        s: Small offset for improved numerical stability

    Returns:
        beta schedule as a 1D tensor
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_diffusion_variables(betas):
    """
    Compute all variables needed for the diffusion process from beta values.

    Args:
        betas: Beta schedule tensor

    Returns:
        Dictionary containing all diffusion variables
    """
    # Define alpha = 1 - beta
    alphas = 1.0 - betas

    # Define cumprod of alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Define sqrt of alphas_cumprod and complement
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Compute alphas used for posterior q(x_{t-1} | x_t, x_0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


def extract(a, t, x_shape):
    """
    Extract timestep-specific values from a tensor and reshape to match x_shape.

    Args:
        a: Source tensor to extract from
        t: Timestep indices
        x_shape: Shape of the target tensor

    Returns:
        Extracted values reshaped to match x_shape
    """
    batch_size = t.shape[0]
    # Debug: Check if t contains valid indices
    # print(f"Debug extract - a shape: {a.shape}, t shape: {
    #       t.shape}, t min/max: {t.min()}-{t.max()}")
    assert t.max() < len(a), f"Index out of bounds: max(t)={t.max()}, len(a)={len(a)}"
    assert t.min() >= 0, f"Negative index: min(t)={t.min()}"
    # Check if indices are valid
    if torch.any(t < 0) or torch.any(t >= a.shape[0]):
        raise ValueError(
            f"Invalid indices in t: min={t.min().item()}, max={
                t.max().item()
            }, a.shape[0]={a.shape[0]}"
        )

    out = a.gather(-1, t)
    result = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(DEVICE)

    return result


def forward_diffusion(x0, t, diffusion_vars):
    """
    Forward diffusion process q(x_t | x_0).

    Args:
        x0: Original clean image
        t: Timestep indices
        diffusion_vars: Diffusion schedule variables

    Returns:
        xt: Noisy image at timestep t
        noise: The noise added to x0
    """
    noise = torch.randn_like(x0)

    # Extract the relevant variables for timestep t
    sqrt_alphas_cumprod_t = extract(diffusion_vars["sqrt_alphas_cumprod"], t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_vars["sqrt_one_minus_alphas_cumprod"], t, x0.shape
    )

    # Compute x_t using the forward process formula,
    # x_t = sqrt(alphas_cumprod_t) * x0 + sqrt(1 - alphas_cumprod_t) * noise
    xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    return xt, noise


def ddpm_train_loop(model, optimizer, dataloader, args, checkpoint_interval=2000):
    """
    Training loop for DDPM.

    Args:
        model: The UNet model to train
        optimizer: The optimizer
        dataloader: Data loader for training
        args: Arguments for training
        checkpoint_interval: How often to save checkpoints

    Returns:
        losses: List of losses during training
    """
    losses = []
    training_steps = args.steps
    start_step = 0

    # Set up noise schedule
    if args.scheduler == "linear":
        betas = linear_beta_schedule(args.noise_steps, args.beta_start, args.beta_end)
    elif args.scheduler == "cosine":
        betas = cosine_beta_schedule(args.noise_steps, args.s)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # Print beta schedule for verification
    print(f"Beta schedule: min={betas.min().item():.6f}, max={betas.max().item():.6f}")

    # Compute diffusion variables once
    diffusion_vars = get_diffusion_variables(betas.to(DEVICE))

    # Resume training if checkpoint exists
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        losses = ckpt["losses"]
        print(f"Resumed at step {start_step}")

    pbar = tqdm(
        range(start_step, training_steps), initial=start_step, total=training_steps
    )
    data_iter = iter(dataloader)

    for step in pbar:
        try:
            image, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            image, _ = next(data_iter)

        image = image.to(DEVICE)
        if image.min() >= 0 and image.max() <= 1:
            # Convert from [0, 1] to [-1, 1] range if needed
            image = 2 * image - 1

        # Sample random timesteps
        t_indices = torch.randint(
            0, args.noise_steps, (image.size(0),), device=DEVICE
        ).long()

        # Convert timestep indices to float in [0, 1] for the UNet
        t = t_indices.float() / args.noise_steps

        # Get noisy image and target noise
        x_t, target_noise = forward_diffusion(image, t_indices, diffusion_vars)

        # Predict noise
        noise_pred = model(x_t, t)

        # Compute loss
        loss = F.mse_loss(noise_pred, target_noise)

        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

        if args.wandb:
            wandb.log({"train_loss": loss.item()}, step=step)

        if (step + 1) % checkpoint_interval == 0 and (step + 1) < training_steps:
            save_checkpoint(model, optimizer, step + 1, losses, SAVE_PATH)

    return losses


def ddpm_sample_batch_generation(
    model, args, img_size, channels, batch_size, plot=True
):
    """
    Generate samples using DDPM reverse diffusion process.

    Args:
        model: The trained model
        args: Arguments for sampling
        img_size: Size of images
        channels: Number of channels
        batch_size: Batch size for generation
        plot: Whether to plot and save images

    Returns:
        x: Generated samples
    """
    model.eval()

    # Set up noise schedule
    if args.scheduler == "linear":
        betas = linear_beta_schedule(args.noise_steps, args.beta_start, args.beta_end)
    elif args.scheduler == "cosine":
        betas = cosine_beta_schedule(args.noise_steps, args.s)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # Compute diffusion variables
    diffusion_vars = get_diffusion_variables(betas.to(DEVICE))

    # Start from pure noise
    x = torch.randn(batch_size, channels, img_size, img_size, device=DEVICE)

    # Lists to store snapshots for plotting
    snapshots = []
    plot_every = (
        args.noise_steps // args.images_to_generate if plot else args.noise_steps + 1
    )

    # Reverse diffusion process (sampling)
    with torch.no_grad():
        for i in tqdm(reversed(range(args.noise_steps)), desc="Sampling"):
            # Get timestep
            t_index = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)

            # Convert to float for the model
            t = t_index.float() / args.noise_steps

            # Predict noise
            predicted_noise = model(x, t)

            # Get the parameters for this timestep
            alpha = diffusion_vars["alphas"][i]
            alpha_cumprod = diffusion_vars["alphas_cumprod"][i]
            beta = diffusion_vars["betas"][i]

            # No noise if we're at the last timestep
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Compute the next x using the reverse diffusion formula
            x = (
                1
                / torch.sqrt(alpha)
                * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
                + torch.sqrt(beta) * noise
            )

            # Store snapshots for plotting
            if plot and (i % plot_every == 0 or i == 0):
                snapshots.append(x.detach().cpu())

    # Plot and save snapshots
    if plot and snapshots:
        nrow = int(batch_size**0.5)
        for idx, snap in enumerate(snapshots, 1):
            plot_image_grid(
                snap, nrow, SAVE_PATH, f"ddpm_sample_batch_{idx}.png", channels
            )
            print(f"Generated DDPM batch {idx}")

    return x
