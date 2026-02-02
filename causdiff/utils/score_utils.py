"""
Utilities for Score-based Generative Models (SGM) with improved numerical stability.

This module provides functions for implementing score-based diffusion models following
the approach in Song et al. (2020). The model predicts the score function (gradient of
log-density) rather than the noise itself.

References:
    - Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients
      of the data distribution. NeurIPS.
    - Song, Y., et al. (2020). Improved techniques for training score-based
      generative models. NeurIPS.
    - Song, Y., et al. (2021). Score-based generative modeling through stochastic
      differential equations. ICLR.
"""

import os
import torch
import wandb

from tqdm import tqdm
from causdiff import DEVICE, SAVE_PATH
from causdiff.utils.misc_utils import save_checkpoint
from causdiff.utils.image_utils import plot_image_grid


def get_sigma_schedule(
    sigma_min=0.01, sigma_max=25.0, num_steps=1000, schedule_type="geometric"
):
    """
    Generate a variance schedule for the noise levels.

    Args:
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        num_steps (int): Number of noise levels
        schedule_type (str): Type of schedule ('geometric', 'linear', etc.)

    Returns:
        Noise level schedule as a 1D tensor
    """
    if schedule_type == "geometric":
        # Geometric progression from sigma_max to sigma_min
        return torch.tensor(
            [
                sigma_max * (sigma_min / sigma_max) ** (i / (num_steps - 1))
                for i in range(num_steps)
            ]
        )
    elif schedule_type == "linear":
        # Linear progression from sigma_max to sigma_min
        return torch.linspace(sigma_max, sigma_min, num_steps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def langevin_dynamics_sample(
    score_model, sigmas, x_shape, num_steps=20, step_size=2e-5
):
    """
    Generate samples using Langevin dynamics with annealed noise.

    Args:
        score_model: Model that predicts score function
        sigmas: Schedule of noise standard deviations
        x_shape: Shape of samples to generate
        num_steps: Number of steps for each noise level
        step_size: Step size multiplier

    Returns:
        Generated samples
    """
    # Initialize from random noise
    x = torch.randn(*x_shape, device=DEVICE)

    # Gradually denoise the samples with Langevin dynamics
    with torch.no_grad():
        for i, sigma in enumerate(tqdm(sigmas, desc="Sampling")):
            # Adjust step size based on noise level
            step_size_sigma = step_size * (sigma / sigmas[-1]) ** 2

            # Run Langevin dynamics for a few steps
            for _ in range(num_steps):
                # Get score estimate from model
                # Normalize input but with numerical stability
                normalized_x = x / (sigma + 1e-5)  # Add small epsilon
                t = torch.ones(x_shape[0], device=DEVICE) * (i / len(sigmas))
                score = score_model(normalized_x, t)

                # Prevent extreme score values
                score = torch.clamp(score, -1000, 1000)

                # Update samples with Langevin dynamics
                z = torch.randn_like(x)
                x = x + step_size_sigma * score + torch.sqrt(2 * step_size_sigma) * z

    return x


def pc_sampler(
    score_model, sigmas, x_shape, predictor_steps=10, corrector_steps=1, snr=0.16
):
    """
    Generate samples using the Predictor-Corrector sampler from NCSN++.

    Args:
        score_model: Model that predicts score function
        sigmas: Schedule of noise levels
        x_shape: Shape of samples to generate
        predictor_steps: Number of Euler steps per noise level
        corrector_steps: Number of Langevin steps per noise level
        snr: Signal-to-noise ratio for Langevin dynamics

    Returns:
        Generated samples
    """
    # Initialize from random noise
    x = torch.randn(*x_shape, device=DEVICE)

    with torch.no_grad():
        for i, sigma in enumerate(tqdm(sigmas, desc="Sampling")):
            # Get normalized time
            t = torch.ones(x_shape[0], device=DEVICE) * (i / len(sigmas))

            # Predictor step (Euler)
            if i < len(sigmas) - 1:
                h = sigmas[i + 1] - sigma
                # If h becomes negative, we need to adjust it
                h = sigmas[i] - sigmas[i + 1] if h < 0 else h
                x_norm = x / (sigma + 1e-5)  # Add epsilon for stability
                score = score_model(x_norm, t)
                # Clamp score to prevent extreme values
                # score = torch.clamp(score, -1000, 1000)
                x = x + h * score

            # Corrector step (Langevin)
            for _ in range(corrector_steps):
                noise = torch.randn_like(x)
                step_size = 2 * (snr * sigma) ** 2
                x_norm = x / (sigma + 1e-5)
                score = score_model(x_norm, t)
                # Clamp score to prevent extreme values
                # score = torch.clamp(score, -1000, 1000)
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise

            # Plot intermediate results
            # if i % 10 == 0:
            #     nrow = int(x.shape[0]**0.5)
            #     plot_image_grid(x, nrow, SAVE_PATH, f"pc_sample_{
            #                     i}.png", x.shape[1])
            #     print(f"Generated PC samples saved at step {i}")

    return x


def score_based_train_loop(
    model, optimizer, dataloader, args, checkpoint_interval=2000
):
    """
    Training loop for score-based generative models.

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

    # Set up noise schedule with lower sigma_max for more stability
    sigma_max = min(args.sigma_max, 25.0)  # Cap at 25.0 for stability
    sigma_min = max(args.sigma_min, 0.01)  # Ensure minimum is not too small

    sigmas = get_sigma_schedule(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=args.noise_levels,
        schedule_type=args.schedule_type,
    ).to(DEVICE)

    # Resume training if checkpoint exists
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        losses = ckpt["losses"]
        print(f"Resumed at step {start_step}")

    # Add gradient clipping for stability
    max_grad_norm = 1.0

    pbar = tqdm(
        range(start_step, training_steps), initial=start_step, total=training_steps
    )
    data_iter = iter(dataloader)

    for step in pbar:
        try:
            image, label = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            image, label = next(data_iter)

        image = image.to(DEVICE)
        # Normalize image to [-1, 1] if needed
        if image.min() >= 0 and image.max() <= 1:
            image = 2 * image - 1

        batch_size = image.shape[0]

        # Sample random noise levels
        noise_level_idx = torch.randint(0, len(sigmas), (batch_size,), device=DEVICE)
        used_sigmas = sigmas[noise_level_idx].view(batch_size, 1, 1, 1)

        # Convert noise level indices to time values in [0, 1] for the UNet
        t = noise_level_idx.float() / (len(sigmas) - 1)

        # Add noise to the images
        noise = torch.randn_like(image)
        noisy_image = image + used_sigmas * noise

        # Prevent division by very small values
        epsilon = 1e-5
        noisy_image_norm = noisy_image / (used_sigmas + epsilon)

        # Get score prediction
        score_pred = model(noisy_image_norm, t)

        # Loss: denoising score matching with clamping for stability
        target = -noise / (used_sigmas + epsilon)

        # Clamp target to reasonable values to prevent extreme gradients
        target = torch.clamp(target, -1000, 1000)

        # loss = F.mse_loss(score_pred, target)
        loss = torch.mean((used_sigmas**2) * (score_pred - target) ** 2)

        # Backprop and update with gradient clipping
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Monitor loss - detect if it's becoming NaN or too high
        if torch.isnan(loss) or loss.item() > 1e6:
            print(
                f"Warning: Loss is {'NaN' if torch.isnan(loss) else 'extremely high'}. "
                f"Consider adjusting hyperparameters or using a lower learning rate."
            )

            if torch.isnan(loss):
                # If we hit NaN, try to recover
                print("Skipping this step due to NaN loss")
                continue

        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

        if args.wandb:
            wandb.log({"train_loss": loss.item()}, step=step)

        if (step + 1) % checkpoint_interval == 0 and (step + 1) < training_steps:
            save_checkpoint(model, optimizer, step + 1, losses, SAVE_PATH)

    return losses


def score_based_sample_generation(
    model, args, img_size, channels, batch_size, plot=True
):
    """
    Generate samples using a trained score-based model.

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
    sigmas = get_sigma_schedule(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        num_steps=args.noise_levels,
        schedule_type=args.schedule_type,
    ).to(DEVICE)

    # Set up sample shape
    x_shape = (batch_size, channels, img_size, img_size)

    # Select sampling method
    if args.sampler == "langevin":
        x = langevin_dynamics_sample(
            model,
            sigmas,
            x_shape,
            num_steps=args.langevin_steps,
            step_size=args.step_size,
        )
    elif args.sampler == "pc":
        x = pc_sampler(
            model,
            sigmas,
            x_shape,
            predictor_steps=args.predictor_steps,
            corrector_steps=args.corrector_steps,
            snr=args.snr,
        )
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # Rescale to [0, 1] for visualization
    x = (x + 1) / 2

    # Lists to store snapshots for plotting
    if plot:
        # Plot samples
        nrow = int(batch_size**0.5)
        plot_image_grid(x, nrow, SAVE_PATH, f"score_sample.png", channels)
        print(f"Generated score-based samples saved")

    return x
