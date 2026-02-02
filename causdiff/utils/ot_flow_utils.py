import os
import tqdm
import wandb
import torch

from causdiff import DEVICE, SAVE_PATH
from causdiff.utils.flow_utils import fd_dx
from causdiff.utils.misc_utils import save_checkpoint
from causdiff.utils.image_utils import plot_image_grid, plot_side_by_side
from causdiff.utils.ot_utils import (
    compute_cost_matrix,
    sinkhorn,
    ot_interpolation,
    ot_displacement_interpolation,
)


def ot_train_loop(
    model,
    optimizer,
    dataloader,
    args,
    two_sided=False,
    sourceloader=None,
    checkpoint_interval=20000,
    ot_method="displacement",  # "displacement" or "linear"
    ot_epsilon=1e-3,
):
    """
    Training loop for flow models using optimal transport.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        dataloader (torch.utils.data.DataLoader): The data loader.
        args (argparse.Namespace): The command-line arguments.
        two_sided (bool): Whether to use two-sided conditioning.
        sourceloader (torch.utils.data.DataLoader): The source data loader.
        checkpoint_interval (int): The interval for saving checkpoints.
        ot_method (str): OT interpolation method ("displacement" or "linear").
        ot_epsilon (float): Regularization parameter for the Sinkhorn algorithm.
    """
    losses = []
    training_steps = args.steps
    start_step = 0

    # Limit batch size for OT calculation, since it's O(n^2) :(
    ot_batch_size = min(args.batch_size, 64)

    # Track transport plans for reuse
    cached_transport_plan = None
    cached_x0, cached_x1 = None, None

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        losses = ckpt["losses"]
        print(f"Resumed at step {start_step}")

    pbar = tqdm.tqdm(
        range(start_step, training_steps), initial=start_step, total=training_steps
    )
    data_iter = iter(dataloader)

    if two_sided:
        assert sourceloader is not None, (
            "Two-sided conditioning requires a source loader."
        )

    source_iter = iter(sourceloader) if two_sided else None

    for step in pbar:
        try:
            image, label = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            image, label = next(data_iter)

        if two_sided:
            try:
                source, source_label = next(source_iter)
            except StopIteration:
                source_iter = iter(sourceloader)
                source, source_label = next(source_iter)
        else:
            source, source_label = None, None

        image, label = image.to(DEVICE), label.to(DEVICE)
        if two_sided:
            source, source_label = source.to(DEVICE), source_label.to(DEVICE)

        # x1 is the real image, x0 is noise ~ N(0, I) or the source image
        # Since ot_batch_size <= batch_size, we can safely use the first ot_batch_size samples
        x1 = image[:ot_batch_size]
        x0 = source[:ot_batch_size] if two_sided else torch.randn_like(x1)

        # Recompute transport plan if needed
        if (
            cached_transport_plan is None
            or not torch.allclose(x0, cached_x0)
            or not torch.allclose(x1, cached_x1)
        ):
            cost_matrix = compute_cost_matrix(x0, x1)
            transport_plan = sinkhorn(cost_matrix, epsilon=ot_epsilon)

            # Cache plan for future use
            cached_transport_plan = transport_plan
            cached_x0, cached_x1 = x0.clone(), x1.clone()
        else:
            transport_plan = cached_transport_plan

        t = torch.rand(x1.size(0)).to(DEVICE)

        if ot_method == "displacement":
            xt = ot_displacement_interpolation(x0, x1, transport_plan, t)
            target = x1 - x0
        else:  # "linear"
            xt = ot_interpolation(x0, x1, transport_plan, t)
            x1_transported = torch.matmul(
                transport_plan, x1.view(x1.size(0), -1)
            ).view_as(x1)
            target = x1_transported - x0

        pred = (
            model(xt, t, label=None, source_label=label, target_label=source_label)
            if two_sided
            else model(xt, t, label)
        )
        loss = ((pred - target) ** 2).mean()

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


def ot_sample_batch_generation(
    model,
    args,
    img_size,
    channels,
    batch_size,
    sourceloader=None,
    label=None,
    source_label=None,
    target_label=None,
    plot=True,
    compute_perturbations=False,
    ot_epsilon=1e-3,
):
    """
    Generate a batch of images using OT-based flow, with optional plotting.

    Args:
        model (torch.nn.Module): The model to sample from.
        args (argparse.Namespace): Command-line arguments.
        img_size (int): Image size.
        channels (int): Number of channels.
        batch_size (int): Number of samples.
        sourceloader (torch.utils.data.DataLoader, optional): Source data loader.
        label (int, optional): Label to condition on.
        source_label (int, optional): Source label for two-sided conditioning.
        target_label (int, optional): Target label for two-sided conditioning.
        plot (bool): Whether to plot/save snapshots.
        compute_perturbations (bool): Whether to compute perturbed samples and dx.
        ot_epsilon (float): Regularization parameter for Sinkhorn.

    Returns:
        Final samples and optionally perturbations.
    """
    assert not plot or int((batch_size**0.5)) ** 2 == batch_size, (
        "Batch size must be a square number if plotting."
    )
    model.eval().requires_grad_(False)

    # Limit batch size for OT calculations, since it's O(n^2) :(
    ot_batch_size = min(batch_size, 64)

    if sourceloader is not None:
        xt = next(iter(sourceloader))[0][:ot_batch_size].to(DEVICE)
    else:
        xt = torch.randn(ot_batch_size, channels, img_size, img_size).to(DEVICE)

    x1 = torch.randn(ot_batch_size, channels, img_size, img_size).to(DEVICE)

    # Compute OT plan for interpolation
    cost_matrix = compute_cost_matrix(xt, x1)
    transport_plan = sinkhorn(cost_matrix, epsilon=ot_epsilon)

    t_steps = torch.linspace(0, 1, args.steps, device=DEVICE)
    label_tensor = torch.tensor([label], device=DEVICE) if label is not None else None

    if source_label is not None and target_label is not None:
        source_label_tensor = torch.tensor([source_label], device=DEVICE)
        target_label_tensor = torch.tensor([target_label], device=DEVICE)
    else:
        source_label_tensor, target_label_tensor = None, None

    snapshots = []
    plot_every = args.steps // args.images_to_generate if plot else args.steps + 1

    for i, t in enumerate(t_steps, start=1):
        t_batch = t.expand(xt.size(0))

        pred = (
            model(xt, t_batch, None, source_label_tensor, target_label_tensor)
            if source_label_tensor is not None and target_label_tensor is not None
            else model(xt, t_batch, label_tensor)
        )

        xt = xt + (1.0 / args.steps) * pred

        if plot and i % plot_every == 0:
            snapshots.append(xt.detach().cpu())

    if plot and snapshots:
        nrow = int(batch_size**0.5)
        for idx, snap in enumerate(snapshots, 1):
            plot_image_grid(
                snap, nrow, SAVE_PATH, f"ot_sample_batch_{idx}.png", channels
            )
            print(f"Generated OT batch {idx}")

    if compute_perturbations:
        dx, dz, xt_final, xt_perturbed = fd_dx(model, xt, t_steps, args.epsilon, label)

        if plot:
            plot_side_by_side(
                [xt_final[-1, 0].detach().cpu()],
                [xt_perturbed[-1, 0].detach().cpu()],
                [dx[-1, 0].detach().cpu()],
                args.steps,
                1,
            )

        return xt, xt_perturbed, dx

    return xt
