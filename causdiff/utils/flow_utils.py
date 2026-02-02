import os
import tqdm
import wandb
import torch
import matplotlib.pyplot as plt

from causdiff import DEVICE, SAVE_PATH
from causdiff.utils.misc_utils import save_checkpoint
from causdiff.utils.image_utils import plot_image_grid, plot_side_by_side


def train_loop_cond(
    model,
    optimizer,
    dataloader,
    args,
    two_sided=False,
    sourceloader=None,
    checkpoint_interval=20000,
):
    """
    Common training loop for a conditional flow model, training the network
    to estimate the conditional expectation E[x1 | x_t]. (In this setup, the
    network is trained to predict x1 directly from the interpolated x_t.)

    After training, the drift is recovered by:
      f(x,t)= (g(x,t)-x_t)/(1-t),
    where g(x,t) is the output of the network.

    Args:
        model (torch.nn.Module): The model to train. Now this should output
            an approximation of g(x,t)=E[x1 | x_t].
        optimizer (torch.optim.Optimizer): The optimizer to use.
        dataloader (torch.utils.data.DataLoader): The data loader yielding (image, label).
        args (argparse.Namespace): Command-line arguments.
        two_sided (bool): Whether to use two-sided conditioning.
        sourceloader (torch.utils.data.DataLoader): If two_sided, the source data loader.
        checkpoint_interval (int): How frequently to save checkpoints.
    """
    losses = []
    training_steps = args.steps
    start_step = 0

    # Resume checkpoint if available
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
        source_iter = iter(sourceloader)
    else:
        source_iter = None

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

        # x1 is the real image, and x0 is either the provided source image or noise
        x1 = image.to(DEVICE)
        if two_sided:
            x0 = source.to(DEVICE)
        else:
            x0 = torch.randn_like(x1)

        # In the standard flow matching formulation, the (ideal) drift would be based on:
        #   x1 - x0 = E[x1 - x0 | x_t].
        # Instead, we now train to predict directly g(x,t)=E[x1|x_t]. Note that
        # since x_t is obtained via linear interpolation:
        #   x_t = (1-t)x0 + t*x1,
        # the optimal prediction is g*(x,t)=E[x1|x_t]=x1 (if the conditional expectation
        # were perfect). Thus, here we simply set the target to x1.
        #
        # When sampling, one can then recover the drift via f(x,t)=(g(x,t)-x_t)/(1-t).

        # t ~ Uniform(0,1) for each sample in the batch
        t = torch.rand(x1.size(0)).to(DEVICE)
        # Expand t to have the same number of dimensions as the images
        t_expanded = t.view(-1, *([1] * (x1.dim() - 1)))
        # Interpolate to create the intermediate state: x_t = (1-t)x0 + t*x1.
        xt = (1 - t_expanded) * x0 + t_expanded * x1

        # The network now predicts g(x,t) ≈ E[x1|x_t]
        pred_g = (
            model(xt, t, label=None, source_label=label, target_label=source_label)
            if two_sided
            else model(xt, t, label)
        )

        # Loss: minimize L2 error between the prediction and the true x1.
        loss = ((pred_g - x1) ** 2).mean()
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


def train_loop(
    model,
    optimizer,
    dataloader,
    args,
    two_sided=False,
    sourceloader=None,
    checkpoint_interval=20000,
):
    """
    Common training loop for a conditional flow model.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        dataloader (torch.utils.data.DataLoader): The data loader.
        args (argparse.Namespace): The command-line arguments.
        two_sided (bool): Whether to use two-sided conditioning.
        targetloader (torch.utils.data.DataLoader): The target data loader.
        checkpoint_interval (int): The interval for saving checkpoints.
    """
    losses = []
    training_steps = args.steps
    start_step = 0

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
            source, source_label = (source.to(DEVICE), source_label.to(DEVICE))

        # x1 is the real image, x0 is noise ~ N(0, I) and target is x1 - x0.
        x1 = image
        x0 = source if two_sided else torch.randn_like(x1)
        source = x1 - x0

        # Sample t ~ Uniform(0,1) and interpolate: xt = (1-t)x0 + t*x1.
        t = torch.rand(x1.size(0)).to(DEVICE)
        t_expanded = t.view(-1, *([1] * (x1.dim() - 1)))
        xt = (1 - t_expanded) * x0 + t_expanded * x1

        pred = (
            model(xt, t, label=None, source_label=label, target_label=source_label)
            if two_sided
            else model(xt, t, label)
        )

        # Compute loss, backprop, and update.
        loss = ((pred - source) ** 2).mean()
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


def integrate_flow(model, xt, t_steps, label=None, method="euler", enable_grad=False):
    """
    Integrate the flow model using the specified numerical integration method.

    Args:
        model (torch.nn.Module): Flow model
        xt (torch.Tensor): Initial state (B, C, H, W)
        t_steps (torch.Tensor): Time steps for integration
        label (int): Label for conditional models
        method (str): Integration method ('euler' or 'rk4')
        enable_grad (bool): Whether to enable gradients

    Returns:
        Final state after integration
    """

    assert method.lower() in ["euler", "rk4"], f"Unknown integration method: {method}"

    xt = xt.clone()

    context = torch.enable_grad() if enable_grad else torch.no_grad()

    with context:
        if method.lower() == "euler":
            # https://en.wikipedia.org/wiki/Euler_method
            # Euler method
            for i in range(len(t_steps) - 1):
                t, next_t = t_steps[i], t_steps[i + 1]
                dt = next_t - t

                t_batch = t.expand(xt.size(0))
                pred = model(xt, t_batch, label)
                xt = xt + dt * pred

        elif method.lower() == "rk4":
            # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
            # 4th-order Runge-Kutta method
            for i in range(len(t_steps) - 1):
                t, next_t = t_steps[i], t_steps[i + 1]
                dt = next_t - t

                # First stage
                t_batch = t.expand(xt.size(0))
                k1 = model(xt, t_batch, label)

                # Second stage
                t_half = t + dt / 2
                t_half_batch = t_half.expand(xt.size(0))
                x_temp = xt + dt / 2 * k1
                k2 = model(x_temp, t_half_batch, label)
                del x_temp  # Free memory

                # Third stage
                x_temp = xt + dt / 2 * k2
                k3 = model(x_temp, t_half_batch, label)
                del x_temp  # Free memory

                # Fourth stage
                t_next_batch = next_t.expand(xt.size(0))
                x_temp = xt + dt * k3
                k4 = model(x_temp, t_next_batch, label)
                del x_temp  # Free memory

                # Combine the stages
                xt = xt + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                del k1, k2, k3, k4
                if xt.is_cuda and not enable_grad:
                    torch.cuda.empty_cache()

    return xt


def fd_dx(model, z, t_steps, epsilon=1e-6, label=None):
    """
    Compute the directional derivative of the flow model
    with respect to the input z using finite differences.

    Args:
        model (torch.nn.Module): The (flow) model to differentiate.
        z (torch.Tensor): The input to differentiate (B, C, H, W).
        t_steps (torch.Tensor): The time steps.
        epsilon (float): The perturbation scale.
        label (int): The label to condition on.

    Returns:
        torch.Tensor: The directional derivative.
        torch.Tensor: The direction vector used.
        torch.Tensor: The output at z.
        torch.Tensor: The output at z + epsilon * dz.
    """
    dz = torch.randn_like(z)
    dz_flat = dz.view(z.size(0), -1)
    norms = dz_flat.norm(dim=1, keepdim=True).view(z.size(0), *([1] * (z.dim() - 1)))
    dz = epsilon * dz / norms

    x = integrate_flow(model, z, t_steps, label)
    x_perturbed = integrate_flow(model, z + dz, t_steps, label)
    dx = (x_perturbed - x) / epsilon

    return dx, dz, x, x_perturbed


def ad_dx(model, z, t_steps, epsilon=1e-6, label=None):
    """
    Efficiently compute the directional derivative of the flow model
    with respect to the input z using reverse-mode automatic differentiation.

    Args:
        model (torch.nn.Module): The (flow) model to differentiate.
        z (torch.Tensor): The input to differentiate (B, C, H, W).
        t_steps (torch.Tensor): The time steps.
        epsilon (float): The perturbation scale.
        label (int): The label to condition on.

    Returns:
        torch.Tensor: The directional derivative.
        torch.Tensor: The direction vector used.
        torch.Tensor: The output at z.
        torch.Tensor: The output at z + epsilon * dz.
    """
    with torch.no_grad():
        dz = torch.randn_like(z)
        dz_flat = dz.view(z.size(0), -1)
        norms = dz_flat.norm(dim=1, keepdim=True).view(z.size(0), 1, 1, 1)
        dz = epsilon * dz / norms

        x = integrate_flow(model, z, t_steps, label, enable_grad=False)
        x_perturbed = integrate_flow(model, z + dz, t_steps, label, enable_grad=False)

        torch.cuda.empty_cache()

    try:
        z_requries_grad = z.detach().clone().requires_grad_(True)

        def func(input_tensor):
            return integrate_flow(model, input_tensor, t_steps, label, enable_grad=True)

        torch.cuda.empty_cache()
        _, tangent = torch.func.jvp(func, (z_requries_grad,), (dz,))
        dx = tangent
    except torch.cuda.OutOfMemoryError:
        # Fall back to CPU if GPU mem is insufficient
        print("Warning: Using CPU for gradient computation due to GPU memory limit.")
        model_cpu = model.cpu()
        z_cpu = z.cpu().detach().clone().requires_grad_(True)
        dz_cpu = dz.cpu()
        t_steps_cpu = t_steps.cpu()
        label_cpu = label.cpu() if label is not None else None

        def func(input_tensor):
            return integrate_flow(
                model_cpu, input_tensor, t_steps_cpu, label_cpu, enable_grad=True
            )

        _, tangent = torch.func.jvp(func, (z_cpu,), (dz_cpu,))
        dx = tangent.to(z.device)
        model.to(z.device)

    del tangent
    torch.cuda.empty_cache()

    return dx, dz, x, x_perturbed


def v_ak(
    model,
    z,
    t_steps,
    constraint_idx=None,
    constraint_value=0,
    label=None,
    normalize_final=False,
    second_order=True,
):
    """
    Compute directional derivatives with constraints using automatic differentiation.

    This function finds a random direction v such that the k-th component of the directional
    derivative is equal to the specified constraint_value (default: 0).
    Mathematically it returns (if second_order = False)
    (f(z + εv) - f(z))/ε ~~ ∇f(z)
    else:
    (f(z + εv) - f(z))/ε ~~ ∇f(z) (ε/2)( v^T Δf(z) v )

    Args:
        model (torch.nn.Module): The flow model to differentiate.
        z (torch.Tensor): The input tensor (B, C, H, W).
        t_steps (torch.Tensor): Time steps for flow integration.
        constraint_idx (tuple): Index (B, C, H, W) of the component to constrain.
                               If None, no constraint is applied.
        constraint_value (float): The value to constrain the specified component to (default: 0).
        label (int, optional): Label for conditional models.
        normalize_final (bool): Whether to normalize the final direction vector (default: True).
                            Note that if True, the constraint might not hold exactly.
        second_order (bool): First or second order approximation.

    Returns:
        tuple: (dx, v, x)
            - dx: Directional derivative
            - v: The constrained direction (normalized if normalize_final=True)
            - x: Original output of the model
    """
    eps = 1e-6
    v = torch.randn_like(z)
    v_flat = v.view(z.size(0), -1)
    v_norm = v_flat.norm(dim=1, keepdim=True).view(z.size(0), 1, 1, 1)
    v = v / v_norm
    v *= eps

    with torch.no_grad():
        x = integrate_flow(model, z, t_steps, label, enable_grad=False)

    if constraint_idx is None:

        def func(input_tensor):
            return integrate_flow(model, input_tensor, t_steps, label, enable_grad=True)

        z_requires_grad = z.detach().clone().requires_grad_(True)
        _, dx = torch.func.jvp(func, (z_requires_grad,), (v,))

        return dx, v, x

    z_param = z.detach().clone().requires_grad_(True)
    x_param = integrate_flow(model, z_param, t_steps, label, enable_grad=True)
    component = x_param[constraint_idx]

    component.backward(retain_graph=False)
    a_k = z_param.grad.clone()

    v_dot_a_k = (v * a_k).sum()
    a_k_norm_squared = (a_k * a_k).sum()

    if a_k_norm_squared > 1e-10:
        v_proj = v - (v_dot_a_k / a_k_norm_squared) * a_k

        if constraint_value != 0:
            v_proj = v_proj + (constraint_value / a_k_norm_squared) * a_k
    else:
        v_proj = v
        print(
            "Warning: Constraint vector has very small magnitude. Constraint may not be enforceable."
        )

    if normalize_final:
        v_proj_flat = v_proj.view(z.size(0), -1)
        v_proj_norm = v_proj_flat.norm(dim=1, keepdim=True).view(z.size(0), 1, 1, 1)
        v_proj = v_proj / v_proj_norm

    def func(input_tensor):
        return integrate_flow(model, input_tensor, t_steps, label, enable_grad=True)

    z_requires_grad = z.detach().clone().requires_grad_(True)

    x = func(z_requires_grad)
    _, dx = torch.func.jvp(func, (z_requires_grad,), (v_proj,))
    dx_constraint = dx[constraint_idx]

    print(
        f"Constraint target: {constraint_value}, achieved (original JVP): {
            dx_constraint.item()
        }"
    )

    if second_order:

        def first_order_func(input_tensor):
            _, dx_inner = torch.func.jvp(func, (input_tensor,), (v_proj,))
            return dx_inner

        _, d2x = torch.func.jvp(first_order_func, (z_requires_grad,), (v_proj,))

        d2x_constraint = d2x[constraint_idx]
        second_order_corr = 0.5 * eps * d2x_constraint

        dx_2_approxs = d2x.clone()
        dx_2_approxs[constraint_idx] = dx_constraint + second_order_corr
        print(
            f"Constraint target: {constraint_value}, achieved (2nd order approx): {
                dx_2_approxs[constraint_idx].item()
            }"
        )
        return dx_2_approxs, v_proj, x
    else:
        return dx, v_proj, x


def sample_batch_generation(
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
):
    """
    Generate a batch of images from noise, with optional plotting and perturbation computation.

    Args:
        model (torch.nn.Module): The model to sample from.
        args (argparse.Namespace): Command-line arguments (steps, images_to_generate, epsilon, etc.).
        img_size (int): Image size.
        channels (int): Number of channels.
        batch_size (int): Number of samples (must be a square number if plotting).
        sourceloader (torch.utils.data.DataLoader, optional): Source data loader for two-sided conditioning.
        label (int, optional): Label to condition on (one-sided conditioning).
        source_label (int, optional): Source label for two-sided conditioning.
        target_label (int, optional): Target label for two-sided conditioning.
        plot (bool): Whether to plot/save snapshots.
        compute_perturbations (bool): Whether to compute perturbed samples and dx.

    Returns:
        torch.Tensor: Final xt (if not compute_perturbations).
        tuple: (xt, xt_perturbed, dx) if compute_perturbations is True.
    """
    assert not plot or int((batch_size**0.5)) ** 2 == batch_size, (
        "Batch size must be a square number if plotting."
    )
    model.eval().requires_grad_(False)

    if sourceloader is not None:
        xt = next(iter(sourceloader))[0].to(DEVICE)
    else:
        xt = torch.randn(batch_size, channels, img_size, img_size).to(DEVICE)

    t_steps = torch.linspace(0, 1, args.steps, device=DEVICE)
    label = torch.tensor([label], device=DEVICE) if label is not None else None

    if source_label is not None and target_label is not None:
        source_label = torch.tensor([source_label], device=DEVICE)
        target_label = torch.tensor([target_label], device=DEVICE)
    else:
        source_label, target_label = None, None

    if compute_perturbations:
        dx, dz, xt, xt_perturbed = fd_dx(model, xt, t_steps, args.epsilon, label)
        if plot:
            plot_side_by_side(
                [xt[-1, 0].detach().cpu()],
                [xt_perturbed[-1, 0].detach().cpu()],
                [dx[-1, 0].detach().cpu()],
                args.steps,
                1,
            )
        return xt, xt_perturbed, dx

    snapshots = []
    plot_every = args.steps // args.images_to_generate if plot else args.steps + 1

    for i, t in enumerate(t_steps, start=1):
        t_batch = t.expand(xt.size(0))
        pred = (
            model(xt, t_batch, None, source_label, target_label)
            if source_label is not None and target_label is not None
            else model(xt, t_batch, label)
        )
        xt = xt + (1.0 / args.steps) * pred
        if plot and i % plot_every == 0:
            snapshots.append(xt.detach().cpu())

    if plot and snapshots:
        nrow = int(batch_size**0.5)
        for idx, snap in enumerate(snapshots, 1):
            plot_image_grid(snap, nrow, SAVE_PATH, f"sample_batch_{idx}.png", channels)
            print(f"Generated batch {idx}")

    return xt


def sample_batch_generation2D(
    model,
    args,
    batch_size,
    sourceloader=None,
    label=None,
    source_label=None,
    target_label=None,
    plot=True,
    compute_perturbations=False,
):
    """
    Specialized sampling function for 2D Gaussian data.

    Args:
        model (torch.nn.Module): The model to sample from.
        args (argparse.Namespace): Arguments (steps, images_to_generate, epsilon).
        batch_size (int): Number of samples.
        sourceloader (torch.utils.data.DataLoader, optional): Source loader (optional).
        label (int, optional): Label for conditioning (optional).
        source_label (int, optional): Source label for two-sided conditioning.
        target_label (int, optional): Target label for two-sided conditioning.
        plot (bool): Whether to plot samples.
        compute_perturbations (bool): Compute finite-difference perturbations.

    Returns:
        torch.Tensor: Final samples (and optionally perturbations).
    """
    model.eval().requires_grad_(False)

    if sourceloader is not None:
        xt = next(iter(sourceloader))[0].to(DEVICE)
    else:
        xt = torch.randn(batch_size, 1, device=DEVICE)

    t_steps = torch.linspace(0, 1, args.steps, device=DEVICE)

    label = torch.tensor([label], device=DEVICE) if label is not None else None
    source_label = torch.tensor([source_label], device=DEVICE) if source_label else None
    target_label = torch.tensor([target_label], device=DEVICE) if target_label else None

    snapshots = []
    plot_every = args.steps // args.images_to_generate if plot else args.steps + 1

    for i, t in enumerate(t_steps, start=1):
        t_batch = t.expand(batch_size)

        if source_label is not None and target_label is not None:
            pred = model(xt, t_batch, None, source_label, target_label)
        else:
            pred = model(xt, t_batch, label)

        xt = xt + (1.0 / args.steps) * pred

        if plot and i % plot_every == 0:
            snapshots.append(xt.detach().cpu().numpy())

    if plot and snapshots:
        for idx, snap in enumerate(snapshots, 1):
            plt.figure(figsize=(5, 5))
            plt.scatter(snap[:, 0], snap[:, 1], color="green", marker="o", alpha=0.5)
            plt.title(f"Step {idx * plot_every}/{args.steps}")
            plt.savefig(f"{SAVE_PATH}/sample_2d_{idx}.png")
            plt.close()
            print(f"Generated 2D batch snapshot {idx}")

    if compute_perturbations:
        dx, dz, xt, xt_perturbed = fd_dx(model, xt, t_steps, args.epsilon, label)

        plt.figure(figsize=(5, 5))
        plt.quiver(
            xt[:, 0].cpu(),
            xt[:, 1].cpu(),
            dx[:, 0].cpu(),
            dx[:, 1].cpu(),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="blue",
        )
        plt.title("Perturbation Vectors")
        plt.savefig(f"{SAVE_PATH}/perturbation_quiver.png")
        plt.close()
        print("Saved perturbation quiver plot")

        return xt, xt_perturbed, dx

    return xt
