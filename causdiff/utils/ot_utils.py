import torch

from causdiff import DEVICE


def sinkhorn(cost_matrix, epsilon=1e-3, max_iter=100):
    """
    # https://x.com/gabrielpeyre/status/1012214072547295232
    # https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem
    Stabilized Sinkhorn algorithm for regularized optimal transport.

    Args:
        cost_matrix (torch.Tensor): The cost matrix C of shape (batch_size, batch_size)
        epsilon (float): Regularization parameter
        max_iter (int): Maximum number of iterations

    Returns:
        transport_plan (torch.Tensor): The optimal transport plan P of shape (batch_size, batch_size)
    """
    batch_size = cost_matrix.shape[0]

    cost_min = cost_matrix.min()
    cost_max = cost_matrix.max()

    # If cost matrix is uniform (all entries equal), use uniform transport plan
    if torch.allclose(cost_min, cost_max):
        return torch.ones_like(cost_matrix) / batch_size**2

    # Scale the cost matrix to avoid numerical issues
    cost_matrix_scaled = (cost_matrix - cost_min) / (cost_max - cost_min + 1e-8)

    # Use log-domain implementation for numerical stability
    log_K = -cost_matrix_scaled / epsilon
    log_K_max = log_K.max()
    log_K = log_K - log_K_max

    # Compute Gibbs kernel
    # K = torch.exp(-cost_matrix / epsilon)
    K = torch.exp(log_K)

    # Check if K has all zeros (numerical underflow)
    if K.sum() < 1e-10:
        # Try again with a larger epsilon
        return sinkhorn(
            cost_matrix, epsilon=epsilon * 10, max_iter=max_iter, device=device
        )

    u = torch.ones(batch_size, 1, device=DEVICE) / batch_size
    v = torch.ones(batch_size, 1, device=DEVICE) / batch_size

    a = torch.ones(batch_size, 1, device=DEVICE) / batch_size
    b = torch.ones(batch_size, 1, device=DEVICE) / batch_size

    for _ in range(max_iter):
        Kv = K @ v
        if (Kv < 1e-10).any():  # Avoid division by zero
            Kv = torch.clamp(Kv, min=1e-10)
        u = a / Kv

        KTu = K.t() @ u
        if (KTu < 1e-10).any():  # Avoid division by zero
            KTu = torch.clamp(KTu, min=1e-10)
        v = b / KTu

        # Check for numerical instability
        if (
            torch.isnan(u).any()
            or torch.isnan(v).any()
            or torch.isinf(u).any()
            or torch.isinf(v).any()
        ):
            # Fall back to a uniform transport plan
            P = torch.ones_like(cost_matrix) / batch_size**2

            # Apply regularization based on cost
            P = P * torch.exp(-cost_matrix_scaled / (epsilon * 5))
            P = P / P.sum()  # Normalize
            return P

    # Compute transport plan with stability check
    P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())
    # Ensure P is a valid transport plan (sums to 1)
    P = P / P.sum()
    assert torch.allclose(P.sum(), torch.tensor(1.0, device=DEVICE), rtol=1e-3)
    return P


def compute_cost_matrix(x, y, cost_fn=None, max_cost=100.0):
    """
    Compute pairwise cost matrix between samples x and y with improved stability.

    Args:
        x (torch.Tensor): The source samples of shape (batch_size, *)
        y (torch.Tensor): The target samples of shape (batch_size, *)
        cost_fn (callable): Custom cost function
        max_cost (float): Maximum cost value

    Returns:
        cost_matrix (torch.Tensor): The cost matrix C of shape (batch_size, batch_size)
    """
    if cost_fn is None:
        # Default cost function: squared Euclidean distance
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        # Normalize to avoid numerical issues
        if x_flat.shape[1] > 100:  # For high-dimensional data (like images)
            x_norm = torch.norm(x_flat, dim=1, keepdim=True)
            y_norm = torch.norm(y_flat, dim=1, keepdim=True)

            # Avoid division by zero
            x_norm = torch.clamp(x_norm, min=1e-8)
            y_norm = torch.clamp(y_norm, min=1e-8)

            x_flat = x_flat / x_norm
            y_flat = y_flat / y_norm

        x_squared = torch.sum(x_flat**2, dim=1, keepdim=True)
        y_squared = torch.sum(y_flat**2, dim=1, keepdim=True)

        xy = torch.matmul(x_flat, y_flat.t())
        C = x_squared + y_squared.t() - 2 * xy

        # Ensure costs are positive and bounded
        C = torch.clamp(C, min=0.0, max=max_cost)
        return C
    else:
        # Custom cost function
        batch_size = x.size(0)
        C = torch.zeros(batch_size, batch_size, device=x.device)
        for i in range(batch_size):
            for j in range(batch_size):
                C[i, j] = torch.clamp(cost_fn(x[i], y[j]), min=0.0, max=max_cost)
        return C


def ot_interpolation(x0, x1, transport_plan, t):
    """
    Interpolate between x0 and x1 using the transport plan.

    Args:
        x0: Source samples of shape (batch_size, *)
        x1: Target samples of shape (batch_size, *)
        transport_plan: Optimal transport plan of shape (batch_size, batch_size)
        t: Interpolation time in [0, 1] of shape (batch_size,)

    Returns:
        xt: Interpolated samples of shape (batch_size, *)
    """
    batch_size = x0.size(0)
    device = x0.device

    # Check transport plan for validity
    if torch.isnan(transport_plan).any() or torch.isinf(transport_plan).any():
        # Fall back to identity transport (straight-line interpolation)
        transport_plan = torch.eye(batch_size, device=device)

    # Ensure transport plan sums to 1
    if not torch.allclose(
        transport_plan.sum(), torch.tensor(1.0, device=device), rtol=1e-3
    ):
        transport_plan = transport_plan / (transport_plan.sum() + 1e-8)

    t_expanded = t.view(-1, *([1] * (x0.dim() - 1)))

    # Linear interpolation along OT paths
    xt = (1 - t_expanded) * x0

    # Apply transport plan to determine the target point
    if x0.dim() > 2:  # For images
        # Reshape for matrix multiplication
        x1_flat = x1.view(batch_size, -1)
        x1_transported = torch.matmul(transport_plan, x1_flat)
        x1_transported = x1_transported.view_as(x0)
        xt = xt + t_expanded * x1_transported
    else:  # For 2D data
        x1_transported = torch.matmul(transport_plan, x1)
        xt = xt + t_expanded * x1_transported

    # Check for NaN/Inf in result
    if torch.isnan(xt).any() or torch.isinf(xt).any():
        # Fall back to straight-line interpolation
        xt = (1 - t_expanded) * x0 + t_expanded * x1

    return xt


def compute_ot_flow(model, x0, x1, t, epsilon=1e-3):
    """
    Compute the OT flow between x0 and x1 at time t.

    Args:
        model (torch.nn.Module): The flow model
        x0 (torch.Tensor): The source samples of shape (batch_size, *)
        x1 (torch.Tensor): The target samples of shape (batch_size, *)
        t (torch.Tensor): The interpolation time in [0, 1] of shape (batch_size,)
        epsilon (float): Regularization parameter for Sinkhorn

    Returns:
        xt (torch.Tensor): Interpolated samples of shape (batch_size, *)
        flow (torch.Tensor): The flow field at time t
    """
    cost_matrix = compute_cost_matrix(x0, x1)
    transport_plan = sinkhorn(cost_matrix, epsilon)
    xt = ot_interpolation(x0, x1, transport_plan, t)

    t_batch = t.expand(x0.size(0))
    flow = model(xt, t_batch)

    return xt, flow, transport_plan


def ot_displacement_interpolation(x0, x1, transport_plan, t):
    """
    Displacement interpolation between x0 and x1 using the transport plan.
    This is more geometry-aware than linear interpolation.

    Args:
        x0 (torch.Tensor): The source samples of shape (batch_size, *)
        x1 (torch.Tensor): The target samples of shape (batch_size, *)
        transport_plan (torch.Tensor): The optimal transport plan P of shape (batch_size, batch_size)
        t (torch.Tensor): The interpolation time in [0, 1] of shape (batch_size,)

    Returns:
        xt (torch.Tensor): Interpolated samples of shape (batch_size, *)
    """
    batch_size = x0.size(0)

    if not torch.allclose(transport_plan.sum(), torch.tensor(1.0), rtol=1e-3):
        transport_plan = transport_plan / (transport_plan.sum() + 1e-8)

    t_expanded = t.view(-1, *([1] * (x0.dim() - 1)))

    # Apply transport plan to determine the displacement
    if x0.dim() > 2:  # For images
        # Reshape for matrix multiplication
        x0_flat = x0.view(batch_size, -1)
        x1_flat = x1.view(batch_size, -1)

        # Get the mapped target positions
        x1_transported = torch.matmul(transport_plan, x1_flat)

        # Compute displacement vectors
        displacement = x1_transported - x0_flat

        # Apply displacement interpolation
        xt_flat = x0_flat + t_expanded.view(-1, 1) * displacement
        xt = xt_flat.view_as(x0)
    else:  # For 2D data
        # Get the mapped target positions
        x1_transported = torch.matmul(transport_plan, x1)

        # Compute displacement vectors
        displacement = x1_transported - x0

        # Apply displacement interpolation
        xt = x0 + t_expanded * displacement

    return xt
