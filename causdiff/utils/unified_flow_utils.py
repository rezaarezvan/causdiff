import inspect

from causdiff.utils.flow_utils import train_loop as fm_train_loop
from causdiff.utils.ot_flow_utils import ot_train_loop, ot_sample_batch_generation
from causdiff.utils.flow_utils import sample_batch_generation as fm_sample_generation


def unified_train_loop(model, optimizer, dataloader, args, **kwargs):
    """
    Unified training loop that supports both standard flow matching and optimal transport.

    Args:
        model (nn.Module): The model to train
        optimizer (torch.optim.Optimizer): The optimizer to use
        dataloader (torch.utils.data.DataLoader): The data loader
        args (argparse.Namespace): Command-line arguments (with .use_ot to select method)
        **kwargs (dict): Additional arguments

    Returns:
        Losses from training
    """
    # Check if we should use OT
    use_ot = getattr(args, "use_ot", False)
    ot_method = getattr(args, "ot_method", "displacement")
    ot_epsilon = getattr(args, "ot_epsilon", 1e-3)

    if use_ot:
        # Filter out any kwargs not accepted by ot_train_loop
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(ot_train_loop).parameters
        }

        # Add OT-specific parameters
        valid_kwargs.update({"ot_method": ot_method, "ot_epsilon": ot_epsilon})

        print(
            f"Using Optimal Transport training (method: {ot_method}, epsilon: {
                ot_epsilon
            })"
        )
        return ot_train_loop(model, optimizer, dataloader, args, **valid_kwargs)
    else:
        # Filter for flow matching kwargs
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(fm_train_loop).parameters
        }

        print("Using standard Flow Matching training")
        return fm_train_loop(model, optimizer, dataloader, args, **valid_kwargs)


def unified_sample_generation(model, args, img_size, channels, batch_size, **kwargs):
    """
    Unified sampling function that supports both standard flow matching and optimal transport.

    Args:
        model (nn.Module): The model to sample from
        args (argparse.Namespace): Command-line arguments (with .use_ot to select method)
        img_size (int): Image size
        channels (int): Number of channels
        batch_size (int): Batch size
        **kwargs (dict): Additional arguments

    Returns:
        Generated samples
    """
    # Check if we should use OT
    use_ot = getattr(args, "use_ot", False)
    ot_epsilon = getattr(args, "ot_epsilon", 1e-3)

    if use_ot:
        # Filter out any kwargs not accepted by ot_sample_batch_generation
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(ot_sample_batch_generation).parameters
        }

        # Add OT-specific parameters
        valid_kwargs["ot_epsilon"] = ot_epsilon

        print(f"Using Optimal Transport sampling (epsilon: {ot_epsilon})")
        return ot_sample_batch_generation(
            model, args, img_size, channels, batch_size, **valid_kwargs
        )
    else:
        # Filter for flow matching kwargs
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(fm_sample_generation).parameters
        }

        print("Using standard Flow Matching sampling")
        return fm_sample_generation(
            model, args, img_size, channels, batch_size, **valid_kwargs
        )
