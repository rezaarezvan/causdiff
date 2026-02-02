import os
import torch


def save_checkpoint(model, optimizer, step, losses, save_path):
    """
    Saves a model checkpoint and other necessary training information.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        step (int): Current training step.
        losses (list): List of training losses.
        save_path (str): Path to save the checkpoint.
    """
    ckpt_path = os.path.join(save_path, f"ckpt_step_{step}.pt")
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "losses": losses,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved at step {step}: {ckpt_path}")
