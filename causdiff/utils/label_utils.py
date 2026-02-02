import torch

from tqdm import tqdm
from causdiff.utils.misc_utils import save_checkpoint
from causdiff.utils.image_utils import (
    plot_image_with_prediction_vector,
    plot_batch_of_images_with_prediction_and_label,
)


def compute_predictions(output, multi_label):
    """
    Compute predictions based on the mode.
    For multi-label, apply sigmoid and threshold at 0.4.
    For single-label, take the argmax.
    """
    if multi_label:
        return (torch.sigmoid(output) > 0.4).float()
    else:
        return output.argmax(dim=1)


def compute_accuracy(preds, target, multi_label):
    """
    Compute accuracy:   For multi-label, compare elementwise;
                        For single-label, compare image-wise.
    Returns: Fraction of correct answers
    """
    if multi_label:
        correct = (preds == target).sum().item()
        total = target.numel()
    else:
        correct = (preds == target).sum().item()
        total = target.size(0)
    return correct, total


def train_epoch(model, optimizer, criterion, device, train_loader, epoch, multi_label):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    tq = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
    for data_, target in tq:
        data_, target = data_.to(device), target.to(device)
        if multi_label:
            # For multi-label, ensure targets are floats
            target = target.float()
        optimizer.zero_grad()
        output = model(data_)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data_.size(0)
        preds = compute_predictions(output, multi_label)
        c, t = compute_accuracy(preds, target, multi_label)
        correct += c
        total += t

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    print(f"Epoch {epoch}: Train Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}")


def evaluate(model, criterion, device, val_loader, epoch, save_dir, multi_label):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data_, target in tqdm(val_loader, desc=f"Epoch {epoch} Evaluation"):
            data_, target = data_.to(device), target.to(device)
            if multi_label:
                target = target.float()
            output = model(data_)
            loss = criterion(output, target)
            total_loss += loss.item() * data_.size(0)
            preds = compute_predictions(output, multi_label)
            c, t = compute_accuracy(preds, target, multi_label)
            correct += c
            total += t

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = correct / total

    # Plot a batch of images with predictions and labels using the last batch.
    with torch.no_grad():
        out = model(data_)
        pred_labels = compute_predictions(out, multi_label)
    if multi_label:
        plot_image_with_prediction_vector(
            data_, pred_labels, target, epoch, save_dir=save_dir
        )
    else:
        plot_batch_of_images_with_prediction_and_label(
            data_, 8, pred_labels, target, epoch, save_dir=save_dir
        )
    save_checkpoint(model, criterion, epoch, avg_loss, save_dir)

    print(f"Validation: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}")


def train(
    model,
    optimizer,
    loss_fct,
    device,
    train_loader,
    val_loader,
    epochs,
    save_every,
    save_dir,
    multi_label_prediction,
):
    """
    Train the model for a given number of epochs, evaluating and saving <b>save_every</b> to the <b>save_dir</b>.
    ### Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Total number of epochs.
        save_every: Frequency (in epochs) to save/validate the model.
        save_dir: Directory to save checkpoints and plots.
        multi_label:If True, uses multi-label mode; else single-label mode.
    """

    for epoch in range(1, epochs + 1):
        train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=loss_fct,
            device=device,
            train_loader=train_loader,
            epoch=epoch,
            multi_label=multi_label_prediction,
        )
        if epoch % save_every == 0:
            evaluate(
                model,
                loss_fct,
                device,
                val_loader,
                epoch,
                save_dir,
                multi_label_prediction,
            )
