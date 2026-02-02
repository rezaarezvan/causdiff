import torch.nn as nn
import torch.optim as optim

from torchvision import models
from causdiff import DEVICE, SAVE_PATH
from causdiff.utils.label_utils import train
from causdiff.data.dataloaders import get_celeba_loader


def main(args):
    if args.wandb:
        import wandb

        wandb.init(
            project="ResNet_celebA_attributes",
            name="ResNet_celebA_attributes",
            config=args.__dict__,
        )
    print(f"Running on {DEVICE}")
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 40)
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_dataset = get_celeba_loader(
        img_size=64, batch_size=args.batch_size, train=True
    )
    train_loader = get_celeba_loader(
        img_size=64, batch_size=args.batch_size, train=False
    )

    train(
        model=model,
        optimizer=optimizer,
        loss_fct=criterion,
        device=DEVICE,
        train_loader=train_dataset,
        val_loader=train_loader,
        epochs=args.epochs,
        save_every=args.save_every,
        save_dir=SAVE_PATH,
        multi_label_prediction=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--on_SUPR", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--save_every", type=bool, default=4)
    args = parser.parse_args()

    main(args)
