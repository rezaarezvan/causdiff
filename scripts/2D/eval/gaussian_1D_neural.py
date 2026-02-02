import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from causdiff import DEVICE, SAVE_PATH, SEED

# Set seeds for reproducibility
sns.set(style="whitegrid", context="talk", font_scale=1.2)
np.random.seed(SEED + 1)
torch.manual_seed(SEED + 1)


# Closed-form flow model (analytical solution)
def analytical_flow(x, t):
    """
    Closed form solution of E[x_1 - x_0 | x_t] when x_0, x_1 ~ N(0, 1)
    = (2t - 1)/(1 + t)^2 + t^2 * x
    """
    return ((2 * t - 1) / ((1 + t) ** 2 + t**2)) * x


# Define a simple neural network for the 1D flow
class SimpleFlow1D(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, t, label=None):
        # Time embedding
        t_emb = self.time_mlp(t.view(-1, 1))

        # Concatenate x and t embedding
        x_input = x.view(-1, 1)
        inp = torch.cat([x_input, t_emb], dim=1)

        return self.net(inp).squeeze(-1)


# Create 1D Gaussian dataloader
def get_1d_gaussian_loader(batch_size, num_samples):
    # Sample from standard Gaussian (N(0, 1))
    x1 = torch.randn(num_samples)
    # Create dataset with dummy labels=0 (not used, but needed for API compatibility)
    dataset = TensorDataset(x1.view(-1, 1), torch.zeros(num_samples, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training function for flow model
def train_flow_model(model, dataloader, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    losses = []

    pbar = tqdm(range(args.epochs))
    data_iter = iter(dataloader)

    for step in pbar:
        try:
            x1, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x1, _ = next(data_iter)

        x1 = x1.to(DEVICE).squeeze()
        batch_size = x1.size(0)

        # Sample Gaussian noise as x0
        x0 = torch.randn_like(x1)

        # Target vector field is (x1 - x0)
        target = x1 - x0

        # Sample random t in [0, 1]
        t = torch.rand(batch_size).to(DEVICE)

        # Interpolate x_t = (1-t)*x0 + t*x1
        x_t = (1 - t) * x0 + t * x1

        # Get model prediction
        pred = model(x_t, t)

        # Compute loss
        loss = ((pred - target) ** 2).mean()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.epochs}, Loss: {loss.item():.6f}")

    return losses


# Integrate using the flow model
def integrate_flow(model, x0, steps=1000):
    model.eval()
    t_steps = torch.linspace(0, 1, steps=steps).to(DEVICE)
    traj = torch.zeros((steps, x0.size(0))).to(DEVICE)
    traj[0] = x0

    with torch.no_grad():
        for idx, t in enumerate(t_steps[1:], start=1):
            x_prev = traj[idx - 1]
            t_batch = t.expand(x0.size(0))
            delta = model(x_prev, t_batch)
            dt = 1.0 / (steps - 1)
            traj[idx] = x_prev + delta * dt

    return traj


def main(args):
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving results to {SAVE_PATH}")

    # Create the flow model
    flow_nn = SimpleFlow1D(hidden_dim=args.hidden_dim).to(DEVICE)

    # Train the flow model if requested
    if args.train:
        print("Training neural flow model...")
        dataloader = get_1d_gaussian_loader(args.batch_size, args.num_samples)
        losses = train_flow_model(flow_nn, dataloader, args)

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(os.path.join(SAVE_PATH, "flow_training_loss_1d.png"))
        plt.close()

        # Save the trained model
        torch.save(flow_nn.state_dict(), os.path.join(SAVE_PATH, "flow_1d_model.pt"))
        print(f"Model saved to {os.path.join(SAVE_PATH, 'flow_1d_model.pt')}")
    else:
        # Load pretrained model if available
        try:
            flow_nn.load_state_dict(
                torch.load(
                    os.path.join(SAVE_PATH, "flow_1d_model.pt"), map_location=DEVICE
                )
            )
            print("Loaded pretrained model")
        except:
            print("No pretrained model found, using untrained model")

    # Visualization parameters
    steps = 1000
    x0_samples = dist.Normal(0, 1).sample((args.viz_samples,)).to(DEVICE)
    t_steps = torch.linspace(0, 1, steps=steps).to(DEVICE)

    # Calculate trajectories using the analytical solution
    analytical_traj = torch.zeros((steps, args.viz_samples)).to(DEVICE)
    analytical_traj[0] = x0_samples

    for idx, t in enumerate(t_steps[1:], start=1):
        x_prev = analytical_traj[idx - 1]
        t_batch = t.expand(args.viz_samples)
        delta = analytical_flow(x_prev, t_batch)
        dt = 1.0 / (steps - 1)
        analytical_traj[idx] = x_prev + delta * dt

    # Calculate trajectories using the neural network
    nn_traj = integrate_flow(flow_nn, x0_samples, steps=steps)

    # Create visualization
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 4, 1], wspace=0.0)

    # Generate x values for the Gaussian PDF
    x_vals = np.linspace(-3, 3, 1000)
    gaussian_pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x_vals**2)

    # Left distribution (x_0)
    ax_left = fig.add_subplot(gs[0])
    ax_left.fill_betweenx(x_vals, 0, -gaussian_pdf, color="steelblue", alpha=0.7)
    ax_left.set_xlim(-0.5, 0)
    ax_left.set_ylim(-3, 3)
    ax_left.set_title("$x_0 \\sim N(0, 1)$", fontsize=16)
    ax_left.get_xaxis().set_visible(False)
    ax_left.set_ylabel("Position $x(t)$", fontsize=16)

    # Main plot for trajectories
    ax_main = fig.add_subplot(gs[1], sharey=ax_left)

    # Plot analytical trajectories
    for i in range(args.viz_samples):
        ax_main.plot(
            t_steps.cpu(),
            analytical_traj[:, i].cpu(),
            color="black",
            alpha=0.8,
            linewidth=1,
        )

    # Plot neural network trajectories
    for i in range(args.viz_samples):
        ax_main.plot(
            t_steps.cpu(), nn_traj[:, i].cpu(), color="red", alpha=1, linewidth=1
        )

    ax_main.scatter(
        torch.zeros_like(x0_samples).cpu(),
        x0_samples.cpu(),
        color="steelblue",
        edgecolor="white",
        zorder=5,
        s=30,
        alpha=0.8,
    )

    # Use the actual final points from the trajectories
    x_final_analytical = analytical_traj[-1].cpu()
    x_final_nn = nn_traj[-1].cpu()

    ax_main.scatter(
        torch.ones_like(x_final_analytical),
        x_final_analytical,
        color="gray",
        edgecolor="white",
        zorder=5,
        s=30,
        alpha=0.8,
    )
    ax_main.scatter(
        torch.ones_like(x_final_nn),
        x_final_nn,
        color="red",
        edgecolor="white",
        zorder=5,
        s=30,
        alpha=0.8,
    )

    ax_main.set_xlabel("Time $t$", fontsize=18)
    ax_main.set_title("Flow Between Gaussian Distributions", fontsize=20, pad=20)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(-3, 3)
    ax_main.set_ylabel("")

    # Right distribution (x_1)
    ax_right = fig.add_subplot(gs[2], sharey=ax_left)
    ax_right.fill_betweenx(x_vals, 0, gaussian_pdf, color="salmon", alpha=0.7)
    ax_right.set_xlim(0, 0.5)
    ax_right.set_ylim(-3, 3)
    ax_right.set_title("$x_1 \\sim N(0, 1)$", fontsize=16)
    ax_right.get_xaxis().set_visible(False)
    ax_right.set_ylabel("")
    ax_right.yaxis.set_ticklabels([])

    handles = [
        plt.Line2D([0], [0], color="steelblue", lw=4, label="Initial Distribution"),
        plt.Line2D([0], [0], color="salmon", lw=4, label="Target Distribution"),
        plt.Line2D(
            [0], [0], color="black", lw=1, alpha=0.5, label="Analytical Trajectories"
        ),
        plt.Line2D(
            [0], [0], color="red", lw=1, alpha=0.5, label="Neural Network Trajectories"
        ),
    ]
    ax_main.legend(handles=handles, loc="upper right", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_PATH, "flow_comparison_1d.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Compare vector fields
    if args.compare_vector_fields:
        plt.figure(figsize=(10, 6))
        xs = torch.linspace(-3, 3, 100).to(DEVICE)

        # Sample a few t values
        t_values = [0.2, 0.5, 0.8]
        colors = ["blue", "green", "purple"]

        for t_val, color in zip(t_values, colors):
            t = torch.full_like(xs, t_val)

            # Analytical vector field
            v_analytical = analytical_flow(xs, t)

            # Neural network vector field
            with torch.no_grad():
                v_nn = flow_nn(xs, t)

            plt.plot(
                xs.cpu(),
                v_analytical.cpu(),
                color=color,
                linestyle="-",
                label=f"Analytical (t={t_val})",
                linewidth=2,
            )
            plt.plot(
                xs.cpu(),
                v_nn.cpu(),
                color=color,
                linestyle="--",
                label=f"Neural Network (t={t_val})",
                linewidth=2,
            )

        plt.grid(True, alpha=0.3)
        plt.title("Comparison of Vector Fields", fontsize=18)
        plt.xlabel("Position $x$", fontsize=14)
        plt.ylabel("Vector Field $v(x, t)$", fontsize=14)
        plt.legend()
        plt.savefig(
            os.path.join(SAVE_PATH, "vector_field_comparison_1d.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--viz_samples", type=int, default=20, help="Number of samples to visualize"
    )
    parser.add_argument("--train", action="store_true", help="Train the flow model")
    parser.add_argument(
        "--epochs", type=int, default=2000, help="Number of training steps"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension of the model"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument(
        "--compare_vector_fields",
        action="store_true",
        help="Compare vector fields of analytical and neural solutions",
    )

    args = parser.parse_args()
    main(args)
