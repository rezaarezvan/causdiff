import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from causdiff import DEVICE, SAVE_PATH, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)


def create_mixture_gaussians(K=3, d=2):
    """
    Create a mixture of Gaussians with K components in d dimensions.
    Returns means, covariances, and mixture weights.
    """
    mu0 = torch.zeros(d).to(DEVICE)
    Sigma0 = torch.eye(d).to(DEVICE)

    means = []
    covs = []

    for _ in range(K):
        # Random mean in [-3, 3]^d
        mu = 3.0 * (2.0 * torch.rand(d) - 1.0).to(DEVICE)
        means.append(mu)

        # Random positive definite covariance
        A = torch.randn(d, d).to(DEVICE)

        # Make diagonally dominant
        Sigma = 0.5 * (A @ A.T) + 0.5 * torch.eye(d).to(DEVICE)
        covs.append(Sigma)

    # Uniform mixture weights
    pi = torch.ones(K).to(DEVICE) / K

    return mu0, Sigma0, means, covs, pi


def compute_responsibilities(xt, t, mu0, Sigma0, means, covs, pi):
    """
    Compute the responsibilities gamma_k(x_t, t) for each component k.
    """
    K = len(means)
    d = mu0.shape[0]

    # Compute the density for each component
    densities = torch.zeros(K).to(DEVICE)
    for k in range(K):
        # Mean and covariance of x_t given K=k
        mut = (1 - t) * mu0 + t * means[k]
        Sigmat = (1 - t) ** 2 * Sigma0 + t**2 * covs[k]

        # Compute Gaussian density
        diff = (xt - mut).unsqueeze(0)
        inv_Sigmat = torch.inverse(Sigmat)
        exponent = -0.5 * diff @ inv_Sigmat @ diff.T
        normalizer = 1.0 / torch.sqrt(
            (2 * torch.tensor(np.pi)) ** d * torch.det(Sigmat)
        )
        densities[k] = pi[k] * normalizer * torch.exp(exponent)

    # Normalize to get responsibilities
    gamma = densities / densities.sum()
    return gamma


def compute_drift(xt, t, mu0, Sigma0, means, covs, pi):
    """
    Compute the drift v(x_t, t) for the mixture of Gaussians.
    v(x_t, t) = Sigma_k gamma_k(x_t, t) [mu_k - mu_0 + A_k(t)(x_t - (1-t)mu_0 - tmu_k)]
    """
    K = len(means)
    gamma = compute_responsibilities(xt, t, mu0, Sigma0, means, covs, pi)

    # Compute drift for each component and weight by responsibilities
    drift = torch.zeros_like(xt)
    for k in range(K):
        # Compute A_k(t) = t Sigma_k[(1-t)^2 Sigma_0 + t^2 Sigma_k]^-1
        inv_term = torch.inverse((1 - t) ** 2 * Sigma0 + t**2 * covs[k])
        A_k = t * covs[k] @ inv_term

        # Compute the deviation term
        delta_k = xt - (1 - t) * mu0 - t * means[k]

        component_drift = (means[k] - mu0) + A_k @ delta_k
        drift += gamma[k] * component_drift

    return drift


def compute_jacobian_observation(xt, t, mu0, Sigma0, means, covs, pi, epsilon=1e-6):
    """
    Compute the Jacobian of the drift field under observation using finite differences.
    """
    d = xt.shape[0]
    jacobian = torch.zeros(d, d).to(DEVICE)

    # Base drift at x_t
    v_base = compute_drift(xt, t, mu0, Sigma0, means, covs, pi)

    # Compute each column of the Jacobian using finite differences
    for i in range(d):
        # Create perturbation in direction i
        delta = torch.zeros_like(xt)
        delta[i] = epsilon

        # Compute drift at perturbed point
        v_perturbed = compute_drift(xt + delta, t, mu0, Sigma0, means, covs, pi)

        jacobian[:, i] = (v_perturbed - v_base) / epsilon

    return jacobian


def compute_jacobian_intervention(xt, t, mu0, Sigma0, means, covs, pi):
    """
    Compute the Jacobian under intervention
    Under intervention, we fix the component K and only use A_k(t).
    """
    K = len(means)
    d = mu0.shape[0]

    gamma = compute_responsibilities(xt, t, mu0, Sigma0, means, covs, pi)

    # For intervention, we use the A_k(t) term weighted by responsibilities
    jacobian = torch.zeros(d, d).to(DEVICE)
    for k in range(K):
        # Compute A_k(t) = t Sigma_k[(1-t)^2 Sigma_0 + t^2 Sigma_k]^-1
        inv_term = torch.inverse((1 - t) ** 2 * Sigma0 + t**2 * covs[k])
        A_k = t * covs[k] @ inv_term
        # Weight by responsibility
        jacobian += gamma[k] * A_k

    return jacobian


def compute_analytic_jacobian(xt, t, mu0, Sigma0, means, covs, pi):
    """
    Compute the analytical Jacobian of the drift using the formula:
    nabla_{x_t} v = Sigma_k gamma_k*A_k(t) +
                    Sigma_k(nabla_{x_t} gamma_k)[mu_k-mu_0+A_k(t)*Delta_k]
    """
    K = len(means)
    d = mu0.shape[0]
    gamma = compute_responsibilities(xt, t, mu0, Sigma0, means, covs, pi)

    # First term: weighted sum of A_k(t)
    jacobian = torch.zeros(d, d).to(DEVICE)

    # Calculate all the A_k(t) and Delta_k values
    A_matrices = []
    delta_vectors = []
    m_vectors = []
    S_matrices = []
    term_vectors = []

    for k in range(K):
        # Compute A_k(t)
        S_k = (1 - t) ** 2 * Sigma0 + t**2 * covs[k]
        S_matrices.append(S_k)

        inv_S_k = torch.inverse(S_k)
        A_k = t * covs[k] @ inv_S_k
        A_matrices.append(A_k)

        # Compute mean
        m_k = (1 - t) * mu0 + t * means[k]
        m_vectors.append(m_k)

        # Compute Delta_k
        delta_k = xt - m_k
        delta_vectors.append(delta_k)

        # Compute [mu_k-mu_0 + A_k(t)*Delta_k]
        term_vectors.append((means[k] - mu0) + A_k @ delta_k)

        # Add to first term
        jacobian += gamma[k] * A_k

    # Second term: contribution from Delta_{x_t} gamma_k
    for k in range(K):
        # Compute gradient of gamma_k with respect to x_t
        grad_gamma_k = torch.zeros(d).to(DEVICE)

        # Direct gradient computation for softmax of quadratics
        S_k_inv = torch.inverse(S_matrices[k])
        weighted_sum = torch.zeros(d).to(DEVICE)

        for j in range(K):
            S_j_inv = torch.inverse(S_matrices[j])
            weighted_sum += gamma[j] * (S_j_inv @ (m_vectors[j] - xt))

        # nabla_{x_t} gamma_k = gamma_k[S_k^-1 (m_k-x_t) - Sigma_j gamma_j*S_j^-1(m_j-x_t)]
        grad_gamma_k = gamma[k] * (S_k_inv @ (m_vectors[k] - xt) - weighted_sum)

        # Outer product to convert to matrix
        jacobian += torch.outer(term_vectors[k], grad_gamma_k)

    return jacobian


def plot_confidence_ellipse(ax, mean, cov, n_std=2.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse.

    Args:
        ax (matplotlib.axes.Axes)
            The axes object to draw the ellipse into.
        mean (array-like, shape (2, ))
            The location of the center of the ellipse.
        cov (array-like, shape (2, 2))
            The covariance matrix to base the ellipse on.
        n_std (float)
            The number of standard deviations to determine the ellipse's radiuses.
        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

    Returns:
        matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Scale the ellipse's width and height by the standard deviation
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean[0], mean[1])
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def compare_jacobians(mu0, Sigma0, means, covs, pi, t, d=2, num_points=10):
    """
    Compare numerical and analytical Jacobians to verify the implementation.
    """
    # Generate random points for testing
    points = 5.0 * torch.randn(num_points, d).to(DEVICE)

    numerical_jacs = []
    analytical_jacs = []
    intervention_jacs = []

    for xt in points:
        # Compute the three types of Jacobians
        numerical_jac = compute_jacobian_observation(
            xt, t, mu0, Sigma0, means, covs, pi
        )
        analytical_jac = compute_analytic_jacobian(xt, t, mu0, Sigma0, means, covs, pi)
        intervention_jac = compute_jacobian_intervention(
            xt, t, mu0, Sigma0, means, covs, pi
        )

        numerical_jacs.append(numerical_jac.cpu().numpy())
        analytical_jacs.append(analytical_jac.cpu().numpy())
        intervention_jacs.append(intervention_jac.cpu().numpy())

    # Compute Frobenius norm differences
    num_vs_an_diffs = []
    int_vs_an_diffs = []

    for i in range(num_points):
        num_vs_an_diff = np.linalg.norm(numerical_jacs[i] - analytical_jacs[i], "fro")
        int_vs_an_diff = np.linalg.norm(
            intervention_jacs[i] - analytical_jacs[i], "fro"
        )

        num_vs_an_diffs.append(num_vs_an_diff)
        int_vs_an_diffs.append(int_vs_an_diff)

        print(f"Point {i + 1}:")
        print(f"  Numerical vs. Analytical: {num_vs_an_diff:.6f}")
        print(f"  Intervention vs. Analytical: {int_vs_an_diff:.6f}")

        # Find the responsibilities for this point
        gamma = compute_responsibilities(xt, t, mu0, Sigma0, means, covs, pi)
        dominant_comp = gamma.argmax().item()
        print(
            f"  Dominant component: {dominant_comp + 1} (γ = {
                gamma[dominant_comp]:.4f
            })"
        )

        # Show the actual Jacobians for inspection
        print(f"  Numerical Jacobian:\n{numerical_jacs[i]}")
        print(f"  Analytical Jacobian:\n{analytical_jacs[i]}")
        print(f"  Intervention Jacobian:\n{intervention_jacs[i]}")
        print()

    print(
        f"Average Numerical vs. Analytical difference: {np.mean(num_vs_an_diffs):.6f}"
    )
    print(
        f"Average Intervention vs. Analytical difference: {
            np.mean(int_vs_an_diffs):.6f
        }"
    )


def create_summary_table(results_by_t):
    """
    Create and save a summary table of Jacobian analysis results.

    Args:
        results_by_t: Dictionary with t values as keys and results as values
    """
    # Table header
    table_lines = [
        "\\begin{tabular}{ccc}",
        "    \\toprule",
        "    \\textbf{Time $t$} & \\textbf{Numerical vs. Analytical} & \\textbf{Intervention vs. Analytical} \\\\",
        "    \\midrule",
    ]

    # Add a row for each time step
    for t in sorted(results_by_t.keys()):
        result = results_by_t[t]
        num_vs_an = result["num_vs_an"]
        int_vs_an = result["int_vs_an"]

        # Format very small values as <0.001
        int_vs_an_str = f"{int_vs_an:.3f}" if int_vs_an >= 0.001 else "$<$0.001"

        table_lines.append(f"    {t:.2f} & {num_vs_an:.3f} & {int_vs_an_str} \\\\")

    # Table footer
    table_lines.extend(["    \\bottomrule", "\\end{tabular}"])

    # Write to file
    with open(f"{SAVE_PATH}/jacobian_table.tex", "w") as f:
        f.write("\n".join(table_lines))

    print(f"Table saved to {SAVE_PATH}/jacobian_table.tex")


def create_composite_figure(time_steps):
    """
    Create a composite figure showing the evolution of drift fields and Jacobians.

    Args:
        time_steps: List of time values to include
    """

    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(2, len(time_steps), figure=fig)

    # Add each time step as a column
    for i, t in enumerate(time_steps):
        # Load the drift field image
        drift_img = plt.imread(f"{SAVE_PATH}/mog_drift_field_t_{t:.2f}.png")
        jac_img = plt.imread(f"{SAVE_PATH}/mog_jacobians_t_{t:.2f}.png")

        # Add drift field
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(drift_img)
        ax1.axis("off")
        ax1.set_title(f"t = {t:.2f}")

        # Add Jacobian comparison
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(jac_img)
        ax2.axis("off")

        # For the first column, add row labels
        if i == 0:
            ax1.text(
                -0.1,
                0.5,
                "Drift Field",
                transform=ax1.transAxes,
                ha="right",
                va="center",
                fontsize=12,
                rotation=90,
            )
            ax2.text(
                -0.1,
                0.5,
                "Jacobian Comparison",
                transform=ax2.transAxes,
                ha="right",
                va="center",
                fontsize=12,
                rotation=90,
            )

    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}/mog_time_evolution.png", dpi=300, bbox_inches="tight")
    print(f"Composite figure saved to {SAVE_PATH}/mog_time_evolution.png")


def create_drift_field_evolution(time_steps):
    """
    Create a composite figure showing only the drift field evolution.

    Args:
        time_steps: List of time values to include
    """
    fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 3))

    for i, t in enumerate(time_steps):
        ax = axes[i]

        # Create drift field visualization for this time step
        K = 3  # Number of components
        d = 2  # Dimensionality
        mu0, Sigma0, means, covs, pi = create_mixture_gaussians(K=K, d=d)

        # Create a grid of points
        # Use fewer points for cleaner visualization
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y)

        # Convert to tensor points
        points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
        ).to(DEVICE)

        # Compute drift for each point
        drifts = []
        for j in range(len(points)):
            xt = points[j]
            drift = compute_drift(xt, t, mu0, Sigma0, means, covs, pi)
            drifts.append(drift.cpu().numpy())
        drifts = np.array(drifts)

        # Plot the vector field
        ax.quiver(
            X.flatten(),
            Y.flatten(),
            drifts[:, 0],
            drifts[:, 1],
            color="blue",
            alpha=0.6,
            scale=30,  # Adjust scale for visibility
        )

        # Plot the Gaussian components
        for k in range(len(means)):
            mu_k = means[k].cpu().numpy()
            sigma_k = covs[k].cpu().numpy()
            mu_t = (1 - t) * mu0.cpu().numpy() + t * mu_k
            sigma_t = (1 - t) ** 2 * Sigma0.cpu().numpy() + t**2 * sigma_k

            # Plot confidence ellipse
            plot_confidence_ellipse(
                ax,
                mu_t,
                sigma_t,
                n_std=2.0,
                edgecolor=f"C{k}",
                linewidth=2,
            )

            # Plot mean location
            ax.scatter(mu_t[0], mu_t[1], color=f"C{k}", s=50, marker="o")

        # Set labels and title
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)

        # Only add x and y labels to the first subplot
        if i == 0:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}/drift_field_evolution.png", dpi=300, bbox_inches="tight")
    print(f"Drift field evolution saved to {SAVE_PATH}/drift_field_evolution.png")


def create_jacobian_comparison(t_values):
    """
    Create clearer Jacobian comparison figures for specific time steps.

    Args:
        t_values: List of time values to create comparisons for
    """

    # Create a mixture of K Gaussians
    K = 3
    d = 2
    mu0, Sigma0, means, covs, pi = create_mixture_gaussians(K=K, d=d)

    for t in t_values:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Create a grid of points
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y)

        # Convert to tensor points
        points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
        ).to(DEVICE)

        # Compute drift for each point
        drifts = []
        for j in range(len(points)):
            xt = points[j]
            drift = compute_drift(xt, t, mu0, Sigma0, means, covs, pi)
            drifts.append(drift.cpu().numpy())
        drifts = np.array(drifts)

        # Plot the vector field in both subplots
        ax1.quiver(
            X.flatten(),
            Y.flatten(),
            drifts[:, 0],
            drifts[:, 1],
            color="blue",
            alpha=0.3,
            scale=30,
        )

        ax2.quiver(
            X.flatten(),
            Y.flatten(),
            drifts[:, 0],
            drifts[:, 1],
            color="blue",
            alpha=0.3,
            scale=30,
        )

        # Plot the Gaussian components in both subplots
        for k in range(len(means)):
            mu_k = means[k].cpu().numpy()
            sigma_k = covs[k].cpu().numpy()
            mu_t = (1 - t) * mu0.cpu().numpy() + t * mu_k
            sigma_t = (1 - t) ** 2 * Sigma0.cpu().numpy() + t**2 * sigma_k

            # Plot confidence ellipse
            plot_confidence_ellipse(
                ax1,
                mu_t,
                sigma_t,
                n_std=2.0,
                edgecolor=f"C{k}",
                linewidth=2,
            )

            plot_confidence_ellipse(
                ax2,
                mu_t,
                sigma_t,
                n_std=2.0,
                edgecolor=f"C{k}",
                linewidth=2,
            )

        # Select a subset of points for Jacobian visualization
        subset_indices = np.linspace(0, len(points) - 1, 10, dtype=int)
        subset_points = points[subset_indices]

        # Visualize the Jacobians for these points
        for xt in subset_points:
            # Generate perturbation directions
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                # Unit vector in direction 'angle'
                v = torch.tensor([np.cos(angle), np.sin(angle)], device=DEVICE)
                v_scaled = 0.5 * v  # Scale for visualization

                # Compute Jacobians
                obs_jac = compute_jacobian_observation(
                    xt, t, mu0, Sigma0, means, covs, pi
                )
                int_jac = compute_jacobian_intervention(
                    xt, t, mu0, Sigma0, means, covs, pi
                )

                # Apply Jacobians to compute effects
                # v is double, obs_jac is float, make v float
                if v.dtype != obs_jac.dtype:
                    v = v.float()
                assert v.dtype == obs_jac.dtype, "Mismatch in dtype"
                obs_effect = obs_jac @ v
                int_effect = int_jac @ v

                # Plot in both subplots
                xt_np = xt.cpu().numpy()
                v_scaled_np = v_scaled.cpu().numpy()

                # Observation Jacobian (left subplot)
                ax1.arrow(
                    xt_np[0],
                    xt_np[1],
                    v_scaled_np[0],
                    v_scaled_np[1],
                    color="red",
                    alpha=0.6,
                    width=0.02,
                )

                ax1.arrow(
                    xt_np[0] + v_scaled_np[0],
                    xt_np[1] + v_scaled_np[1],
                    obs_effect[0].item(),
                    obs_effect[1].item(),
                    color="green",
                    alpha=0.8,
                    width=0.02,
                )

                # Intervention Jacobian (right subplot)
                ax2.arrow(
                    xt_np[0],
                    xt_np[1],
                    v_scaled_np[0],
                    v_scaled_np[1],
                    color="red",
                    alpha=0.6,
                    width=0.02,
                )

                ax2.arrow(
                    xt_np[0] + v_scaled_np[0],
                    xt_np[1] + v_scaled_np[1],
                    int_effect[0].item(),
                    int_effect[1].item(),
                    color="purple",
                    alpha=0.8,
                    width=0.02,
                )

        # Set titles and limits
        ax1.set_title("Observation Jacobian (Full Nonlinear)")
        ax2.set_title("Intervention Jacobian (Fixed Component)")

        ax1.set_xlim(-6, 6)
        ax1.set_ylim(-6, 6)
        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)

        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax2.set_xlabel("$x_1$")

        # Create legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0], [0], color="blue", marker="", linestyle="-", label="Drift Field"
            ),
            Line2D(
                [0],
                [0],
                color="red",
                marker="",
                linestyle="-",
                label="Perturbation Direction",
            ),
            Line2D(
                [0],
                [0],
                color="green",
                marker="",
                linestyle="-",
                label="Observation Effect",
            ),
            Line2D(
                [0],
                [0],
                color="purple",
                marker="",
                linestyle="-",
                label="Intervention Effect",
            ),
        ]

        ax1.legend(handles=legend_elements, loc="upper right")
        ax2.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig(
            f"{SAVE_PATH}/observation_intervention_t{t:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Jacobian comparison saved to {SAVE_PATH}/observation_intervention_t{
                t:.2f
            }.png"
        )


def main():
    """
    Main function to demonstrate mixture of Gaussians flow matching.
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    print("Demonstrating flow matching for mixture of Gaussians")
    print(f"Results will be saved to {SAVE_PATH}")

    # Create a mixture of K Gaussians
    K = 3
    d = 2
    mu0, Sigma0, means, covs, pi = create_mixture_gaussians(K=K, d=d)

    # Dictionary to store results for each time step
    results_by_t = {}

    ts = [0.0, 0.25, 0.5, 0.75, 0.99]

    # Generate the numerical results
    for t in ts:
        # Run comparative analysis
        numerical_diffs, intervention_diffs, dom_comps, gamma_vals = [], [], [], []

        # Run multiple points evaluation
        for _ in range(10):
            xt = torch.randn(d).to(DEVICE)  # Random test point
            numerical_jac = compute_jacobian_observation(
                xt, t, mu0, Sigma0, means, covs, pi
            )
            analytical_jac = compute_analytic_jacobian(
                xt, t, mu0, Sigma0, means, covs, pi
            )
            intervention_jac = compute_jacobian_intervention(
                xt, t, mu0, Sigma0, means, covs, pi
            )

            # Compute differences
            num_vs_an_diff = torch.norm(numerical_jac - analytical_jac, "fro").item()
            int_vs_an_diff = torch.norm(intervention_jac - analytical_jac, "fro").item()

            # Compute responsibilities
            gamma = compute_responsibilities(xt, t, mu0, Sigma0, means, covs, pi)
            dominant_comp = gamma.argmax().item() + 1  # 1-indexed

            numerical_diffs.append(num_vs_an_diff)
            intervention_diffs.append(int_vs_an_diff)
            dom_comps.append(dominant_comp)
            gamma_vals.append(gamma[dominant_comp - 1].item())

        # Store average results
        results_by_t[t] = {
            "num_vs_an": sum(numerical_diffs) / len(numerical_diffs),
            "int_vs_an": sum(intervention_diffs) / len(intervention_diffs),
            # Most common
            "dominant_comp": max(set(dom_comps), key=dom_comps.count),
            "gamma_val": sum(gamma_vals) / len(gamma_vals),
        }

        print(
            f"Time t={t:.2f}: Numerical vs. Analytical: {results_by_t[t]['num_vs_an']:.6f}, "
            f"Intervention vs. Analytical: {results_by_t[t]['int_vs_an']:.6f}"
        )

    # Create improved visualizations
    create_drift_field_evolution(ts)

    # Create Jacobian comparisons for interesting time steps
    create_jacobian_comparison([0.50, 0.75])

    # Create the LaTeX table
    create_summary_table(results_by_t)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
