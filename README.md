# causdiff

Code and paper artifacts for **Causality--Δ: Jacobian-Based Dependency Analysis in Flow Matching Models**.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Using `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Reproducing Core Figures

All scripts assume working directory at the repo root and will download datasets via `torchvision` when available.

### 1D Gaussian (Figure: flow comparison)
```bash
python scripts/2D/eval/gaussian_1D_neural.py --train
```
Outputs: `result/gaussian_1D_neural/flow_comparison_1d.png`

### Mixture of Gaussians (drift field evolution)
```bash
python scripts/2D/eval/mixture_gaussian.py
```
Outputs: `result/mixture_gaussian/drift_field_evolution.png`

### MNIST pixel correlation (fives)
```bash
python scripts/image/train/mnist_digit.py
python scripts/image/eval/fives_correlation.py \
  --model result/mnist_digit/unet_mnist_digit_weights_final.pt
```
Outputs: `result/fives_correlation/dx_dz_last_iteration.png`,
`result/fives_correlation/dx_dz_scatter.png`,
`result/fives_correlation/joint_distribution_mnist_fives.png`

### CelebA attribute analysis
Train the flow model (FlowUNet) and attribute classifier:
```bash
python scripts/image/train/celeba.py
python scripts/image/train/ResNet_Attributes_classifier_celebA.py
```
Then run the JVP-based correlation analysis:
```bash
python scripts/image/eval/celeb_attributes_logits.py \
  --flow_model result/celeba/unet_celeba_weights_final.pt \
  --classifier_ckpt result/ResNet_Attributes_classifier_celebA/ckpt_step_40.pt
```
Outputs: `result/celeb_attributes_logits/corr_diff.png`,
`result/celeb_attributes_logits/corr_diff_heavy_makeup.png`

Note: CelebA download may require manual acceptance in `torchvision`.

## Paper
The LaTeX sources are in `paper/`. To compile:
```bash
cd paper
pdflatex sample_paper.tex
bibtex sample_paper
pdflatex sample_paper.tex
pdflatex sample_paper.tex
```

## License
MIT. See `LICENSE`.
