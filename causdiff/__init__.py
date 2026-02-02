import sys
import torch
from pathlib import Path

SEED = 42069
SAVE_PATH = Path("result") / Path(sys.argv[0]).stem
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__all__ = ["DEVICE", "SAVE_PATH", "SEED"]
