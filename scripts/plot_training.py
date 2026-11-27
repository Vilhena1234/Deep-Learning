import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path(sys.argv[1]).expanduser()
OUT = Path(sys.argv[2]).expanduser()
OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 200


def read_losses(name):
    text = (LOGS / name).read_text()
    return np.array([float(x) for x in re.findall(r"Step \d+: training loss=([\d.]+)", text)])


def smooth(values, window=WINDOW):
    return np.convolve(values, np.ones(window) / window, mode="valid")


runs = [
    ("Batch 16, 3 epochs", "dpr_30359806.out", 16, "tab:blue"),
    ("Batch 16, 1 epoch", "dpr_30352660.out", 16, "tab:orange"),
    ("Batch 8, 1 epoch", "dpr_30364503.out", 8, "tab:green"),
]

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, name, batch, colour in runs:
    losses = read_losses(name)
    steps = np.arange(WINDOW, len(losses) + 1)
    ax.plot(steps, smooth(losses), label=label, color=colour, linewidth=1.5)
ax.axhline(math.log(16), color="grey", linestyle="--", linewidth=1)
ax.text(14793, math.log(16) * 1.15, "guessing, batch 16: ln 16 = 2.77", ha="right", fontsize=8, color="grey")
ax.axhline(math.log(8), color="grey", linestyle=":", linewidth=1)
ax.text(14793, math.log(8) * 0.62, "guessing, batch 8: ln 8 = 2.08", ha="right", fontsize=8, color="grey")
for step in (4931, 9862):
    ax.axvline(step, color="black", alpha=0.15, linewidth=1)
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel(f"training loss (mean of {WINDOW} steps)")
ax.set_title("Training loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "training_loss.png", dpi=200)

labels = ["Pretrained", "Batch 16\n1 epoch", "Batch 8\n1 epoch", "Batch 16\nepoch 1 of 3", "Batch 16\nepoch 2 of 3", "Batch 16\nepoch 3 of 3"]
mrr = [0.0998, 0.8349, 0.8466, 0.8454, 0.8722, 0.8903]
colours = ["tab:grey", "tab:orange", "tab:green", "tab:blue", "tab:blue", "tab:blue"]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(range(len(mrr)), mrr, color=colours)
for bar, value in zip(bars, mrr):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(mrr)), labels, fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("validation MRR")
ax.set_title("Validation MRR (8,692 questions, 1,889 passages)")
fig.tight_layout()
fig.savefig(OUT / "validation_mrr.png", dpi=200)
STEPS = 4931
bert_runs = [
    ("A: own encoder, mean", "dpr_30383528.out", 0, "tab:blue"),
    ("B: Hugging Face BERT, mean", "dpr_30383528.out", 1, "tab:orange"),
    ("C: own encoder, CLS", "dpr_30383528.out", 2, "tab:green"),
]
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, name, part, colour in bert_runs:
    losses = read_losses(name)[part * STEPS:(part + 1) * STEPS]
    steps = np.arange(WINDOW, len(losses) + 1)
    ax.plot(steps, smooth(losses), label=label, color=colour, linewidth=1.5)
ax.axhline(math.log(16), color="grey", linestyle="--", linewidth=1)
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel(f"training loss (mean of {WINDOW} steps)")
ax.set_title("Training loss, BERT-base encoders (batch 16, 1 epoch)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "training_loss_bert.png", dpi=200)

print("saved", sorted(p.name for p in OUT.iterdir()))
