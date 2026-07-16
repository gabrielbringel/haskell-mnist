#!/usr/bin/env python3
"""Generates the Results figures for the haskell-mnist paper.

Figures:
  results/fig-training.pdf   — two stacked panels: (a) loss, (b) accuracy per
                               epoch, for the full 25-epoch run.
  results/fig-confusion.pdf  — 10x10 confusion matrix over the test set.

Inputs:
  results/training-metrics.csv   (mirrors training-log.md)
  results/confusion-matrix.csv   (10x10 integer grid, written by the executable)

Usage:
  pip install -r requirements.txt
  python scripts/plot_results.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Shared style — sizes that stay legible in a paper column.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.titlepad": 14,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    # Uncomment to match the serif font of the ACM template:
    # "font.family": "serif",
})

# Single curve per panel; solid line with a marker on every epoch.
LINE_LOSS = dict(color="#d6942a", marker="o", markersize=2.5, linewidth=1.3,
                  linestyle="-")
LINE_ACC = dict(color="#2a78d6", marker="o", markersize=2.5, linewidth=1.3,
                 linestyle="-")


def confusion_figure():
    """Writes results/fig-confusion.pdf: the row-normalized confusion matrix
    over the test set, annotated with the raw counts.

    Does nothing (and warns) if confusion-matrix.csv has not been generated
    yet, or if it is not square.
    """
    path = os.path.join(RESULTS_DIR, "confusion-matrix.csv")
    if not os.path.exists(path):
        print(
            "AVISO: {} nao existe. Rode `stack exec haskell-mnist-train` "
            "primeiro (ver prompt-confusion-matrix.md). "
            "Pulando a figura da matriz de confusao.".format(path),
            file=sys.stderr,
        )
        return

    counts = np.loadtxt(path, delimiter=",", dtype=int)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        print("AVISO: confusion-matrix.csv nao e quadrada; pulando.",
              file=sys.stderr)
        return

    n = counts.shape[0]
    row_sums = counts.sum(axis=1, keepdims=True)
    # Normalize by row (per-class recall); guard empty rows so a zero sum stays
    # 0 instead of picking up uninitialized memory.
    norm = np.zeros_like(counts, dtype=float)
    np.divide(counts, row_sums, out=norm, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_title("Confusion matrix (test set, 10,000 images)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Predicted label", fontweight="bold")
    ax.set_ylabel("True label", fontweight="bold")

    # Annotate with the raw count; text color chosen for contrast.
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized frequency")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig-confusion.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Escrito:", out)


def main():
    """Generates every figure of the Results section."""
    training_figure()
    confusion_figure()


def style_axes(ax):
    """Applies the shared line-plot style to `ax`: horizontal-only grid drawn
    behind the data and no top/right spines.
    """
    ax.grid(True, axis="y", color="#d9d9d6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=8)


def training_figure():
    """Writes results/fig-training.pdf: mean training loss and test accuracy
    per epoch, as two vertically stacked panels read from
    results/training-metrics.csv.
    """
    path = os.path.join(RESULTS_DIR, "training-metrics.csv")
    df = pd.read_csv(path).sort_values("epoch")

    # Vertically stacked panels: (a) loss on top, (b) accuracy below.
    # A narrow, tall figure fits one column of the ACM template.
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(3.0, 4.2))

    loss = df.dropna(subset=["loss"])            # no loss at epoch 0
    ax_loss.plot(loss["epoch"], loss["loss"], **LINE_LOSS)
    ax_acc.plot(df["epoch"], df["accuracy"], **LINE_ACC)

    ax_loss.set_title("(a) Mean training loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")

    ax_acc.set_title("(b) Test accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")

    # Shared x-range for both panels: 0 to 25 in steps of 5, no padding.
    max_epoch = int(df["epoch"].max())
    ticks = range(0, max_epoch + 1, 5)
    for ax in (ax_loss, ax_acc):
        ax.set_xlim(0, max_epoch + 1)
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", rotation=45)
        style_axes(ax)

    ax_loss.set_ylim(0.15, 0.35)
    ax_loss.set_yticks(np.arange(0.00, 0.42, 0.10))

    ax_acc.set_ylim(90, 98)
    ax_acc.set_yticks(range(90, 99, 2))

    fig.tight_layout(h_pad=3.0)
    out = os.path.join(RESULTS_DIR, "fig-training.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Escrito:", out)


if __name__ == "__main__":
    main()
