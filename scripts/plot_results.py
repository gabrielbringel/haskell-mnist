#!/usr/bin/env python3
"""Gera as figuras de Results do paper haskell-mnist.

Figuras:
  results/fig-training.pdf   — 2 painéis: (a) loss, (b) acurácia por época,
                               comparando lr=0.10 e lr=0.05.
  results/fig-confusion.pdf  — matriz de confusão 10x10 no test set.

Entradas:
  results/training-metrics.csv   (espelha training-log.md)
  results/confusion-matrix.csv   (grade 10x10 de inteiros, gerada pelo executável)

Uso:
  pip install matplotlib pandas numpy
  python scripts/plot_results.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Estilo compartilhado — tamanhos legíveis em coluna de paper.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    # Descomente para casar com a fonte serifada do template ACM:
    # "font.family": "serif",
})

# Cores + marcadores + traço: distinguíveis também em preto-e-branco.
STYLES = {
    "0.10": dict(color="#1b9e77", marker="s", linestyle="--", label="lr = 0.10"),
    "0.05": dict(color="#7570b3", marker="o", linestyle="-",  label="lr = 0.05"),
}


def training_figure():
    path = os.path.join(RESULTS_DIR, "training-metrics.csv")
    df = pd.read_csv(path)
    df["lr_str"] = df["lr"].map(lambda x: f"{x:.2f}")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(7, 2.9))

    for lr_str, style in STYLES.items():
        run = df[df["lr_str"] == lr_str].sort_values("epoch")
        loss = run.dropna(subset=["loss"])           # sem loss na época 0
        ax_loss.plot(loss["epoch"], loss["loss"], **style)
        ax_acc.plot(run["epoch"], run["accuracy"], **style)

    ax_loss.set_title("(a) Mean training loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")

    ax_acc.set_title("(b) Test accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.legend()

    for ax in (ax_loss, ax_acc):
        ax.set_xticks(range(0, 6))
        ax.set_xlim(-0.2, 5.2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig-training.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Escrito:", out)


def confusion_figure():
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
    # Normaliza por linha (recall por classe); protege linhas vazias (soma 0
    # fica 0 em vez de lixo de memoria nao inicializada).
    norm = np.zeros_like(counts, dtype=float)
    np.divide(counts, row_sums, out=norm, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Anota a contagem crua; cor do texto por contraste.
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
    training_figure()
    confusion_figure()


if __name__ == "__main__":
    main()
