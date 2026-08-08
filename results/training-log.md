# Training log

Run of the `haskell-mnist-train` executable (`app/Main.hs`) on the standard
MNIST dataset. The network is a two-layer feedforward classifier
`784 → 64 → 10` with a ReLU hidden layer and a softmax + cross-entropy output,
trained with **online (per-example) SGD** — one gradient step per image, no
mini-batching.

Setup:

| setting            | value                                             |
| ------------------ | ------------------------------------------------- |
| architecture       | 784 → 64 (ReLU) → 10 (softmax)                    |
| weight init        | He-uniform `U(-√(6/fanIn), √(6/fanIn))`, biases 0 |
| examples per epoch | 60 000 (full train set)                           |
| test examples      | 10 000 (full test set)                            |
| epochs             | 25                                                |
| learning rate      | 0.05                                              |
| RNG seed           | 42                                                |

Accuracy is measured on the full 10 000-image test set after each epoch; "mean
loss" is the average per-example cross-entropy over the epoch, measured against
each example's pre-update parameters.

## Training run — 25 epochs

```
Initial test accuracy: 10.08%
epoch  1: mean loss 0.3871, test accuracy 92.17%
epoch  2: mean loss 0.2881, test accuracy 92.60%
epoch  3: mean loss 0.2661, test accuracy 94.07%
epoch  4: mean loss 0.2448, test accuracy 94.15%
epoch  5: mean loss 0.2376, test accuracy 94.00%
epoch  6: mean loss 0.2267, test accuracy 91.93%
epoch  7: mean loss 0.2159, test accuracy 94.81%
epoch  8: mean loss 0.2155, test accuracy 92.93%
epoch  9: mean loss 0.2143, test accuracy 94.51%
epoch 10: mean loss 0.1994, test accuracy 94.04%
epoch 11: mean loss 0.1916, test accuracy 94.28%
epoch 12: mean loss 0.1988, test accuracy 94.13%
epoch 13: mean loss 0.2112, test accuracy 93.21%
epoch 14: mean loss 0.1894, test accuracy 94.22%
epoch 15: mean loss 0.1795, test accuracy 93.64%
epoch 16: mean loss 0.1699, test accuracy 94.59%
epoch 17: mean loss 0.1739, test accuracy 94.63%
epoch 18: mean loss 0.1564, test accuracy 94.60%
epoch 19: mean loss 0.1587, test accuracy 94.90%
epoch 20: mean loss 0.1714, test accuracy 94.61%
epoch 21: mean loss 0.1603, test accuracy 95.37%
epoch 22: mean loss 0.1703, test accuracy 95.09%
epoch 23: mean loss 0.1725, test accuracy 94.35%
epoch 24: mean loss 0.1668, test accuracy 95.32%
epoch 25: mean loss 0.1629, test accuracy 95.49%
```

Test accuracy climbs from **10.08%** at initialisation (near the 1/10 chance
classification rate) to **95.49%** after 25 epochs,
while the mean per-example cross-entropy falls from 0.3871 to 0.1629. The
network is already on its ~94–95% plateau by epoch 3; from there the
epoch-to-epoch accuracy oscillates (e.g. a dip to 91.93% at epoch 6), as online
SGD with a fixed learning rate need not improve the test metric monotonically.

## Confusion matrix

The confusion matrix over the full test set is in
[`confusion-matrix.csv`](confusion-matrix.csv) (row = true label, column =
predicted label). Its diagonal sums to **9 549 / 10 000**, consistent with the
95.49% final accuracy.

The errors are not spread uniformly. The most confused digit pairs are
4→9 (28), 6→5 (28), 2→8 (26), 3→8 (24), 9→8 (24), 7→9 (21), and 5→8 (20) —
the standout is the digit **8**: it attracts **127 false positives** (2, 3, 5
and 9 are all frequently misread as 8 in this run), giving it the lowest
precision of any class (~87.9%). By recall, the hardest digits are 7 (93.8%),
6 (94.0%) and 2 (94.3%); the easiest are 0 (98.9%) and 1 (98.7%).

## Implementation details

### Compiler and build tool
- **GHC:** 9.4.8
- **Build tool:** Stack (resolver lts-21.25)
- **Compilation flags:** `-Wall` package-wide; training executable built with `-O2`
- **Package pragmas:** `DataKinds`, `GADTs`, `KindSignatures`, `TypeFamilies`,
  `TypeOperators`, `ScopedTypeVariables`, `FlexibleContexts`, `FlexibleInstances`,
  `RankNTypes`

### Network architecture (production)
- Input layer: 784 (28×28 MNIST images, normalized to [0, 1])
- Hidden layer: 64 neurons with ReLU activation
- Output layer: 10 neurons with softmax (cross-entropy loss)
- Parameter count: 784×64 + 64 + 64×10 + 10 = 50,890

### Dataset processing
- Training set: 60,000 examples (full MNIST train set)
- Test set: 10,000 examples (full MNIST test set)
- Pixel normalization: each byte ∈ [0, 255] divided by 255.0 → [0, 1]
- Training order: fixed (no reshuffling between epochs)
- One-hot encoding: class labels encoded as standard basis vectors

The SHA-256 checksums of the four uncompressed IDX files used by this run are:

```text
ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db  train-images-idx3-ubyte
65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5  train-labels-idx1-ubyte
0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7  t10k-images-idx3-ubyte
ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2  t10k-labels-idx1-ubyte
```

### Training procedure
- Optimizer: online SGD (one gradient step per example)
- Loss function: cross-entropy(softmax(z₂), y)
- Backward pass: hand-written, type-indexed (not automatic differentiation)
- Dropout/regularization: none

## Notes

- Hyperparameters live as top-level constants at the top of `app/Main.hs`
  (`learningRate`, `epochs`, `trainSize`, `testSize`, `seed`) — adjust and
  rebuild to reproduce or extend this run.
- The training metrics and confusion matrix are produced from a **single archived
  execution** with seed 42, ensuring that the reported metrics and matrix come
  from the same network state.
