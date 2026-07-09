# haskell-mnist

Feedforward neural network trained on MNIST, implemented in Haskell with dependent types. Matrix and vector dimensions are encoded at the type level using DataKinds and GADTs, turning dimensionality errors into compile-time failures. Backpropagation implemented manually as a typed reverse fold.

## Architecture

A two-layer classifier: `784 → 64 (ReLU) → 10 (softmax)`, trained with online
(per-example) stochastic gradient descent and cross-entropy loss.

| module            | responsibility                                                |
| ----------------- | ------------------------------------------------------------- |
| `Vec`             | length-indexed vectors (`Vec n a`)                            |
| `Mat`             | dimension-indexed matrices (`Mat r c a`) + backprop ops       |
| `Activation`      | ReLU / ReLU′ and numerically-stable softmax                   |
| `Loss`            | cross-entropy and its gradient (`yhat - y`)                   |
| `Layer`           | a linear layer with typed `forward` / `backward`              |
| `Network`         | composition of two layers; forward, backprop, `predict`, `accuracy` |
| `Init`            | He-uniform random weight initialisation                       |
| `Train`           | SGD training steps / epochs                                   |
| `MNIST`           | IDX file-format loader                                        |
| `app/Main.hs`     | executable: load data → train → report accuracy              |

Every dimension is a type-level `Nat`, so a shape mismatch (e.g. feeding a
`Vec 10` where a `Vec 784` is expected, or composing incompatible layers) is a
compile error, not a runtime crash.

## Dataset

The MNIST binaries are **not** committed. Download the four IDX files and place
them in `data/`:

```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

(If you have the `.gz` versions, gunzip them first.)

## Build, test, run

```sh
stack build          # compile the library + executable
stack test           # QuickCheck property suite (incl. finite-difference gradient checks)
stack exec haskell-mnist-train   # train on MNIST and print per-epoch test accuracy
```

Hyperparameters (learning rate, epochs, dataset sizes, RNG seed) are top-level
constants at the top of `app/Main.hs`.

## Results

With the default settings (lr = 0.05, 10 000 examples/epoch, 5 epochs) the
network reaches **~90% test accuracy**:

```
Initial test accuracy: 9.90%
epoch 1: mean loss 0.5347, test accuracy 86.60%
epoch 3: mean loss 0.2685, test accuracy 87.10%
epoch 5: mean loss 0.2263, test accuracy 89.75%   (peak 91.35% at epoch 4)
```

See [`results/training-log.md`](results/training-log.md) for the full logs,
including a comparison with a too-high learning rate.
