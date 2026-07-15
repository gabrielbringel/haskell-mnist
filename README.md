# haskell-mnist

A feedforward neural network that classifies MNIST handwritten digits,
implemented from scratch in Haskell with **dependently-typed dimensions**.
Every matrix and vector carries its shape at the type level (via `DataKinds`
and `GADTs`), so a dimension mismatch — feeding a `Vec 10` where a `Vec 784`
is expected, or composing two incompatible layers — is a *compile-time* error
rather than a runtime crash. Backpropagation is written by hand as a typed
reverse pass; there is no autodiff library and no BLAS.

## Key idea: shapes in the types

```haskell
newtype Vec (n :: Nat) a      -- a vector of exactly n elements
newtype Mat (r :: Nat) (c :: Nat) a  -- an r-by-c matrix

data Layer   (c :: Nat) (r :: Nat)   -- linear map: c inputs -> r outputs
data Network (i :: Nat) (h :: Nat) (o :: Nat)  -- i -> h (ReLU) -> o
```

Because the sizes are `Nat`s recovered with `KnownNat`, the type checker
guarantees that weights, activations, and gradients all line up. For example
`Mat.mulV :: Mat r c a -> Vec c a -> Vec r a` can only be applied to a vector
whose length matches the matrix's column count.

## Architecture

A two-layer classifier: **784 → 64 (ReLU) → 10 (softmax)**, trained with
online (per-example) stochastic gradient descent against a cross-entropy loss.
The softmax + cross-entropy gradient collapses to the clean `yhat - y` form,
and the ReLU derivative gates gradient flow through the hidden layer.

Backprop identities used, for a linear layer `Z = W x + b`:

| gradient | formula      | code                          |
| -------- | ------------ | ----------------------------- |
| `dL/dW`  | `dZ · xᵀ`    | `Mat.outer dZ x`              |
| `dL/db`  | `dZ`         | `dZ`                          |
| `dL/dx`  | `Wᵀ · dZ`    | `Mat.mulV (Mat.transpose w) dZ` |

## Project layout

| module          | responsibility                                                     |
| --------------- | ------------------------------------------------------------------ |
| `Vec`           | length-indexed vectors and numeric operations                      |
| `Mat`           | dimension-indexed matrices; `mulV`, `transpose`, `outer` for backprop |
| `Activation`    | ReLU / ReLU′ and numerically-stable `softmax`                      |
| `Loss`          | cross-entropy and its gradient                                     |
| `Layer`         | a linear layer with typed `forward` / `backward`                   |
| `Network`       | two-layer composition; `networkForward`, `networkBackward`, `predict`, `accuracy`, `confusionMatrix` |
| `Init`          | He-uniform random weight initialisation (pure, seeded)             |
| `Train`         | SGD `trainStep` / `trainEpoch` / `trainEpochs`                     |
| `MNIST`         | IDX file-format loader for images and labels                       |
| `app/Main.hs`   | executable entry point: load data → train → report accuracy → write confusion matrix |
| `test/Spec.hs`  | QuickCheck property suite                                           |
| `scripts/plot_results.py` | matplotlib figures (training curves, confusion matrix) for the paper |

## Requirements

- [Stack](https://docs.haskellstack.org/) (the resolver `lts-21.25` pins
  **GHC 9.4.8**; Stack installs it for you).

The package enables `DataKinds`, `GADTs`, `KindSignatures`, `TypeFamilies`,
`TypeOperators`, `ScopedTypeVariables`, `FlexibleContexts`, `FlexibleInstances`,
and `RankNTypes` package-wide, and builds with `-Wall`.

## Dataset

The MNIST binaries are **not** committed (they are gitignored). Download the
four IDX files and place them in `data/`:

```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

If you downloaded the `.gz` versions, gunzip them first. The loader expects
28×28 images and validates the IDX magic numbers on read.

## Build, test, run

```sh
stack build                       # compile the library + executable
stack test                        # run the QuickCheck property suite
stack exec haskell-mnist-train    # train on MNIST, print per-epoch accuracy
```

The executable prints the mean loss and test accuracy after each epoch, then
prints the final confusion matrix and writes it to
[`results/confusion-matrix.csv`](results/confusion-matrix.csv).

Hyperparameters are top-level constants at the top of `app/Main.hs`
(`learningRate`, `epochs`, `trainSize`, `testSize`, `seed`) — edit and rebuild
to reproduce or extend the runs.

## Results

With the defaults (lr = 0.05, 60 000 examples/epoch, 25 epochs, seed 42) the
network reaches **95.49% test accuracy** on the full 10 000-image test set:

```
Initial test accuracy: 10.08%
epoch  1: mean loss 0.3871, test accuracy 92.17%
epoch  3: mean loss 0.2661, test accuracy 94.07%
epoch 25: mean loss 0.1629, test accuracy 95.49%
```

Starting accuracy is ≈ 1/10, confirming the softmax output is near-uniform at
initialisation. The network reaches its ~94–95% plateau by epoch 3, after which
the accuracy oscillates from step to step as expected for online SGD with a
fixed learning rate. See [`results/training-log.md`](results/training-log.md)
for the full per-epoch log and the confusion-matrix error analysis.

### Reproducing the figures

The training curves and confusion-matrix plots in `paper/` are generated from
the CSVs in `results/` by [`scripts/plot_results.py`](scripts/plot_results.py):

```sh
pip install -r requirements.txt        # matplotlib, numpy, pandas
python scripts/plot_results.py         # writes results/fig-training.pdf and fig-confusion.pdf
```

## Testing

`stack test` runs 16 QuickCheck properties (100 cases each) covering the linear
algebra, activations, loss, and — most importantly — the backward passes:
`Layer.backward` and `Network.networkBackward` are checked against **numerical
finite-difference gradients**, so the hand-written backprop is verified, not
just assumed.

## Notes and limitations

- Training uses **online SGD** (one gradient step per image), not mini-batches.
- Matrices and vectors are boxed `Data.Vector`s with no BLAS, so a full run of
  the defaults takes a few minutes; the emphasis here is on type-level
  correctness, not throughput.
- `paper/` contains the LaTeX write-up (`main.tex`, `references.bib`), which
  embeds the figures produced by `scripts/plot_results.py`.

## License

MIT — see [`LICENSE`](LICENSE).
