# haskell-mnist

A feedforward neural network that classifies MNIST handwritten digits,
implemented from scratch in Haskell with **type-indexed dimensions**.
Every matrix and vector carries its shape at the type level (via `DataKinds`),
so a dimension mismatch — feeding a `Vec 10` where a `Vec 784`
is expected, or composing two incompatible layers — is a *compile-time* error
rather than a runtime crash. Backpropagation is written by hand as a typed
reverse pass; there is no automatic-differentiation library and no BLAS.

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

Starting accuracy is near the 1/10 chance classification rate. The network
reaches its ~94–95% plateau by epoch 3, after which the accuracy oscillates from
epoch to epoch; online SGD with a fixed learning rate need not improve that
metric monotonically. See
[`results/training-log.md`](results/training-log.md) for the full per-epoch log
and the confusion-matrix error analysis.

### Reproducing the figures

The training curves and confusion-matrix plots are generated under `results/`
from the CSVs by [`scripts/plot_results.py`](scripts/plot_results.py), then
copied to `paper/` for inclusion by LaTeX:

```sh
pip install -r requirements.txt        # matplotlib, numpy, pandas
python scripts/plot_results.py         # writes results/fig-training.pdf and fig-confusion.pdf
cp results/fig-training.pdf paper/
cp results/fig-confusion.pdf paper/
python scripts/validate_results.py     # checks data, dataset hashes, and PDF labels
```

## Testing

`stack test` runs 17 QuickCheck properties (100 cases each) covering the linear
algebra, activations, loss, and — most importantly — the backward passes.
The core property, `prop_network_backward_fd`, validates the complete **gradient
composition (network + cross-entropy + softmax)** by checking all 23 parameter
gradients (W₁:12, b₁:3, W₂:6, b₂:2) per-element against **central finite
differences** (ε = 1e-6, tolerance atol=1e-6, rtol=1e-4). The test network uses
reduced dimensions (4 inputs → 3 hidden → 2 outputs) for speed; the production
network is 784 → 64 → 10. Its weights, biases, and inputs are generated
independently and uniformly in `[-1,1]`. A case is retained only when
`|z₁| > 2ε·max(1, ‖x‖∞)`, ensuring that no finite-difference perturbation crosses
the ReLU kink. The recorded 100-case run and 1,000-case stress run had no
discarded cases.

Use `QC_CASES` to change the number of successful cases per property:

```sh
stack test                         # 100 cases per property
QC_CASES=1000 cabal test all       # stress validation
```

## Notes and limitations

- Training uses **online SGD** (one gradient step per image), not mini-batches.
- Matrices and vectors are boxed `Data.Vector`s with no BLAS; the emphasis here
  is on type-level correctness, not competitive throughput.
- `paper/` contains the LaTeX write-up (`main.tex`, `references.bib`), which
  embeds the figures produced by `scripts/plot_results.py`.

## License

MIT — see [`LICENSE`](LICENSE).
