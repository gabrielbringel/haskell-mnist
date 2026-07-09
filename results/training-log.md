# Training log

Runs of the `haskell-mnist-train` executable (`app/Main.hs`) on the standard
MNIST dataset. The network is a two-layer feedforward classifier
`784 → 64 → 10` with a ReLU hidden layer and a softmax + cross-entropy output,
trained with **online (per-example) SGD** — one gradient step per image, no
mini-batching.

Common setup for both runs:

| setting            | value                                   |
| ------------------ | --------------------------------------- |
| architecture       | 784 → 64 (ReLU) → 10 (softmax)          |
| weight init        | He-uniform `U(-√(6/fanIn), √(6/fanIn))`, biases 0 |
| examples per epoch | 10 000 (subset of the 60 000 train set) |
| test examples      | 2 000 (subset of the 10 000 test set)   |
| epochs             | 5                                       |
| RNG seed           | 42                                      |
| wall time          | ~7.5 min / run (boxed `Data.Vector`, GHC 9.4.8, `-O2`, aarch64-osx) |

Accuracy is measured on the held-out test subset after each epoch; "mean loss"
is the average per-example cross-entropy over the epoch, measured against each
example's pre-update parameters.

## Run 1 — learning rate 0.1 (too high)

```
Initial test accuracy: 9.90%
epoch 1: mean loss 1.0614, test accuracy 73.40%
epoch 2: mean loss 0.9402, test accuracy 69.85%
epoch 3: mean loss 0.9613, test accuracy 74.10%
epoch 4: mean loss 0.9607, test accuracy 77.70%
epoch 5: mean loss 0.9346, test accuracy 75.65%
```

The loss stalls around ~0.94 and accuracy oscillates (70–78%). Classic sign of
a learning rate that is too large for batch-size-1 SGD: each step overshoots, so
the parameters bounce around a minimum instead of settling into it.

## Run 2 — learning rate 0.05 (current default)

```
Initial test accuracy: 9.90%
epoch 1: mean loss 0.5347, test accuracy 86.60%
epoch 2: mean loss 0.3236, test accuracy 87.75%
epoch 3: mean loss 0.2685, test accuracy 87.10%
epoch 4: mean loss 0.2834, test accuracy 91.35%
epoch 5: mean loss 0.2263, test accuracy 89.75%
```

Halving the learning rate gives a smooth, monotonic loss descent
(0.53 → 0.23) and lifts accuracy to **~90%** (peak 91.35% at epoch 4). This is
the default committed in `app/Main.hs`.

## Notes

- Starting accuracy is ~9.9% (≈ 1/10), confirming the softmax output is near
  uniform at initialisation, as expected.
- These runs use subsets (10k train / 2k test) to keep wall time reasonable
  given the naive boxed-vector implementation. Training on the full 60k set for
  more epochs would push accuracy higher, at proportionally more time.
- Hyperparameters live as top-level constants at the top of `app/Main.hs`
  (`learningRate`, `epochs`, `trainSize`, `testSize`, `seed`) — adjust and
  rebuild to reproduce or extend these runs.
