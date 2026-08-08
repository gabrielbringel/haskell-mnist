# Reproducibility Facts

## Compiler and build environment

**Source:** `stack.yaml` and `.cabal` file

**Description:**
- **GHC version:** 9.4.8 (pinned via Stack resolver `lts-21.25`)
- **Build tool:** Stack (https://docs.haskellstack.org/)
- **Compilation flags:** `-Wall` package-wide; the training executable adds `-O2`
- **Haskell language extensions enabled project-wide:**
  `DataKinds`, `GADTs`, `KindSignatures`, `TypeFamilies`, `TypeOperators`,
  `ScopedTypeVariables`, `FlexibleContexts`, `FlexibleInstances`, `RankNTypes`

**Reproducibility implication:** Same versions and flags ensure identical type
checking and code generation across machines.

---

## Network architecture (production)

**Source:** `app/Main.hs:22–24`

**Description:**
- Input dimension: 784 (28×28 MNIST images, each normalized to [0, 1])
- Hidden layer: 64 neurons with ReLU activation
- Output dimension: 10 (digit classes 0–9) with softmax + cross-entropy loss
- Total parameters: 784×64 + 64 + 64×10 + 10 = 50,890

---

## Test suite: QuickCheck properties and reduced dimensions

**Source:** `test/Spec.hs:27–45` (property list) and `test/Spec.hs:183–295`
(main gradient test)

**Description:**
The test suite runs **17 QuickCheck properties** (100 test cases per property):

1. **Linear algebra properties (4):**
   - `prop_transpose_involution`: M^T^T = M
   - `prop_dot_comm`: dot product commutativity
   - `prop_outer_elements`: outer product element-wise correctness
   - `prop_mulV_naive`: matrix-vector product vs. naive reference

2. **Activation and loss properties (5):**
   - `prop_softmax_sums_to_one`: softmax output sums to 1
   - `prop_relu_nonneg`: ReLU output is non-negative
   - `prop_relu_deriv`: ReLU derivative correctness
   - `prop_crossentropy_nonneg`: cross-entropy ≥ 0
   - `prop_softmax_crossentropy_grad_fd`: gradient of cross-entropy ∘ softmax
     via finite differences

3. **Single-layer gradient properties (3):**
   - `prop_layer_forward_linear`: forward pass linearity
   - `prop_layer_backward_dW`: weight gradient vs. finite differences
   - `prop_layer_backward_dB`: bias gradient equals upstream gradient

4. **Network-level properties (3):**
   - `prop_network_forward_zero`: zero-initialized network maps to zero
   - `prop_network_backward_zero`: zero weights kill gradient flow
   - **`prop_network_backward_fd` (primary):** Validates the complete gradient
     composition (network + cross-entropy + softmax) by checking all 23 parameter
     gradients per-element against central finite differences.

5. **Evaluation properties (2):**
   - `prop_confusion_diagonal`: confusion matrix diagonal = accuracy
   - `prop_confusion_total`: all examples counted exactly once

### Reduced test dimensions

For speed, the test network uses dimensions **4 → 3 → 2** (not the production
784 → 64 → 10):
- Input: 4 dimensions
- Hidden: 3 neurons
- Output: 2 classes

This results in **23 total parameters**:
- W₁: 4×3 = 12 weights
- b₁: 3 biases
- W₂: 2×3 = 6 weights
- b₂: 2 biases

All 23 parameters are tested in `prop_network_backward_fd` (lines 276–295).

### Numerical scheme for `prop_network_backward_fd`

- **Finite-difference step:** ε = 1e-6
- **Tolerance:** close(a, n) ≡ |a - n| ≤ 1e-6 + 1e-4 · max(|a|, |n|)
  (scale-aware: relative 1e-4, absolute floor 1e-6)
- **Generator for the network property:** independent uniform components in
  [-1, 1] for W₁, b₁, W₂, b₂, and x; label uniform in {0, 1}
- **Discard filter (differentiability):** keep a case only when
  |z₁ᵢ| > 2ε·max(1, ‖x‖∞) for every hidden unit
- **Observed discards:** 0 in the recorded 100-case run and 0 in the recorded
  1,000-case stress run
- **Stress command:** `QC_CASES=1000 cabal test all`

---

## Normalization [0,1]

**Source:** `src/MNIST.hs:90`

**Description:**
Pixel values from raw byte data (0–255) are normalized to [0, 1] using the expression:
```haskell
map (\b -> fromIntegral b / 255.0) bytes
```
Each byte `b` is converted to a Double and divided by 255.0, placing all pixels in the interval [0, 1].

---

## No reshuffling between epochs

**Source:** `src/Train.hs:32–41` (trainEpoch) and `src/Train.hs:46–60` (trainEpochs)

**Description:**
- `trainEpoch` (lines 32–41) uses `mapAccumL (trainStep lr) net0 examples`, which processes the examples list in order without shuffling.
- `trainEpochs` (lines 46–60) calls `trainEpoch lr net examples` repeatedly with the same dataset `examples` in each iteration, passing `n - 1` epochs recursively without reordering.
- No random shuffling is performed between epochs; the training set is processed in its original order for every epoch.

**Limitation:**
This is a limitation of the training procedure: a fixed presentation order across all epochs may lead to oscillations in training accuracy and suboptimal convergence rates compared to shuffled SGD. The README (line 112) notes that "online SGD with a fixed learning rate need not improve that metric monotonically," and oscillations are indeed observed (e.g., epoch 3 to epoch 25 in the reported results).

---

## Auditable artifact

**Source:** Repository root (Git remote configuration)

**Description:**
The codebase is available at the public GitHub repository:
```
https://github.com/gabrielbringel/haskell-mnist
```

All code, documentation, and experiment results are version-controlled and reproducible from this repository.

---

## MNIST dataset instructions

**Source:** `README.md:67–81` and `src/MNIST.hs:1–103`

**Description:**

### Required files
Four IDX format files must be downloaded and placed in the `data/` directory:
```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

### SHA-256 of the uncompressed files used

```text
ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db  train-images-idx3-ubyte
65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5  train-labels-idx1-ubyte
0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7  t10k-images-idx3-ubyte
ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2  t10k-labels-idx1-ubyte
```

### Format handling
- If files are downloaded as `.gz` (gzip-compressed), they must be decompressed with `gunzip` before use.
- The loader validates the IDX magic numbers:
  - Line 79 (`src/MNIST.hs`): Images must have magic number 2051
  - Line 97 (`src/MNIST.hs`): Labels must have magic number 2049
- Line 39–40 (`src/MNIST.hs`): The loader verifies that images are exactly 28 × 28 pixels and raises an error otherwise.

### Loader validation (src/MNIST.hs)
- `parseImageHeader` (lines 76–80): Checks magic number 2051 for images
- `parseLabelHeader` (lines 94–98): Checks magic number 2049 for labels
- `loadMNIST` (lines 32–54): Verifies image dimensions (28 × 28) and that image and label counts agree
- `parseImages` (lines 84–92): Normalizes each raw byte to [0, 1] by dividing by 255.0
- `parseLabels` (lines 100–102): Reads label bytes as plain integers (0–9)
