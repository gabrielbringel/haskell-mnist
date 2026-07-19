# Reproducibility Facts

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

## Numerical scheme of tests

**Source:** `test/Spec.hs` (multiple lines detailed below)

**Description:**

### Finite-difference step size
- ε = 1e−6
  - Line 97 (prop_crossentropy_grad_fd): `eps = 1e-6`
  - Line 134 (prop_layer_backward_dW): `eps = 1e-6`
  - Line 198 (prop_network_backward_fd): `eps = 1e-6`

### Numerical tolerance for comparisons
- Line 98 (prop_crossentropy_grad_fd): `close a n = abs (a - n) < 1e-4 * (1 + abs a) + 1e-3`
- Line 130 (prop_layer_backward_dW): Same tolerance formula
- Line 197 (prop_network_backward_fd): Same tolerance formula

This is a **scale-aware tolerance**: the relative error (1e−4 times the scale of the value) plus an absolute floor (1e−3) to handle small magnitudes.

### Reduced dimensions in tests
- Hidden layer: 3 neurons
- Input dimension: 4
- Output dimension: 2
- Layer weight matrix: `Mat 4 3 Double` (4 inputs → 3 hidden)
- Output layer matrix: `Mat 2 3 Double` (3 hidden → 2 outputs)
- Network type: `Network.Network 4 3 2`
  - Line 57: `prop_confusion_diagonal`
  - Line 72: `prop_confusion_total`
  - Line 194: `prop_network_backward_fd`

### Logits bounding for finite-difference test
- Line 93 (prop_crossentropy_grad_fd): `logits = Vec.vmap (\x -> 4 * tanh (x / 4)) rawLogits`
- **Effect:** compresses raw logits to the band [−4, 4] (since tanh maps ℝ → (−1, 1), multiplying by 4 gives approximately (−4, 4))
- **Purpose:** prevents softmax underflow/overflow during numerical gradient computation while testing the gradient of `crossEntropy . softmax`

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
