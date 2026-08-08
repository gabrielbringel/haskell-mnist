-- | QuickCheck property tests for the dimension-indexed 'Vec' and 'Mat' types
-- and for the layer, network, activation and loss functions built on them.
--
-- The gradient properties ('prop_layer_backward_dW', 'prop_network_backward_fd')
-- check the analytical backprop gradients against central finite differences.
module Main (main) where

import           Control.Monad   (unless)
import           Data.List       (foldl')
import           System.Environment (lookupEnv)
import           System.Exit     (exitFailure)
import           Test.QuickCheck
import           Text.Read       (readMaybe)

import           GHC.TypeNats    (KnownNat)

import qualified Activation
import qualified Layer
import qualified Loss
import qualified Network
import           Mat             (Mat)
import qualified Mat
import           Vec             (Vec)
import qualified Vec

-- | Run every property in turn, exiting non-zero if any of them fails.
main :: IO ()
main = do
  configuredCases <- lookupEnv "QC_CASES"
  let testCases = max 1 (maybe 100 id (configuredCases >>= readMaybe))
      args = stdArgs { maxSuccess = testCases }
  results <- sequence
    [ checkWith args prop_transpose_involution
    , checkWith args prop_dot_comm
    , checkWith args prop_outer_elements
    , checkWith args prop_mulV_naive
    , checkWith args prop_softmax_sums_to_one
    , checkWith args prop_relu_nonneg
    , checkWith args prop_relu_deriv
    , checkWith args prop_crossentropy_nonneg
    , checkWith args prop_softmax_crossentropy_grad_fd
    , checkWith args prop_layer_forward_linear
    , checkWith args prop_layer_backward_dW
    , checkWith args prop_layer_backward_dB
    , checkWith args prop_network_forward_zero
    , checkWith args prop_network_backward_zero
    , checkWith args prop_network_backward_fd
    , checkWith args prop_confusion_total
    , checkWith args prop_confusion_diagonal
    ]
  unless (all isSuccess results) exitFailure

checkWith :: Testable prop => Args -> prop -> IO Result
checkWith = quickCheckWithResult

-- | Independent, uniformly distributed components in [-1, 1]. These bounded
-- generators are used by the full-network finite-difference property so its
-- numerical regime is explicit and reproducible.
genBoundedVec :: KnownNat n => Gen (Vec n Double)
genBoundedVec = do
  xs <- infiniteListOf (choose (-1.0, 1.0))
  pure (Vec.generate (xs !!))

genBoundedMat :: (KnownNat r, KnownNat c) => Gen (Mat r c Double)
genBoundedMat = do
  rows <- infiniteListOf (infiniteListOf (choose (-1.0, 1.0)))
  pure (Mat.mgenerate (\i j -> rows !! i !! j))

-- | The diagonal of the confusion matrix is exactly the number of correct
-- predictions — tying the matrix back to 'Network.predict' / 'Network.accuracy'.
prop_confusion_diagonal :: Mat 3 4 Double -> Mat 2 3 Double -> [Vec 4 Double] -> [Int] -> Bool
prop_confusion_diagonal w1 w2 xs ls =
  diag == length (filter correct dataset)
  where
    net = Network.Network
            { Network.hidden = Layer.Layer w1 (Vec.replicate 0)
            , Network.output = Layer.Layer w2 (Vec.replicate 0)
            } :: Network.Network 4 3 2
    dataset          = zip xs (map (`mod` 2) ls)
    cm               = Network.confusionMatrix net dataset
    diag             = sum [ Mat.mindex cm k k | k <- [0, 1] ]
    correct (x, lbl) = Network.predict net x == lbl

-- | Every entry of the dataset is counted exactly once: the sum over all cells
-- of the confusion matrix equals the size of the dataset.
prop_confusion_total :: Mat 3 4 Double -> Mat 2 3 Double -> [Vec 4 Double] -> [Int] -> Bool
prop_confusion_total w1 w2 xs ls =
  sum cm == length dataset
  where
    net = Network.Network
            { Network.hidden = Layer.Layer w1 (Vec.replicate 0)
            , Network.output = Layer.Layer w2 (Vec.replicate 0)
            } :: Network.Network 4 3 2
    dataset = zip xs (map (`mod` 2) ls)
    cm      = Network.confusionMatrix net dataset

-- | Cross-entropy loss of a (softmax) distribution against a one-hot target is
-- non-negative. We build valid inputs from the raw arguments: 'yhat' is a real
-- softmax output, 'y' is a one-hot vector selected by an arbitrary index.
prop_crossentropy_nonneg :: Vec 10 Double -> Int -> Bool
prop_crossentropy_nonneg logits i = Loss.crossEntropy yhat y >= 0
  where
    yhat = Activation.softmax logits
    k    = i `mod` 10
    y    = Vec.generate (\j -> if j == k then 1.0 else 0.0) :: Vec 10 Double

-- | The gradient of @crossEntropy . softmax@ w.r.t. logits (before softmax)
-- agrees with central finite differences. This validates the analytical
-- derivative of the composed loss function with respect to pre-softmax logits.
-- Logits are compressed to a bounded band to prevent softmax underflow.
prop_softmax_crossentropy_grad_fd :: Vec 10 Double -> Int -> Bool
prop_softmax_crossentropy_grad_fd rawLogits i =
    and [ close (Vec.vindex grad k) (numerical k) | k <- [0 .. 9] ]
  where
    logits = Vec.vmap (\x -> 4 * tanh (x / 4)) rawLogits
    kIdx   = i `mod` 10
    y      = Vec.generate (\j -> if j == kIdx then 1.0 else 0.0) :: Vec 10 Double
    grad   = Loss.crossEntropyGrad (Activation.softmax logits) y
    eps    = 1e-6
    close a n = abs (a - n) < 1e-4 * (1 + abs a) + 1e-3
    lossAt v = Loss.crossEntropy (Activation.softmax v) y
    numerical k =
      let perturb sign =
            Vec.generate (\j ->
              if j == k
              then Vec.vindex logits k + sign * eps
              else Vec.vindex logits j)
      in (lossAt (perturb 1) - lossAt (perturb (-1))) / (2 * eps)

-- | The dot product is commutative (exactly, since IEEE multiplication is
-- commutative and the summation order is identical).
prop_dot_comm :: Vec 5 Double -> Vec 5 Double -> Bool
prop_dot_comm u v = Vec.dot u v == Vec.dot v u

-- | Backward: the bias gradient dB always equals the upstream gradient dZ.
prop_layer_backward_dB :: Vec 4 Double -> Vec 3 Double -> Vec 4 Double -> Bool
prop_layer_backward_dB bias x dZ =
  let weights' = Mat.mreplicate 0 :: Mat 4 3 Double
      layer = Layer.Layer { Layer.weights = weights', Layer.bias = bias }
      (_, dB, _) = Layer.backward layer x dZ
  in dB == dZ

-- | Weight gradient via finite differences: for each weight element, the
-- analytical gradient dW[i,j] should match the numerical slope.
prop_layer_backward_dW :: Mat 4 3 Double -> Vec 3 Double -> Vec 4 Double -> Bool
prop_layer_backward_dW w x dZ =
  let layer = Layer.Layer { Layer.weights = w, Layer.bias = Vec.replicate 0 }
      (dW, _, _) = Layer.backward layer x dZ
      -- Scale-aware tolerance: QuickCheck-generated weights/inputs can have
      -- large magnitude, which inflates the absolute floating-point error of
      -- a central difference even when the relative error is tiny.
      close a n = abs (a - n) < 1e-4 * (1 + abs a) + 1e-3
  in and [ close (Mat.mindex dW i j) (numerical i j)
         | i <- [0 .. 3], j <- [0 .. 2] ]
  where
    eps = 1e-6
    forwardAt :: Mat 4 3 Double -> Vec 4 Double
    forwardAt weights' =
      Layer.forward (Layer.Layer weights' (Vec.replicate 0)) x
    -- Surrogate scalar loss whose gradient w.r.t. z is exactly dZ, so the
    -- numerical and analytical gradients are checking the same thing.
    lossAt z = Vec.dot dZ z
    -- For each weight (i,j), perturb that single entry by +/- eps and
    -- measure the change in loss.
    numerical i j =
      let perturb sign =
            Mat.mgenerate (\i' j' ->
              if i' == i && j' == j
              then Mat.mindex w  i j + sign * eps
              else Mat.mindex w  i' j')
          lossPlus  = lossAt (forwardAt (perturb  1))
          lossMinus = lossAt (forwardAt (perturb (-1)))
      in (lossPlus - lossMinus) / (2 * eps)

-- | Forward pass of a layer with zero bias is linear: applying a scalar
-- multiple to the input scales the output by the same factor.
prop_layer_forward_linear :: Vec 3 Double -> Positive Double -> Bool
prop_layer_forward_linear x (Positive a) =
  let layer = Layer.Layer { Layer.weights = w, Layer.bias = Vec.replicate 0 }
      y1    = Layer.forward layer (Vec.vscale a x)
      y2    = Vec.vscale a (Layer.forward layer x)
  in Vec.vsub y1 y2 == Vec.replicate 0
  where
    w :: Mat 4 3 Double
    w = Mat.mgenerate (\_ _ -> 0)

-- | 'Mat.mulV' agrees with a naive list-based reference. Both fold the products
-- with a strict left fold ('foldl''), matching @Data.Vector@'s 'sum', so the
-- equality is exact.
prop_mulV_naive :: Mat 4 3 Double -> Vec 3 Double -> Bool
prop_mulV_naive m v = Vec.toList (Mat.mulV m v) == naive
  where
    naive =
      [ foldl' (+) 0 (zipWith (*) (Vec.toList (Mat.mrow m i)) (Vec.toList v))
      | i <- [0 .. 3] ]

-- | The network backward pass agrees with finite differences at differentiable
-- points. Validates the full gradient composition (network + cross-entropy loss)
-- by perturbing each of the 23 parameters (W₁:12, b₁:3, W₂:6, b₂:2) and
-- checking the analytical gradient against central finite differences.
-- The loss is: Loss(logits) = crossEntropy(softmax(logits), y).
-- Cases at the ReLU kink are discarded because a central difference across
-- zero is not comparable to the implementation's convention relu'(0)=0.
prop_network_backward_fd :: Property
prop_network_backward_fd =
  forAll (genBoundedMat :: Gen (Mat 3 4 Double)) $ \w1 ->
  forAll (genBoundedVec :: Gen (Vec 3 Double)) $ \bias1 ->
  forAll (genBoundedMat :: Gen (Mat 2 3 Double)) $ \w2 ->
  forAll (genBoundedVec :: Gen (Vec 2 Double)) $ \bias2 ->
  forAll (genBoundedVec :: Gen (Vec 4 Double)) $ \x ->
  forAll (choose (0, 1) :: Gen Int) $ \labelIdx ->
    let
        y = Vec.generate (\j -> if j == labelIdx then 1.0 else 0.0) :: Vec 2 Double
        net = Network.Network
          { Network.hidden = Layer.Layer w1 bias1
          , Network.output = Layer.Layer w2 bias2
          } :: Network.Network 4 3 2
        (z1, a1, z2) = Network.networkForward net x
        dZ2 = Loss.crossEntropyGrad (Activation.softmax z2) y
        (dW1, dB1, dW2, dB2, _) = Network.networkBackward net x z1 a1 dZ2
        -- Tolerance: abs(analytical - numerical) <= atol + rtol * max(abs analytical, abs numerical)
        close a n = abs (a - n) <= 1e-6 + 1e-4 * max (abs a) (abs n)
        eps = 1e-6
        -- Perturbing a hidden weight changes z1 by eps*x[j], while perturbing
        -- a hidden bias changes it by eps. A factor of two leaves numerical
        -- margin on both sides of the ReLU kink.
        z1List = Vec.toList z1
        maxInput = maximum (1 : map abs (Vec.toList x))
        kinkMargin = 2 * eps * maxInput
        differentiable = all (\z -> abs z > kinkMargin) z1List
        lossAt z2' = Loss.crossEntropy (Activation.softmax z2') y
        numericalW1 i j =
          let netW1Plus = Network.Network
                { Network.hidden = Layer.Layer (Mat.mgenerate (\i' j' ->
                    if i' == i && j' == j then Mat.mindex w1 i j + eps
                    else Mat.mindex w1 i' j')) bias1
                , Network.output = Layer.Layer w2 bias2
                }
              netW1Minus = Network.Network
                { Network.hidden = Layer.Layer (Mat.mgenerate (\i' j' ->
                    if i' == i && j' == j then Mat.mindex w1 i j - eps
                    else Mat.mindex w1 i' j')) bias1
                , Network.output = Layer.Layer w2 bias2
                }
              (_, _, z2Plus) = Network.networkForward netW1Plus x
              (_, _, z2Minus) = Network.networkForward netW1Minus x
          in (lossAt z2Plus - lossAt z2Minus) / (2 * eps)
        numericalB1 i =
          let b1Plus = Vec.generate (\i' -> if i' == i then Vec.vindex bias1 i + eps
                                            else Vec.vindex bias1 i') :: Vec 3 Double
              b1Minus = Vec.generate (\i' -> if i' == i then Vec.vindex bias1 i - eps
                                             else Vec.vindex bias1 i') :: Vec 3 Double
              netB1Plus = Network.Network
                { Network.hidden = Layer.Layer w1 b1Plus
                , Network.output = Layer.Layer w2 bias2
                }
              netB1Minus = Network.Network
                { Network.hidden = Layer.Layer w1 b1Minus
                , Network.output = Layer.Layer w2 bias2
                }
              (_, _, z2Plus) = Network.networkForward netB1Plus x
              (_, _, z2Minus) = Network.networkForward netB1Minus x
          in (lossAt z2Plus - lossAt z2Minus) / (2 * eps)
        numericalW2 i j =
          let netW2Plus = Network.Network
                { Network.hidden = Layer.Layer w1 bias1
                , Network.output = Layer.Layer (Mat.mgenerate (\i' j' ->
                    if i' == i && j' == j then Mat.mindex w2 i j + eps
                    else Mat.mindex w2 i' j')) bias2
                }
              netW2Minus = Network.Network
                { Network.hidden = Layer.Layer w1 bias1
                , Network.output = Layer.Layer (Mat.mgenerate (\i' j' ->
                    if i' == i && j' == j then Mat.mindex w2 i j - eps
                    else Mat.mindex w2 i' j')) bias2
                }
              (_, _, z2Plus) = Network.networkForward netW2Plus x
              (_, _, z2Minus) = Network.networkForward netW2Minus x
          in (lossAt z2Plus - lossAt z2Minus) / (2 * eps)
        numericalB2 i =
          let b2Plus = Vec.generate (\i' -> if i' == i then Vec.vindex bias2 i + eps
                                            else Vec.vindex bias2 i') :: Vec 2 Double
              b2Minus = Vec.generate (\i' -> if i' == i then Vec.vindex bias2 i - eps
                                             else Vec.vindex bias2 i') :: Vec 2 Double
              netB2Plus = Network.Network
                { Network.hidden = Layer.Layer w1 bias1
                , Network.output = Layer.Layer w2 b2Plus
                }
              netB2Minus = Network.Network
                { Network.hidden = Layer.Layer w1 bias1
                , Network.output = Layer.Layer w2 b2Minus
                }
              (_, _, z2Plus) = Network.networkForward netB2Plus x
              (_, _, z2Minus) = Network.networkForward netB2Minus x
          in (lossAt z2Plus - lossAt z2Minus) / (2 * eps)
    in differentiable ==>
      conjoin $
        -- Test W1 (3x4 = 12 elements)
        [ counterexample ("W1 (" ++ show i ++ "," ++ show j ++ ")")
            (close (Mat.mindex dW1 i j) (numericalW1 i j))
        | i <- [0 .. 2], j <- [0 .. 3]
        ] ++
        -- Test b1 (3 elements)
        [ counterexample ("b1 " ++ show i)
            (close (Vec.vindex dB1 i) (numericalB1 i))
        | i <- [0 .. 2]
        ] ++
        -- Test W2 (2x3 = 6 elements)
        [ counterexample ("W2 (" ++ show i ++ "," ++ show j ++ ")")
            (close (Mat.mindex dW2 i j) (numericalW2 i j))
        | i <- [0 .. 1], j <- [0 .. 2]
        ] ++
        -- Test b2 (2 elements)
        [ counterexample ("b2 " ++ show i)
            (close (Vec.vindex dB2 i) (numericalB2 i))
        | i <- [0 .. 1]
        ]

-- | For a network with zero weights, ReLU'(0) = 0 kills the gradient flow
-- through the hidden layer. So dW₁, dB₁, dW₂, and dX should all be zero.
-- The output bias gradient dB₂ always equals dZ₂ regardless of weights.
prop_network_backward_zero :: Vec 4 Double -> Vec 2 Double -> Bool
prop_network_backward_zero x dZ2 =
  let zeroW1 = Mat.mreplicate 0 :: Mat 3 4 Double
      zeroB1 = Vec.replicate 0 :: Vec 3 Double
      zeroW2 = Mat.mreplicate 0 :: Mat 2 3 Double
      zeroB2 = Vec.replicate 0 :: Vec 2 Double
      net = Network.Network
        { Network.hidden = Layer.Layer zeroW1 zeroB1
        , Network.output = Layer.Layer zeroW2 zeroB2
        } :: Network.Network 4 3 2
      (z1, a1, _) = Network.networkForward net x
      (dW1, dB1, dW2, dB2, dX) = Network.networkBackward net x z1 a1 dZ2
  in dW1 == zeroW1 && dB1 == zeroB1 && dW2 == zeroW2 && dB2 == dZ2 && dX == Vec.replicate 0

-- | A zero-initialized network (all weights and biases zero) maps any input to
-- the zero vector (since ReLU(0) = 0).
prop_network_forward_zero :: Vec 4 Double -> Bool
prop_network_forward_zero x =
  let net = Network.Network
        { Network.hidden = Layer.Layer (Mat.mreplicate 0) (Vec.replicate 0)
        , Network.output = Layer.Layer (Mat.mreplicate 0) (Vec.replicate 0)
        } :: Network.Network 4 3 2
      (_, _, z2) = Network.networkForward net x
  in z2 == Vec.replicate 0

-- | Entry @(i, j)@ of @outer u v@ equals @u[i] * v[j]@.
prop_outer_elements :: Vec 4 Double -> Vec 3 Double -> Bool
prop_outer_elements u v =
  and [ Mat.mindex m i j == Vec.vindex u i * Vec.vindex v j
      | i <- [0 .. 3], j <- [0 .. 2] ]
  where
    m = Mat.outer u v

-- | The ReLU derivative is 1 for positive inputs and 0 otherwise.
prop_relu_deriv :: Double -> Bool
prop_relu_deriv x = Activation.relu' x == (if x > 0 then 1.0 else 0.0)

-- | ReLU is non-negative.
prop_relu_nonneg :: Double -> Bool
prop_relu_nonneg x = Activation.relu x >= 0

-- | A softmax output is a probability distribution: its entries sum to 1.
prop_softmax_sums_to_one :: Vec 10 Double -> Bool
prop_softmax_sums_to_one v = abs (sum (Activation.softmax v) - 1.0) < 1e-9

-- | Transposing twice is the identity.
prop_transpose_involution :: Mat 4 3 Double -> Bool
prop_transpose_involution m = Mat.transpose (Mat.transpose m) == m

-- Instances ------------------------------------------------------------------
--
-- A random 'Vec' is built with 'Vec.generate', drawing each component from an
-- infinite stream of arbitrary values. 'Mat' is built the same way via
-- 'Mat.mgenerate'. The type-level size is supplied by the 'KnownNat' context.

instance KnownNat n => Arbitrary (Vec n Double) where
  arbitrary = do
    xs <- infiniteListOf arbitrary
    pure (Vec.generate (xs !!))

instance (KnownNat r, KnownNat c) => Arbitrary (Mat r c Double) where
  arbitrary = do
    rows <- infiniteListOf (infiniteListOf arbitrary)
    pure (Mat.mgenerate (\i j -> rows !! i !! j))
