-- | QuickCheck property tests for the dimension-indexed 'Vec' and 'Mat' types.
module Main (main) where

import           Control.Monad   (unless)
import           Data.List       (foldl')
import           System.Exit     (exitFailure)
import           Test.QuickCheck

import           GHC.TypeNats    (KnownNat)

import qualified Activation
import qualified Layer
import qualified Loss
import qualified Network
import           Mat             (Mat)
import qualified Mat
import           Vec             (Vec)
import qualified Vec

-- Arbitrary instances --------------------------------------------------------
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

-- Properties -----------------------------------------------------------------

-- | Transposing twice is the identity.
prop_transpose_involution :: Mat 4 3 Double -> Bool
prop_transpose_involution m = Mat.transpose (Mat.transpose m) == m

-- | The dot product is commutative (exactly, since IEEE multiplication is
-- commutative and the summation order is identical).
prop_dot_comm :: Vec 5 Double -> Vec 5 Double -> Bool
prop_dot_comm u v = Vec.dot u v == Vec.dot v u

-- | Entry @(i, j)@ of @outer u v@ equals @u[i] * v[j]@.
prop_outer_elements :: Vec 4 Double -> Vec 3 Double -> Bool
prop_outer_elements u v =
  and [ Mat.mindex m i j == Vec.vindex u i * Vec.vindex v j
      | i <- [0 .. 3], j <- [0 .. 2] ]
  where
    m = Mat.outer u v

-- | 'Mat.mulV' agrees with a naive list-based reference. Both fold the products
-- with a strict left fold ('foldl''), matching @Data.Vector@'s 'sum', so the
-- equality is exact.
prop_mulV_naive :: Mat 4 3 Double -> Vec 3 Double -> Bool
prop_mulV_naive m v = Vec.toList (Mat.mulV m v) == naive
  where
    naive =
      [ foldl' (+) 0 (zipWith (*) (Vec.toList (Mat.mrow m i)) (Vec.toList v))
      | i <- [0 .. 3] ]

-- Activation / Loss properties -----------------------------------------------

-- | A softmax output is a probability distribution: its entries sum to 1.
prop_softmax_sums_to_one :: Vec 10 Double -> Bool
prop_softmax_sums_to_one v = abs (sum (Activation.softmax v) - 1.0) < 1e-9

-- | ReLU is non-negative.
prop_relu_nonneg :: Double -> Bool
prop_relu_nonneg x = Activation.relu x >= 0

-- | The ReLU derivative is 1 for positive inputs and 0 otherwise.
prop_relu_deriv :: Double -> Bool
prop_relu_deriv x = Activation.relu' x == (if x > 0 then 1.0 else 0.0)

-- | Cross-entropy loss of a (softmax) distribution against a one-hot target is
-- non-negative. We build valid inputs from the raw arguments: 'yhat' is a real
-- softmax output, 'y' is a one-hot vector selected by an arbitrary index.
prop_crossentropy_nonneg :: Vec 10 Double -> Int -> Bool
prop_crossentropy_nonneg logits i = Loss.crossEntropy yhat y >= 0
  where
    yhat = Activation.softmax logits
    k    = i `mod` 10
    y    = Vec.generate (\j -> if j == k then 1.0 else 0.0) :: Vec 10 Double

-- Layer properties ------------------------------------------------------------

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

-- | Backward: the bias gradient dB always equals the upstream gradient dZ.
prop_layer_backward_dB :: Vec 4 Double -> Vec 3 Double -> Vec 4 Double -> Bool
prop_layer_backward_dB bias x dZ =
  let weights' = Mat.mreplicate 0 :: Mat 4 3 Double
      layer = Layer.Layer { Layer.weights = weights', Layer.bias = bias }
      (_, dB, _) = Layer.backward layer x dZ
  in dB == dZ

-- Network properties ----------------------------------------------------------

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

-- | The network backward pass agrees with finite differences: perturb each
-- weight in the hidden layer by +/- eps, compute the change in a scalar loss
-- (dZ₂ · z₂), and compare to the analytical gradient component.
prop_network_backward_fd :: Mat 3 4 Double -> Mat 2 3 Double -> Vec 4 Double -> Vec 2 Double -> Bool
prop_network_backward_fd w1 w2 x dZ2 = and [ close (Mat.mindex dW1 i j) (numerical i j)
                                              | i <- [0 .. 2], j <- [0 .. 3] ]
  where
    bias1 = Vec.replicate 0 :: Vec 3 Double
    bias2 = Vec.replicate 0 :: Vec 2 Double
    net = Network.Network
      { Network.hidden = Layer.Layer w1 bias1
      , Network.output = Layer.Layer w2 bias2
      } :: Network.Network 4 3 2
    (z1, a1, _) = Network.networkForward net x
    (dW1, _, _, _, _) = Network.networkBackward net x z1 a1 dZ2
    close a n = abs (a - n) < 1e-4 * (1 + abs a) + 1e-3
    eps = 1e-6
    netAt w1' = Network.Network
      { Network.hidden = Layer.Layer w1' bias1
      , Network.output = Layer.Layer w2 bias2
      }
    lossAt w1' =
      let (_, _, z2') = Network.networkForward (netAt w1') x
      in Vec.dot dZ2 z2'
    numerical i j =
      let perturb sign =
            Mat.mgenerate (\i' j' ->
              if i' == i && j' == j
              then Mat.mindex w1 i j + sign * eps
              else Mat.mindex w1 i' j')
      in (lossAt (perturb 1) - lossAt (perturb (-1))) / (2 * eps)

-- Runner ---------------------------------------------------------------------

main :: IO ()
main = do
  results <- sequence
    [ quickCheckResult prop_transpose_involution
    , quickCheckResult prop_dot_comm
    , quickCheckResult prop_outer_elements
    , quickCheckResult prop_mulV_naive
    , quickCheckResult prop_softmax_sums_to_one
    , quickCheckResult prop_relu_nonneg
    , quickCheckResult prop_relu_deriv
    , quickCheckResult prop_crossentropy_nonneg
    , quickCheckResult prop_layer_forward_linear
    , quickCheckResult prop_layer_backward_dW
    , quickCheckResult prop_layer_backward_dB
    , quickCheckResult prop_network_forward_zero
    , quickCheckResult prop_network_backward_zero
    , quickCheckResult prop_network_backward_fd
    ]
  unless (all isSuccess results) exitFailure
