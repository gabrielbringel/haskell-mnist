-- | A linear (fully-connected) layer with typed dimensions @c@ (input size)
-- and @r@ (output size). The type system guarantees that weights, input, and
-- gradients all have compatible shapes, catching mismatches at compile time.
--
-- 'forward' computes @W * x + b@ (pre-activation logits).
-- 'backward' computes the weight gradient, bias gradient, and the gradient to
-- propagate to the previous layer, given the upstream gradient @dZ@.
--
-- Key backprop identities for a linear layer @Z = W x + b@:
--
--   @dL/dW = dZ * x^T@   →  @outer dZ x@  (same shape as @W@)
--   @dL/db = dZ@                      (same shape as @b@)
--   @dL/dx = W^T * dZ@   →  @mulV (transpose w) dZ@  (same shape as @x@)
module Layer
  ( Layer(..)
  , backward
  , forward
  ) where

import           GHC.TypeNats (KnownNat, Nat)

import           Mat          (Mat)
import qualified Mat
import           Vec          (Vec)
import qualified Vec

-- | A linear layer from an input of size @c@ to an output of size @r@.
-- The weights matrix is @r x c@ and the bias is an @r@-vector.
data Layer (c :: Nat) (r :: Nat) = Layer
  { weights :: Mat r c Double
  , bias    :: Vec r Double
  }

-- | Backward pass: given the upstream gradient @dZ@ (gradient of the loss
-- with respect to this layer's /pre-activation/ output), returns:
--
--   1. The weight gradient @dW@ (same shape as @weights@)
--   2. The bias gradient @dB@ (same shape as @bias@)
--   3. The gradient to propagate to the previous layer @dX@ (same shape as @x@)
backward
  :: (KnownNat c, KnownNat r)
  => Layer c r
  -> Vec c Double    -- ^ input @x@ that was used in the forward pass
  -> Vec r Double    -- ^ upstream gradient @dZ@
  -> (Mat r c Double, Vec r Double, Vec c Double)
backward layer x dZ = (dW, dB, dX)
  where
    w  = weights layer
    dW = Mat.outer dZ x           -- dL/dW = dZ * x^T  (outer product)
    dB = dZ                       -- dL/db = dZ
    dX = Mat.mulV (Mat.transpose w) dZ  -- dL/dx = W^T * dZ

-- | Forward pass: @Z = W x + b@ (pre-activation logits).
forward
  :: (KnownNat c, KnownNat r)
  => Layer c r
  -> Vec c Double    -- ^ input activations (from previous layer or raw data)
  -> Vec r Double    -- ^ pre-activation output
forward layer x = Vec.vadd (Mat.mulV (weights layer) x) (bias layer)
