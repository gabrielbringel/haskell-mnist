-- | Composition of linear layers with ReLU activations into a feedforward
-- network. Dimensions are threaded through at the type level so incompatible
-- layers are rejected at compile time.
--
-- A 'Network i h o' takes an input of size @i@, passes it through a hidden
-- layer of size @h@ with ReLU, then through an output layer of size @o@. The
-- final output is raw logits (softmax is applied separately).
--
-- 'networkForward' chains the forward passes: @x → relu(W₁·x + b₁) → W₂·a₁ + b₂@.
-- 'networkBackward' runs reverse-mode backprop through both layers, returning
-- gradients for all four parameter tensors and the input gradient.
module Network
  ( Network(..)
  , networkForward
  , networkBackward
  ) where

import           GHC.TypeNats (KnownNat, Nat)

import           Activation   (relu, relu')
import           Layer        (Layer (..))
import qualified Layer
import           Mat          (Mat)
import           Vec          (Vec)
import qualified Vec

-- | A two-layer feedforward network with typed dimensions. The first layer
-- maps @i@ inputs to @h@ hidden units; the second maps @h@ hidden units to
-- @o@ outputs. ReLU is the hidden-layer nonlinearity; the output layer returns
-- raw logits.
data Network (i :: Nat) (h :: Nat) (o :: Nat) = Network
  { hidden :: Layer i h
  , output :: Layer h o
  }

-- | Full forward pass: input → hidden layer (pre-activation) → ReLU → output
-- layer (pre-activation logits). Returns the raw logits, which should be
-- passed through 'Activation.softmax' before computing the loss.
--
-- Also returns the hidden-layer pre-activation @z₁@ and post-activation @a₁@,
-- which are needed by 'networkBackward'.
networkForward
  :: (KnownNat i, KnownNat h, KnownNat o)
  => Network i h o
  -> Vec i Double               -- ^ input @x@
  -> (Vec h Double, Vec h Double, Vec o Double)
     -- ^ @(z₁, a₁, z₂)@ — hidden pre-activation, hidden post-activation, output logits
networkForward net x = (z1, a1, z2)
  where
    z1 = Layer.forward (hidden net) x
    a1 = Vec.vmap relu z1
    z2 = Layer.forward (output net) a1

-- | Full backward pass (reverse-mode differentiation). Given the upstream
-- gradient @dZ₂@ (gradient of the loss w.r.t. the output logits), computes
-- gradients for all four parameter tensors and the gradient w.r.t. the input.
--
-- The ReLU derivative is applied elementwise when the gradient flows through
-- the hidden layer's activation.
networkBackward
  :: (KnownNat i, KnownNat h, KnownNat o)
  => Network i h o
  -> Vec i Double               -- ^ input @x@
  -> Vec h Double               -- ^ hidden pre-activation @z₁@
  -> Vec h Double               -- ^ hidden post-activation @a₁@
  -> Vec o Double               -- ^ upstream gradient @dZ₂@
  -> (Mat h i Double, Vec h Double, Mat o h Double, Vec o Double, Vec i Double)
     -- ^ @(dW₁, dB₁, dW₂, dB₂, dX)@
networkBackward net x z1 a1 dZ2 = (dW1, dB1, dW2, dB2, dX)
  where
    -- Output layer backward
    (dW2, dB2, dA1) = Layer.backward (output net) a1 dZ2
    -- ReLU derivative: chain rule multiplies dA₁ elementwise by relu'(z₁)
    dZ1 = Vec.vzipWith (\da z -> da * relu' z) dA1 z1
    -- Hidden layer backward
    (dW1, dB1, dX)  = Layer.backward (hidden net) x dZ1