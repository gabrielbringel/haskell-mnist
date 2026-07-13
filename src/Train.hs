-- | Online (per-example) gradient descent training for a two-layer
-- 'Network'. Each step runs a forward pass, scores the prediction with
-- cross-entropy against a one-hot target, backpropagates, and nudges every
-- weight and bias against its gradient by the learning rate.
module Train
  ( Example
  , trainEpoch
  , trainEpochs
  , trainStep
  ) where

import           Data.List    (mapAccumL)
import           GHC.TypeNats (KnownNat)

import           Activation   (softmax)
import           Layer        (Layer (..))
import           Loss         (crossEntropy, crossEntropyGrad)
import qualified Mat
import           Network      (Network (..))
import qualified Network
import           Vec          (Vec)
import qualified Vec

-- | A labelled training example: an input and its one-hot target
-- distribution.
type Example i o = (Vec i Double, Vec o Double)

-- | One epoch: fold 'trainStep' over the whole dataset in order, threading
-- the updated network through each example (online SGD, not batched).
-- Returns the network after the epoch and the mean loss over the epoch's
-- examples (each measured against its own pre-update parameters).
trainEpoch
  :: (KnownNat i, KnownNat h, KnownNat o)
  => Double
  -> Network i h o
  -> [Example i o]
  -> (Network i h o, Double)
trainEpoch lr net0 examples = (net', mean losses)
  where
    (net', losses) = mapAccumL (trainStep lr) net0 examples
    mean xs = sum xs / fromIntegral (length xs)

-- | Repeat 'trainEpoch' for a fixed number of epochs over the same dataset,
-- returning the final network and the mean loss recorded after each epoch
-- (in order, so the last element is the final epoch's loss).
trainEpochs
  :: (KnownNat i, KnownNat h, KnownNat o)
  => Double            -- ^ learning rate
  -> Int               -- ^ number of epochs
  -> Network i h o
  -> [Example i o]
  -> (Network i h o, [Double])
trainEpochs lr epochs net0 examples = go epochs net0
  where
    go n net
      | n <= 0    = (net, [])
      | otherwise =
          let (net1, loss)   = trainEpoch lr net examples
              (netF, losses) = go (n - 1) net1
          in (netF, loss : losses)

-- | One step of stochastic gradient descent on a single example: forward
-- pass, cross-entropy loss against the one-hot target, backward pass, and a
-- @param' = param - lr * grad@ update for every weight and bias in both
-- layers. Returns the updated network and the loss incurred by the
-- /pre-update/ parameters (the ones that actually produced the prediction).
trainStep
  :: (KnownNat i, KnownNat h, KnownNat o)
  => Double            -- ^ learning rate
  -> Network i h o
  -> Example i o
  -> (Network i h o, Double)
trainStep lr net (x, y) = (net', loss)
  where
    (z1, a1, z2) = Network.networkForward net x
    yhat         = softmax z2
    loss         = crossEntropy yhat y
    dZ2          = crossEntropyGrad yhat y
    (dW1, dB1, dW2, dB2, _dX) = Network.networkBackward net x z1 a1 dZ2

    -- Explicit signatures matter here: GADTs/TypeFamilies imply
    -- MonoLocalBinds, so without them these helpers would be monomorphised
    -- to their first use site (the hidden layer's shape) and fail to type
    -- check at the output layer's different shape.
    descendW :: Mat.Mat r c Double -> Mat.Mat r c Double -> Mat.Mat r c Double
    descendW w dw = Mat.msub w (Mat.mscale lr dw)

    descendB :: Vec r Double -> Vec r Double -> Vec r Double
    descendB b db = Vec.vsub b (Vec.vscale lr db)

    net' = Network
      { hidden = Layer (descendW (weights (hidden net)) dW1) (descendB (bias (hidden net)) dB1)
      , output = Layer (descendW (weights (output net)) dW2) (descendB (bias (output net)) dB2)
      }
