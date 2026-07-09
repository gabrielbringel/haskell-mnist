-- | Activation functions and their derivatives.
--
-- 'relu' / 'relu'' are the hidden-layer nonlinearity and its derivative (the
-- latter is needed when propagating gradients backwards). 'softmax' is the
-- output-layer activation; it is computed in a numerically stable way by
-- subtracting the maximum logit before exponentiating.
module Activation
  ( relu
  , relu'
  , softmax
  ) where

import           Vec (Vec)
import qualified Vec

-- | Rectified linear unit: @max 0 x@.
relu :: Double -> Double
relu x = max 0 x

-- | Derivative of 'relu'. Undefined at @0@ mathematically; we follow the usual
-- convention and treat it as @0@ there.
relu' :: Double -> Double
relu' x = if x > 0 then 1.0 else 0.0

-- | Softmax over a vector of logits, producing a probability distribution that
-- sums to @1@. The maximum logit is subtracted before exponentiating so that
-- @exp@ never overflows.
softmax :: Vec n Double -> Vec n Double
softmax v = Vec.vmap (/ total) exps
  where
    m     = maximum v
    exps  = Vec.vmap (\x -> exp (x - m)) v
    total = sum exps
