-- | Cross-entropy loss for a softmax output layer.
module Loss
  ( crossEntropy
  , crossEntropyGrad
  ) where

import           Vec (Vec)
import qualified Vec

-- | Cross-entropy loss @-sum_i y_i * log(yhat_i)@, where @yhat@ is the
-- predicted distribution (post-'Activation.softmax') and @y@ is the one-hot
-- target. Returns a scalar.
crossEntropy
  :: Vec n Double  -- ^ predicted distribution @yhat@
  -> Vec n Double  -- ^ one-hot target @y@
  -> Double
crossEntropy yhat y = negate (sum (Vec.vzipWith term y yhat))
  where
    -- Follow the convention 0 * log 0 = 0, so a zero target never turns an
    -- underflowed (0) probability into a NaN via 0 * (-Infinity).
    term t p
      | t == 0    = 0
      | otherwise = t * log p

-- | Gradient of the cross-entropy loss with respect to the softmax /input/
-- (the pre-softmax logits). Because the softmax and cross-entropy derivatives
-- telescope, this collapses to the clean @yhat - y@ form.
crossEntropyGrad
  :: Vec n Double  -- ^ predicted distribution @yhat@
  -> Vec n Double  -- ^ one-hot target @y@
  -> Vec n Double
crossEntropyGrad yhat y = Vec.vsub yhat y
