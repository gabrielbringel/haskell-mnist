-- | Random initialization of a 'Network'.
--
-- Weights are drawn from a He-style uniform distribution — @U(-l, l)@ with
-- @l = sqrt(6 / fanIn)@, where @fanIn@ is the number of inputs to the layer —
-- which keeps the pre-activation variance roughly constant across the ReLU
-- hidden layer. Biases are initialised to zero, the usual convention.
--
-- Randomness is threaded purely through a 'StdGen'; there is no 'IO' here, so
-- initialisation is deterministic given a seed and easy to test.
module Init
  ( randomNetwork
  ) where

import           Data.Proxy    (Proxy (..))
import           GHC.TypeNats  (KnownNat, natVal)
import           System.Random (StdGen, randomRs, split)

import           Layer         (Layer (..))
import           Mat           (Mat)
import qualified Mat
import           Network       (Network (..))
import qualified Vec

-- | Build a random two-layer network from a seed generator. The hidden and
-- output weight matrices get independent generators (via 'split'); both bias
-- vectors are zero.
randomNetwork
  :: forall i h o. (KnownNat i, KnownNat h, KnownNat o)
  => StdGen
  -> Network i h o
randomNetwork g = Network
  { hidden = Layer w1 (Vec.replicate 0)
  , output = Layer w2 (Vec.replicate 0)
  }
  where
    fanIn1   = fromIntegral (natVal (Proxy :: Proxy i)) :: Double
    fanIn2   = fromIntegral (natVal (Proxy :: Proxy h)) :: Double
    (g1, g2) = split g
    w1 = randomMat (sqrt (6 / fanIn1)) g1
    w2 = randomMat (sqrt (6 / fanIn2)) g2

-- | An @r@-by-@c@ matrix whose entries are drawn uniformly from @[-l, l]@.
randomMat
  :: forall r c. (KnownNat r, KnownNat c)
  => Double     -- ^ limit @l@
  -> StdGen
  -> Mat r c Double
randomMat limit g =
  case Mat.mfromList (take n (randomRs (-limit, limit) g)) of
    Just m  -> m
    Nothing -> error "randomMat: size mismatch (unreachable)"
  where
    n = fromIntegral (natVal (Proxy :: Proxy r))
      * fromIntegral (natVal (Proxy :: Proxy c)) :: Int
