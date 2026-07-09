-- | Length-indexed vectors. The type-level 'Nat' @n@ records the number of
-- elements, so dimension mismatches are caught by the type checker rather than
-- at runtime. The invariant @length internalVector == n@ is preserved by every
-- smart constructor below.
module Vec
  ( Vec(..)
    -- * Construction
  , replicate
  , generate
  , fromList
  , toList
  , size
    -- * Combinators
  , vmap
  , vzipWith
    -- * Numeric operations
  , dot
  , vadd
  , vsub
  , vscale
    -- * Access
  , vindex
  ) where

import           Prelude       hiding (replicate)

import           Data.Foldable (toList)
import           Data.Proxy    (Proxy (..))
import           Data.Vector   (Vector)
import qualified Data.Vector   as V
import           GHC.TypeNats  (KnownNat, Nat, natVal)

-- | A vector of exactly @n@ elements of type @a@.
newtype Vec (n :: Nat) a = Vec (Vector a)

-- Construction ---------------------------------------------------------------

-- | A vector with every element equal to the given value.
replicate :: forall n a. KnownNat n => a -> Vec n a
replicate x = Vec (V.replicate (fromIntegral (natVal (Proxy :: Proxy n))) x)

-- | Build a vector from an index-to-value function (@0 <= i < n@).
generate :: forall n a. KnownNat n => (Int -> a) -> Vec n a
generate f = Vec (V.generate (fromIntegral (natVal (Proxy :: Proxy n))) f)

-- | Build a vector from a list, returning 'Nothing' unless the list has
-- exactly @n@ elements.
fromList :: forall n a. KnownNat n => [a] -> Maybe (Vec n a)
fromList xs
  | length xs == fromIntegral (natVal (Proxy :: Proxy n)) = Just (Vec (V.fromList xs))
  | otherwise                                             = Nothing

-- | The length @n@, recovered from the type.
size :: forall n a. KnownNat n => Vec n a -> Int
size _ = fromIntegral (natVal (Proxy :: Proxy n))

-- Combinators ----------------------------------------------------------------

-- | Map a function over every element. Equivalent to 'fmap'.
vmap :: (a -> b) -> Vec n a -> Vec n b
vmap f (Vec v) = Vec (V.map f v)

-- | Combine two equal-length vectors pointwise.
vzipWith :: (a -> b -> c) -> Vec n a -> Vec n b -> Vec n c
vzipWith f (Vec a) (Vec b) = Vec (V.zipWith f a b)

-- Numeric operations ---------------------------------------------------------

-- | Dot product.
dot :: Num a => Vec n a -> Vec n a -> a
dot (Vec a) (Vec b) = V.sum (V.zipWith (*) a b)

-- | Pointwise addition.
vadd :: Num a => Vec n a -> Vec n a -> Vec n a
vadd = vzipWith (+)

-- | Pointwise subtraction.
vsub :: Num a => Vec n a -> Vec n a -> Vec n a
vsub = vzipWith (-)

-- | Scalar multiplication.
vscale :: Num a => a -> Vec n a -> Vec n a
vscale k = vmap (k *)

-- Access ---------------------------------------------------------------------

-- | Index into the vector. No bounds are checked beyond the underlying
-- 'Vector'; callers are expected to pass @0 <= i < n@.
vindex :: Vec n a -> Int -> a
vindex (Vec v) i = v V.! i

-- Instances ------------------------------------------------------------------

instance Functor (Vec n) where
  fmap = vmap

instance Foldable (Vec n) where
  foldMap f (Vec v) = foldMap f v
  foldr f z (Vec v) = V.foldr f z v
  length (Vec v)    = V.length v

instance Traversable (Vec n) where
  traverse f (Vec v) = Vec <$> traverse f v

instance Show a => Show (Vec n a) where
  show (Vec v) = "Vec " ++ show (V.toList v)

instance Eq a => Eq (Vec n a) where
  Vec a == Vec b = a == b
