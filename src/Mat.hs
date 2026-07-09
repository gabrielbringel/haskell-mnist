-- | Dimension-indexed matrices stored row-major in a flat 'Vector'. The
-- type-level 'Nat's @r@ and @c@ record the number of rows and columns; the
-- invariant is @length internalVector == r * c@, with element @(i, j)@ living
-- at flat index @i * c + j@.
--
-- The three operations 'mulV', 'transpose' and 'outer' are the reason this
-- module exists: they are exactly what backprop needs.
--
--   * @'mulV' w x@      — forward pre-activation
--   * @'mulV' ('transpose' w) delta@ — propagate the gradient backwards
--   * @'outer' delta x@ — gradient of the weights (same shape as @w@)
module Mat
  ( Mat(..)
    -- * Construction
  , mgenerate
  , mreplicate
  , mfromList
    -- * Access
  , mrows
  , mcols
  , mindex
  , mrow
    -- * Backprop operations
  , mulV
  , transpose
  , outer
    -- * Elementwise
  , madd
  , msub
  , mscale
  ) where

import           Data.Proxy   (Proxy (..))
import           Data.Vector  (Vector)
import qualified Data.Vector  as V
import           GHC.TypeNats (KnownNat, Nat, natVal)

import           Vec          (Vec (..), dot, generate, vindex)

-- | An @r@-by-@c@ matrix of elements of type @a@, stored row-major.
newtype Mat (r :: Nat) (c :: Nat) a = Mat (Vector a)

-- Construction ---------------------------------------------------------------

-- | Build a matrix from an @(i, j)@-to-value function.
mgenerate :: forall r c a. (KnownNat r, KnownNat c) => (Int -> Int -> a) -> Mat r c a
mgenerate f = Mat (V.generate (r * c) (\k -> f (k `div` c) (k `mod` c)))
  where
    r = fromIntegral (natVal (Proxy :: Proxy r))
    c = fromIntegral (natVal (Proxy :: Proxy c))

-- | A matrix with every entry equal to the given value.
mreplicate :: forall r c a. (KnownNat r, KnownNat c) => a -> Mat r c a
mreplicate x = Mat (V.replicate (r * c) x)
  where
    r = fromIntegral (natVal (Proxy :: Proxy r))
    c = fromIntegral (natVal (Proxy :: Proxy c))

-- | Build a matrix from a row-major list, returning 'Nothing' unless the list
-- has exactly @r * c@ elements.
mfromList :: forall r c a. (KnownNat r, KnownNat c) => [a] -> Maybe (Mat r c a)
mfromList xs
  | length xs == r * c = Just (Mat (V.fromList xs))
  | otherwise          = Nothing
  where
    r = fromIntegral (natVal (Proxy :: Proxy r))
    c = fromIntegral (natVal (Proxy :: Proxy c))

-- Access ---------------------------------------------------------------------

-- | Number of rows, recovered from the type.
mrows :: forall r c a. KnownNat r => Mat r c a -> Int
mrows _ = fromIntegral (natVal (Proxy :: Proxy r))

-- | Number of columns, recovered from the type.
mcols :: forall r c a. KnownNat c => Mat r c a -> Int
mcols _ = fromIntegral (natVal (Proxy :: Proxy c))

-- | Index element @(i, j)@. Callers are expected to pass valid indices.
mindex :: forall r c a. (KnownNat r, KnownNat c) => Mat r c a -> Int -> Int -> a
mindex m@(Mat v) i j = v V.! (i * mcols m + j)

-- | Extract row @i@ as a 'Vec'.
mrow :: forall r c a. (KnownNat r, KnownNat c) => Mat r c a -> Int -> Vec c a
mrow m@(Mat v) i = Vec (V.slice (i * c) c v)
  where
    c = mcols m

-- Backprop operations --------------------------------------------------------

-- | Matrix-vector product: @r@-by-@c@ matrix times a @c@-vector gives an
-- @r@-vector. Row @i@ of the result is @'dot' (mrow m i) x@.
mulV :: forall r c a. (KnownNat r, KnownNat c, Num a) => Mat r c a -> Vec c a -> Vec r a
mulV m x = generate (\i -> dot (mrow m i) x)

-- | Transpose: entry @(i, j)@ of the result is entry @(j, i)@ of the input.
transpose :: forall r c a. (KnownNat r, KnownNat c) => Mat r c a -> Mat c r a
transpose m = mgenerate (\i j -> mindex m j i)

-- | Outer product: entry @(i, j)@ is @u[i] * v[j]@.
outer :: forall r c a. (KnownNat r, KnownNat c, Num a) => Vec r a -> Vec c a -> Mat r c a
outer u v = mgenerate (\i j -> vindex u i * vindex v j)

-- Elementwise ----------------------------------------------------------------

-- | Pointwise addition.
madd :: Num a => Mat r c a -> Mat r c a -> Mat r c a
madd (Mat a) (Mat b) = Mat (V.zipWith (+) a b)

-- | Pointwise subtraction.
msub :: Num a => Mat r c a -> Mat r c a -> Mat r c a
msub (Mat a) (Mat b) = Mat (V.zipWith (-) a b)

-- | Scalar multiplication.
mscale :: Num a => a -> Mat r c a -> Mat r c a
mscale k (Mat a) = Mat (V.map (k *) a)

-- Instances ------------------------------------------------------------------

instance Functor (Mat r c) where
  fmap f (Mat v) = Mat (V.map f v)

instance Foldable (Mat r c) where
  foldMap f (Mat v) = foldMap f v
  foldr f z (Mat v) = V.foldr f z v
  length (Mat v)    = V.length v

instance Show a => Show (Mat r c a) where
  show (Mat v) = "Mat " ++ show (V.toList v)

instance Eq a => Eq (Mat r c a) where
  Mat a == Mat b = a == b
