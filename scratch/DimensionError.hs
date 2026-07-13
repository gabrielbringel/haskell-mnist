{-# LANGUAGE DataKinds #-}
-- | scratch/DimensionError.hs
--
-- NOT part of the project build (it is not listed in the .cabal
-- exposed-modules). It exists only to trigger, on purpose, the compile error
-- used as "Listing 1" in the paper: proof that an incompatible dimension is
-- rejected at compile time rather than at run time.
--
-- The mismatch: 'weirdMat' is a @Mat 10 5@, so 'Mat.mulV' expects a @Vec 5@.
-- We deliberately pass it a @Vec 3@.
module Main (main) where

import Mat (Mat)
import qualified Mat
import Vec (Vec)
import qualified Vec

-- | A 3-element vector — deliberately the wrong width for 'weirdMat'.
badVec :: Vec 3 Double
badVec = Vec.replicate 0

-- | Applies 'weirdMat' to 'badVec', which fails to type check.
main :: IO ()
main = print (Mat.mulV weirdMat badVec)

-- | A 10-by-5 zero matrix, whose 'Mat.mulV' expects a 5-element vector.
weirdMat :: Mat 10 5 Double
weirdMat = Mat.mgenerate (\_ _ -> 0)
