{-# LANGUAGE DataKinds #-}
-- | scratch/DimensionError.hs
--
-- NAO faz parte da build do projeto (nao esta em exposed-modules do .cabal).
-- Existe so para gerar, de proposito, o erro de compilacao usado como
-- "Listing 1" no artigo: prova de que uma dimensao incompativel e rejeitada
-- em tempo de compilacao, nao em tempo de execucao.
--
-- O mismatch: `weirdMat` e uma Mat 10 5, entao `mulV` espera um Vec 5.
-- Passamos um Vec 3 de proposito.
module Main (main) where

import Mat (Mat)
import qualified Mat
import Vec (Vec)
import qualified Vec

weirdMat :: Mat 10 5 Double
weirdMat = Mat.mgenerate (\_ _ -> 0)

badVec :: Vec 3 Double
badVec = Vec.replicate 0

main :: IO ()
main = print (Mat.mulV weirdMat badVec)
