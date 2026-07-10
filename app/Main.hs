-- | Entry point: load MNIST, train a two-layer network with online SGD, and
-- report test-set accuracy after each epoch.
module Main (main) where

import           Control.Monad (foldM)
import           Data.List     (intercalate)
import           System.Random (mkStdGen)
import           Text.Printf   (printf)

import           Init          (randomNetwork)
import qualified Mat
import qualified MNIST
import           Network       (Network)
import qualified Network
import qualified Train
import           Vec           (Vec)
import qualified Vec

-- Network shape (dimensions live at the type level).
type Input  = 784
type Hidden = 64
type Output = 10

-- Hyperparameters.
learningRate :: Double
learningRate = 0.05

epochs :: Int
epochs = 25

trainSize :: Int
trainSize = 60000     -- examples per epoch (subset of the full 60k set)

testSize :: Int
testSize = 10000       -- examples used to estimate accuracy

seed :: Int
seed = 42

-- | One-hot encode a class index into an 'Output'-dimensional target vector.
oneHot :: Int -> Vec Output Double
oneHot k = Vec.generate (\j -> if j == k then 1 else 0)

main :: IO ()
main = do
  putStrLn "Loading MNIST..."
  (trainImgs, trainLbls) <- MNIST.loadTraining
  (testImgs,  testLbls)  <- MNIST.loadTest

  let trainExamples = take trainSize (zip trainImgs (map oneHot trainLbls))
      testSet       = take testSize  (zip testImgs testLbls)
      net0          = randomNetwork (mkStdGen seed) :: Network Input Hidden Output

  printf "Training: %d examples/epoch, %d epochs, lr=%.3f\n"
    trainSize epochs learningRate
  printf "Initial test accuracy: %.2f%%\n" (100 * Network.accuracy net0 testSet)

  netF <- foldM
    (\net e -> do
        let (net', loss) = Train.trainEpoch learningRate net trainExamples
            acc          = Network.accuracy net' testSet
        printf "epoch %d: mean loss %.4f, test accuracy %.2f%%\n"
          e loss (100 * acc)
        pure net')
    net0
    [1 .. epochs]

  let cm = Network.confusionMatrix netF testSet
  putStrLn "\nMatriz de confusão (linha = rótulo real, coluna = previsto):"
  putStr (renderConfusion cm)
  writeFile "results/confusion-matrix.csv" (toCSV cm)
  putStrLn "Escrito: results/confusion-matrix.csv"

-- | CSV 10×10 puro: linha = rótulo real (0..9), coluna = previsto (0..9),
-- sem cabeçalho nem coluna de índice — direto para matplotlib/numpy.
toCSV :: Mat.Mat Output Output Int -> String
toCSV cm = unlines
  [ intercalate "," [ show (Mat.mindex cm t p) | p <- [0 .. n - 1] ]
  | t <- [0 .. n - 1] ]
  where n = Mat.mrows cm

-- | Render legível para o stdout, colunas alinhadas.
renderConfusion :: Mat.Mat Output Output Int -> String
renderConfusion cm = unlines
  [ concatMap (pad . show . Mat.mindex cm t) [0 .. n - 1]
  | t <- [0 .. n - 1] ]
  where
    n       = Mat.mrows cm
    pad s   = replicate (6 - length s) ' ' ++ s
