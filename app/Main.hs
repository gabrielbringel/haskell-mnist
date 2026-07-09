-- | Entry point: load MNIST, train a two-layer network with online SGD, and
-- report test-set accuracy after each epoch.
module Main (main) where

import           Control.Monad (foldM)
import           System.Random (mkStdGen)
import           Text.Printf   (printf)

import           Init          (randomNetwork)
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
epochs = 5

trainSize :: Int
trainSize = 10000     -- examples per epoch (subset of the full 60k set)

testSize :: Int
testSize = 2000       -- examples used to estimate accuracy

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

  _ <- foldM
    (\net e -> do
        let (net', loss) = Train.trainEpoch learningRate net trainExamples
            acc          = Network.accuracy net' testSet
        printf "epoch %d: mean loss %.4f, test accuracy %.2f%%\n"
          e loss (100 * acc)
        pure net')
    net0
    [1 .. epochs]

  pure ()
