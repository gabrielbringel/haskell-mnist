-- | MNIST dataset loading.
-- Parses the standard IDX file format (big-endian 32-bit integers followed by
-- raw bytes) and returns images as 'Vec 784 Double' (pixels in [0,1]) and
-- labels as plain 'Int's (0–9).
--
-- The two file pairs are:
--   * @data/train-images-idx3-ubyte@ + @data/train-labels-idx1-ubyte@
--   * @data/t10k-images-idx3-ubyte@  + @data/t10k-labels-idx1-ubyte@
module MNIST
  ( MNISTImage
  , MNISTLabel
  , loadTest
  , loadTraining
  ) where

import           Data.Bits      (shiftL, (.|.))
import           Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import           Data.List      (foldl')

import           Vec            (Vec)
import qualified Vec

-- | One flattened 28×28 image (@784@ pixels), each in @[0,1]@.
type MNISTImage = Vec 784 Double

-- | A digit label in @0..9@.
type MNISTLabel = Int

-- | Load an image/label file pair, checking the IDX magic numbers, the 28×28
-- image shape, and that the two files agree on the example count.
loadMNIST :: FilePath -> FilePath -> IO ([MNISTImage], [MNISTLabel])
loadMNIST imagesPath labelsPath = do
  imgBytes <- BS.readFile imagesPath
  let imgHeader = parseHeader imgBytes
      (nImgs, rows, cols) = parseImageHeader imgHeader
      imgData = BS.drop 16 imgBytes     -- skip 4 × 32-bit header fields
  -- Verify dimensions match expected MNIST format.
  if rows /= 28 || cols /= 28
    then error ("MNIST: expected 28×28 images, got " ++ show rows ++ "×" ++ show cols)
    else pure ()

  lblBytes <- BS.readFile labelsPath
  let lblHeader = parseHeader lblBytes
      nLbls    = parseLabelHeader lblHeader
      lblData  = BS.drop 8 lblBytes     -- skip 2 × 32-bit header fields

  if nImgs /= nLbls
    then error ("MNIST: image count " ++ show nImgs ++ " ≠ label count " ++ show nLbls)
    else pure ()

  let images = parseImages nImgs imgData
      labels = parseLabels nLbls lblData
  pure (images, labels)

-- | Load the 10 000-image test set. Images and labels are paired by index.
loadTest :: IO ([MNISTImage], [MNISTLabel])
loadTest = loadMNIST "data/t10k-images-idx3-ubyte" "data/t10k-labels-idx1-ubyte"

-- | Load the 60 000-image training set. Images and labels are paired by index.
loadTraining :: IO ([MNISTImage], [MNISTLabel])
loadTraining = loadMNIST "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"

-- | Slices the first 16 bytes of a ByteString into four big-endian 32-bit
-- integers. The caller is responsible for interpreting the fields according to
-- the IDX format: [magic, count, rows?, cols?] for images; [magic, count] for
-- labels.
parseHeader :: ByteString -> (Int, Int, Int, Int)
parseHeader bs =
  let ubytes = BS.take 16 bs
      asInt offset =
        foldl' (\acc b -> (acc `shiftL` 8) .|. fromIntegral b) (0 :: Int)
               (map (\i -> BS.index ubytes (offset + i)) [0 .. 3])
  in (asInt 0, asInt 4, asInt 8, asInt 12)

-- | Verify image magic number @2051@ and extract (count, rows, cols).
parseImageHeader :: (Int, Int, Int, Int) -> (Int, Int, Int)
parseImageHeader (magic, count, rows, cols)
  | magic /= 2051 = error ("MNIST: bad image magic number " ++ show magic ++ ", expected 2051")
  | otherwise     = (count, rows, cols)

-- | Parse @count@ images, each @28×28 = 784@ raw bytes (grayscale, 0–255),
-- normalised to @[0, 1]@.
parseImages :: Int -> ByteString -> [MNISTImage]
parseImages count bs = map parseImage (take count slices)
  where
    slices = [ BS.take 784 (BS.drop (i * 784) bs) | i <- [0 ..] ]
    parseImage chunk =
      let bytes = BS.unpack chunk
      in case Vec.fromList (map (\b -> fromIntegral b / 255.0) bytes) of
           Just v  -> v
           Nothing -> error "MNIST: internal error — Vec.fromList failed for 784 elements"

-- | Verify label magic number @2049@ and extract count.
parseLabelHeader :: (Int, Int, Int, Int) -> Int
parseLabelHeader (magic, count, _, _)
  | magic /= 2049 = error ("MNIST: bad label magic number " ++ show magic ++ ", expected 2049")
  | otherwise     = count

-- | Parse @count@ labels, each a single byte (digit 0–9).
parseLabels :: Int -> ByteString -> [MNISTLabel]
parseLabels count bs = map fromIntegral (BS.unpack (BS.take count bs))
