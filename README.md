# haskell-mnist
Feedforward neural network trained on MNIST, implemented in Haskell with dependent types. Matrix and vector dimensions are encoded at the type level using DataKinds and GADTs, turning dimensionality errors into compile-time failures. Backpropagation implemented manually as a typed reverse fold.
