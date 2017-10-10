Conditional Variational Autoencoder (CVAE)
==========================================
Implemented for MNIST images by modifying pytorch VAE example. Condition by providing labels and passing them along with input images.
We condition in 2 places:

1) As the input to the encoder
2) As the input to the decoder

Creating labels was a bit of a problem, solved by using the scatter_() method.
See here:

https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
