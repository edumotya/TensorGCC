# TensorGCC
This is a TensorFlow implementation of the generalized cross-correlation (GCC), which can be used at any point of the graph. It implements the Carter GCC, as well as the  PHAT transform. The only limitation of this implementation is that the length of the input signals must be statically defined.

Example results for pseudo-chirp signals embedded in real noise:

For two chirp signals with spectrogram 

![alt text](https://github.com/edumotya/TensorGCC/blob/master/images/specs.png)

we get the unbiased GCC estimator

![alt text](https://github.com/edumotya/TensorGCC/blob/master/images/gcc.png)



