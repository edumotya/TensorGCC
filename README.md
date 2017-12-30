# TensorGCC
This is a TensorFlow implementation of the generalized cross-correlation (GCC), which can be used at any point of the graph. It implements the Carter GCC and the  PHAT transform.

Testing scripts still belongs to a wider software package, but these are some results for pseudo-chirp signals embedded in real noise:

The unbiased standard GCC for chirp signals with spectrogram 

![alt text](https://github.com/edumotya/TensorGCC/blob/master/images/specs.png)

result in the following cross-correlation estimator

![alt text](https://github.com/edumotya/TensorGCC/blob/master/images/gcc.png)

