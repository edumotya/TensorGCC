"""Generalized cross-correlation"""

import numpy as np
import tensorflow as tf
import ops

def gcc(waveforms, max_delay=None, weighting=None, scale=None):
    """
    Estimates the cross-correlation sequence of a random process of length M from the generalized 
    cross-correlation (fft/ifft), as specified in (Knapp & Carter 1976). By default, there is neither
    normalisation nor scaling and the output sequence has a length of min(2*M+1, max_delay)
    Parameters:
        waveforms: tensor arrays of time signals with shape [batch(optional), channel, waveform]
        max_delay: maximum lag in number of samples
        weighting: frequency domain filtering: None, 'PHAT'
        scale: cross-correlation scaling: None, 'biased', 'unbiased'
    Outputs:
        r01: cross-correlation function in samples
    """
    with tf.name_scope(None, "gcc"):
        with tf.control_dependencies([tf.assert_rank_at_least(waveforms, 2)]):
            ndims = waveforms.shape.ndims
            if ndims == 2:
                # Slice channels
                x0 = waveforms[0,:]
                x1 = waveforms[1,:]
            elif ndims == 3:
                # Slice channels
                x0 = waveforms[:,0,:]
                x1 = waveforms[:,1,:]
            else:
                return None
            # Get lengths
            m = 2*waveforms.get_shape()[-1].value - 1
            if max_delay is None:
                ncorr = m
            else:
                ncorr =  min(m, int(max_delay))
            nfft = 2**ops.next_power_of_two(m)
            # Remove DC
            x0 = x0 - tf.reduce_mean(x0, 1, keep_dims=True)
            x1 = x1 - tf.reduce_mean(x1, 1, keep_dims=True)
            return _gcc(x0, x1, nfft, ncorr, weighting, scale)

def _gcc(x0, x1, nfft, ncorr, weighting, scale):
    # Spectrums
    X0 = tf.conj(tf.spectral.rfft(x0, [nfft])) # Reference signal spectrum
    X1 = tf.spectral.rfft(x1, [nfft])          # Delayed replica spectrum
    # Cross-spectrum
    R01 = X1*X0
    # Filter cross-spectrum
    R01 = _filter(R01, weighting)
    # Get back to time
    r01 = tf.spectral.irfft(R01, fft_length=[nfft], name='ifft_R01')
    # Check numerics, since ifft gives NaNs for very large or non power of 2 rfft or when using very large ffts
    r01 = tf.check_numerics(r01,
    'ifft gives NaNs. Hint: ensure nfft=2^n or reduce ifft length')
    # Keep only the lags we want and move negative lags before positive lags
    r01 = _shift(r01, nfft, ncorr)
    return _scale(x0, r01, scale)

def _shift(x, N, ncorr):
    assert(N>ncorr)
    with tf.name_scope(None, 'ifft_shift'):
        left_x =   x[:,:ncorr]
        right_x =  x[:, N-ncorr:N-1]
        y = tf.concat([right_x, left_x], axis = 1)
    return y

def _filter(R01, weighting):
    with tf.name_scope(None, 'weighting'):
        if weighting == 'PHAT':
            filtered_R01 = tf.exp( tf.complex(0.0, tf.angle(R01)), name='exp_angle' )
        else:
            filtered_R01 = R01
    return filtered_R01

def _scale(x, gcc, scale):
    m = x.get_shape()[1].value
    with tf.name_scope(None, 'scale'):
        if scale == 'biased':
            return gcc/m
        elif scale == 'unbiased':
            L = int((gcc.get_shape()[1].value-1)/2)
            den = (m - np.abs(np.arange(-L, L+1)))
            den[den<=0] = 1
            return gcc/den
        else:
            return gcc
