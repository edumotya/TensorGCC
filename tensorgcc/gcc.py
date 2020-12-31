"""Generalized cross-correlation"""

import numpy as np
import tensorflow as tf
import math


def next_power_of_two(n):
    """Return res for integer n such that 2^res>= n."""
    mantissa, res = math.frexp(n)
    if mantissa == 0.5:
        res -= 1
    return res


def gcc(x0, x1, max_delay=None, weighting=None, scale=None):
    """
    Estimates the cross-correlation sequence of a random process of length M from the generalized
    cross-correlation (fft/ifft), as specified in (Knapp & Carter 1976). By default, there is neither
    normalisation nor scaling and the output sequence has a length of min(2*M+1, max_delay)
    Parameters:
        x0: waveform tensor array with shape [batch_size(optional), num_samples]
        x1: waveform tensor array with shape [batch_size(optional), num_samples]
        max_delay: maximum lag in number of samples
        weighting: frequency domain filtering: None, 'PHAT'
        scale: cross-correlation scaling: None, 'biased', 'unbiased'
    Outputs:
        r01: cross-correlation function in samples
    """
    if weighting not in [None, "PHAT"]:
        raise ValueError(f"weighting {weighting} not supported.")
    if scale not in [None, "biased", "unbiased"]:
        raise ValueError(f"scale {scale} not supported.")
    with tf.name_scope("gcc"):
        with tf.control_dependencies(
            [
                tf.debugging.assert_rank_in(x0, [1, 2]),
                tf.debugging.assert_rank_in(x1, [1, 2]),
            ]
        ):
            # Get lengths
            num_samples = x0.get_shape()[-1]
            if num_samples is None:
                raise ValueError(
                    "The inner most dimension of x0 and x1 must be statically defined."
                )
            m = 2 * num_samples - 1
            if max_delay is None:
                ncorr = m
            else:
                ncorr = min(m, int(max_delay))
            nfft = 2 ** next_power_of_two(m)
            # Remove DC
            x0 = x0 - tf.reduce_mean(x0, axis=-1, keepdims=True)
            x1 = x1 - tf.reduce_mean(x1, axis=-1, keepdims=True)
            return _gcc(x0, x1, nfft, ncorr, weighting, scale)


def _gcc(x0, x1, nfft, ncorr, weighting, scale):
    # Spectrums
    X0 = tf.math.conj(tf.signal.rfft(x0, [nfft]))  # Reference signal spectrum
    X1 = tf.signal.rfft(x1, [nfft])  # Delayed replica spectrum
    # Cross-spectrum
    R01 = X1 * X0
    # Filter cross-spectrum
    R01 = _filter(R01, weighting)
    # Get back to time
    r01 = tf.signal.irfft(R01, fft_length=[nfft], name="ifft_R01")
    # Check numerics, since ifft gives NaNs for very large or non power of 2 rfft or when using very large ffts
    r01 = tf.debugging.check_numerics(
        r01, "ifft gives NaNs. Hint: ensure nfft=2^n or reduce ifft length"
    )
    # Keep only the lags we want and move negative lags before positive lags
    r01 = _shift(r01, nfft, ncorr)
    return _scale(x0, r01, scale)


def _shift(x, N, ncorr):
    assert N > ncorr
    with tf.name_scope("ifft_shift"):
        left_x = x[..., :ncorr]
        right_x = x[..., N - ncorr : N - 1]
        y = tf.concat([right_x, left_x], axis=-1)
    return y


def _filter(R01, weighting):
    with tf.name_scope("weighting"):
        if weighting == "PHAT":
            filtered_R01 = tf.exp(tf.complex(0.0, tf.math.angle(R01)), name="exp_angle")
        else:
            filtered_R01 = R01
    return filtered_R01


def _scale(x, gcc, scale):
    m = x.get_shape()[-1]
    with tf.name_scope("scale"):
        if scale == "biased":
            return gcc / m
        elif scale == "unbiased":
            gcc_len = gcc.get_shape()[-1]
            L = int((gcc_len - 1) / 2)
            den = m - np.abs(np.arange(-L, L + 1))
            den[den <= 0] = 1
            # workaround for gcc estimators with even number of samples
            den = np.pad(den, (0, gcc_len - len(den)), mode="edge")
            return gcc / den
        else:
            return gcc
