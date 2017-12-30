import math

def next_power_of_two(n):
    """Return res for integer n such that 2^res>= n."""
    mantissa, res = math.frexp(n)
    if (mantissa == 0.5):
        res -= 1
    return res
