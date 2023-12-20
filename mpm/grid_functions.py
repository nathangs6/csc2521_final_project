import warp as wp
import numpy as np

@wp.func
def N(x: float) -> float:
    """
    Computes the value of N(x).
    """
    ax = wp.abs(x)
    if ax >= 2:
        return 0.0
    ax3 = ax*ax*ax
    x2 = x*x
    if ax >= 1:
        return -(1.0/6.0) * ax3 + x2 - 2.0*ax + 4.0/3.0
    return 0.5 * ax3 - x2 + 2.0/3.0

@wp.func
def dN(x: float) -> float:
    ax = wp.abs(x)
    if ax >= 2:
        return 0.0
    half_axx = 0.5*ax*x
    double_x = 2.0*x
    if ax >= 1:
        return -half_axx + double_x - double_x/ax
    return 3.0*half_axx - double_x

@wp.kernel
def construct_interpolations(wip: wp.array(dtype=float, ndim=2), wip_grad: wp.array(dtype=wp.vec2, ndim=2),
                             index: wp.array(dtype=wp.vec2), points: wp.array(dtype=wp.vec2), h: float) -> None:
    a, b = wp.tid()
    i = index[a]
    p = points[b]
    z0 = p[0]/h - i[0]
    z1 = p[1]/h - i[1]
    N0 = N(z0)
    N1 = N(z1)
    wip[a,b] = N0 * N1
    wip_grad[a,b] = wp.vec2(dN(z0) * N1 / h, N0 * dN(z1) / h)
