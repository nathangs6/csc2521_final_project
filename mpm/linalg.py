"""
Title: linalg.py
Description: linear algebra routines for warp arrays.
"""
import warp as wp
LINALG_TOL = wp.constant(1e-10)

@wp.func
def clamp_vec2(v: wp.vec2, lower: float, upper: float) -> wp.vec2:
    """
    Clamps the entries of v to be in the interval [lower, upper].
    """
    return wp.vec2(wp.clamp(v[0], lower, upper),
                   wp.clamp(v[1], lower, upper))

@wp.func
def sum_array_outer(v: wp.array(dtype=wp.vec2), w: wp.array(dtype=wp.vec2)) -> wp.mat22:
    result = wp.mat22(0.0,0.0,0.0,0.0)
    for i in range(v.shape[0]):
        result += wp.outer(v[i], w[i])
    return result

@wp.func
def sum_array_outer_if(v: wp.array(dtype=wp.vec2), w: wp.array(dtype=wp.vec2), c: wp.array(dtype=wp.int8)) -> wp.mat22:
    result = wp.mat22(0.0,0.0,0.0,0.0)
    for i in range(v.shape[0]):
        if c[i] > 0:
            result += wp.outer(v[i],w[i])
    return result

@wp.kernel
def array_sum_array_outer_if(output: wp.array(dtype=wp.mat22),
                             v: wp.array(dtype=wp.vec2),
                             A: wp.array(dtype=wp.vec2, ndim=2),
                             C: wp.array(dtype=wp.int8, ndim=2)) -> None:
    p = wp.tid()
    output[p] = sum_array_outer_if(v, A[p], C[p])


@wp.func
def cw_sum_scalar_vector(a: wp.array(dtype=wp.float32), v: wp.array(dtype=wp.vec2)) -> wp.vec2:
    result = wp.vec2(0.0,0.0)
    for i in range(a.shape[0]):
        result += a[i] * v[i]
    return result

@wp.kernel
def array_cw_sum_scalar_vector(output: wp.array(dtype=wp.vec2),
                               v: wp.array(dtype=wp.vec2),
                               A: wp.array(dtype=wp.float32, ndim=2)) -> None:
    p = wp.tid()
    output[p] = cw_sum_scalar_vector(A[p], v)

@wp.kernel
def array_determinant(J: wp.array(dtype=wp.float32),
                      F: wp.array(dtype=wp.mat22)) -> None:
    """
    Computes the determinant of each entry of F and stores it in J.
    """
    tid = wp.tid()
    J[tid] = wp.determinant(F[tid])

@wp.kernel
def is_positive_1d(output: wp.array(dtype=wp.int8),
                   data: wp.array(dtype=wp.float32)) -> None:
    """
    Stores whether data is positive into output.
    """
    tid = wp.tid()
    check = data[tid] > LINALG_TOL
    if check:
        output[tid] = wp.int8(1)
    else:
        output[tid] = wp.int8(-1)

@wp.kernel
def is_positive_2d(output: wp.array(dtype=wp.int8, ndim=2), data: wp.array(dtype=wp.float32, ndim=2)):
    """
    Stores whether data is positive into output.
    """
    i,j = wp.tid()
    check = data[i,j] > LINALG_TOL
    if check:
        output[i,j] = wp.int8(1)
    else:
        output[i,j] = wp.int8(-1)
