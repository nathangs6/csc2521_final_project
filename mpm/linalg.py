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

@wp.kernel
def array_clamp_vec2(v: wp.array(dtype=wp.vec2), lower: float, upper: float) -> None:
    """
    Clamps all entries of v.
    """
    p = wp.tid()
    v[p] = clamp_vec2(v[p], lower, upper)

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

@wp.func
def cw_sum_scalar_vector_if(a: wp.array(dtype=wp.float32), v: wp.array(dtype=wp.vec2), check: wp.array(dtype=wp.int8)) -> wp.vec2:
    result = wp.vec2(0.0,0.0)
    for i in range(a.shape[0]):
        if check[i] > 0:
            result += a[i] * v[i]
    return result

@wp.kernel
def array_cw_sum_scalar_vector(output: wp.array(dtype=wp.vec2),
                               v: wp.array(dtype=wp.vec2),
                               A: wp.array(dtype=wp.float32, ndim=2)) -> None:
    p = wp.tid()
    output[p] = cw_sum_scalar_vector(A[p], v)

@wp.kernel
def array_cw_sum_scalar_vector_if(output: wp.array(dtype=wp.vec2),
                               v: wp.array(dtype=wp.vec2),
                               A: wp.array(dtype=wp.float32, ndim=2),
                               check: wp.array(dtype=wp.int8, ndim=2)) -> None:
    p = wp.tid()
    output[p] = cw_sum_scalar_vector_if(A[p], v, check[p])

@wp.kernel
def array_determinant(J: wp.array(dtype=wp.float32),
                      F: wp.array(dtype=wp.mat22)) -> None:
    """
    Computes the determinant of each entry of F and stores it in J.
    """
    tid = wp.tid()
    J[tid] = wp.determinant(F[tid])

@wp.struct
class SVD2_Struct:
    U: wp.mat22
    s: wp.vec2
    V: wp.mat22

@wp.func
def svd2(A: wp.mat22) -> SVD2_Struct:
    result = SVD2_Struct()
    AAT     = wp.mul(A, wp.transpose(A))
    theta   = 0.5 * wp.atan2(AAT[0,1] + AAT[1,0], AAT[0,0]-AAT[1,1])
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)
    result.U = wp.mat22(cos_theta,
                 -sin_theta,
                 sin_theta,
                 cos_theta)
    ATA     = wp.mul(wp.transpose(A), A)
    phi     = 0.5 * wp.atan2(ATA[0,1] + ATA[1,0], ATA[0,0]-ATA[1,1])
    cos_phi = wp.cos(phi)
    sin_phi = wp.sin(phi)
    trace_AAT = AAT[0,0] + AAT[1,1]
    diff_AAT = AAT[0,0] - AAT[1,1]
    stuff_AAT = wp.sqrt(diff_AAT*diff_AAT + 4.0*AAT[0,1]*AAT[1,0])
    result.s = wp.vec2(wp.sqrt(0.5 * (trace_AAT + stuff_AAT)),
                wp.sqrt(0.5 * (trace_AAT - stuff_AAT)))
    V = wp.mat22(cos_phi, -sin_phi, sin_phi, cos_phi)
    C = wp.mul(wp.transpose(result.U), wp.mul(A, V))
    C = wp.mat22(wp.sign(C[0,0]),0.0,0.0,wp.sign(C[1,1]), dtype=wp.float32)
    result.V = wp.mul(V, C)
    return result

@wp.kernel
def array_svd2(A: wp.array(dtype=wp.mat22),
               U: wp.array(dtype=wp.mat22),
               s: wp.array(dtype=wp.vec2),
               V: wp.array(dtype=wp.mat22)) -> None:
    i = wp.tid()
    result = svd2(A[i])
    U[i] = result.U
    s[i] = result.s
    V[i] = result.V

@wp.kernel
def array_polar_from_svd(RE: wp.array(dtype=wp.mat22),
                         U: wp.array(dtype=wp.mat22),
                         V: wp.array(dtype=wp.mat22)) -> None:
    p = wp.tid()
    RE[p] = wp.mul(U[p], wp.transpose(V[p]))

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
