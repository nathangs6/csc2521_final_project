import numpy as np
import warp as wp

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

@wp.kernel
def init_cell_density(cd: wp.array(dtype=wp.float32),
                      m_g: wp.array(dtype=wp.float32),
                      h: float) -> None:
    i = wp.tid()
    cd[i] = m_g[i] / (h*h*h)

@wp.kernel
def init_particle_density(pd: wp.array(dtype=wp.float32),
                          mg: wp.array(dtype=wp.float32),
                          wpi: wp.array(dtype=wp.float32, ndim=2),
                          h: float) -> None:
    p = wp.tid()
    wp = wpi[p]
    pd_p = float(0.0)
    for i in range(wp.shape[0]):
        pd_p += mg[i] * wp[i]
    pd[p] = pd_p / (h*h*h)

@wp.kernel
def init_particle_volume(v: wp.array(dtype=wp.float32),
                         m: wp.array(dtype=wp.float32),
                         d: wp.array(dtype=wp.float32)) -> None:
    p = wp.tid()
    v[p] = m[p] / d[p]

@wp.func
def sum_points_m(m: wp.array(dtype=wp.float32),
                 wi: wp.array(dtype=wp.float32)) -> wp.float32:
    result = float(0.0)
    for p in range(m.shape[0]):
        result += m[p] * wi[p]
    return result

@wp.kernel
def rasterize_mass(m_p: wp.array(dtype=wp.float32),
                   wip: wp.array(dtype=wp.float32, ndim=2),
                   m_g: wp.array(dtype=wp.float32)) -> None:
    i = wp.tid()
    m_g[i] = sum_points_m(m_p, wip[i])


@wp.func
def sum_points_v(m: wp.array(dtype=wp.float32),
                 v: wp.array(dtype=wp.vec2),
                 wi: wp.array(dtype=wp.float32)) -> wp.vec2:
    result = wp.vec2(0.0,0.0)
    for p in range(m.shape[0]):
        result += m[p] * wi[p] * v[p]
    return result

@wp.kernel
def rasterize_velocity(m_p: wp.array(dtype=wp.float32),
                       v_p: wp.array(dtype=wp.vec2),
                       wip: wp.array(dtype=wp.float32, ndim=2),
                       m_g: wp.array(dtype=wp.float32),
                       v_g: wp.array(dtype=wp.vec2)) -> None:
    i = wp.tid()
    if m_g[i] > 0:
        v_g[i] = sum_points_v(m_p, v_p, wip[i]) / m_g[i]
    else:
        v_g[i] = wp.vec2(0.0,0.0)
