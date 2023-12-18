import numpy as np
import warp as wp

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
