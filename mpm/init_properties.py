import numpy as np
import warp as wp

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
