import numpy as np
import warp as wp

@wp.kernel
def array_determinant(J: wp.array(dtype=wp.float32),
                      F: wp.array(dtype=wp.mat22)) -> None:
    p = wp.tid()
    J[p] = wp.determinant(F[p])

@wp.kernel
def update_particle_position(
    position: wp.array(dtype=wp.vec2),
    velocity: wp.array(dtype=wp.vec2),
    dt: float
):
    """
    Update position using the old position and velocity.
    """
    p = wp.tid()
    position[p] = position[p] + dt * velocity[p]

@wp.func
def compute_vWp(vg: wp.array(dtype=wp.vec2),
               wp: wp.array(dtype=wp.float32)) -> wp.vec2:
    result = wp.vec2(0.0, 0.0)
    for i in range(vg.shape[0]):
        result += vg[i] * wp[i]
    return result

@wp.kernel
def compute_vW(vWp: wp.array(dtype=wp.vec2),
               vg: wp.array(dtype=wp.vec2),
               wpi: wp.array(dtype=wp.float32, ndim=2)) -> None:
    p = wp.tid()
    result = wp.vec2(0.0,0.0)
    for i in range(vg.shape[0]):
        result += vg[i] * wpi[p,i]
    vWp[p] = result

@wp.kernel
def update_particle_velocity(vp: wp.array(dtype=wp.vec2),
                             new_viWi: wp.array(dtype=wp.vec2),
                             viWi: wp.array(dtype=wp.vec2),
                             a: float) -> None:
    """
    Update particle velocities using the grid velocities.
    """
    p = wp.tid()
    vp[p] = new_viWi[p] + a*(vp[p] - viWi[p])

@wp.kernel
def shift_fe(fe_shifted: wp.array(dtype=wp.mat22),
             fe: wp.array(dtype=wp.mat22),
             gv: wp.array(dtype=wp.mat22),
             dt: float,
             mi: wp.array(dtype=wp.float32)) -> None:
    p = wp.tid()
    fe_shifted[p] = fe[p] + dt * wp.mul(gv[p], fe[p])

@wp.kernel
def update_grad_velocity(gv: wp.array(dtype=wp.mat22),
                         vi: wp.array(dtype=wp.vec2),
                         grad_wpi: wp.array(dtype=wp.vec2, ndim=2),
                         mass: wp.array(dtype=wp.float32)) -> None:
    p = wp.tid()
    result = wp.mat22(0.0,0.0,0.0,0.0)
    for i in range(vi.shape[0]):
        if mass[i] > 0:
            result += wp.outer(vi[i], grad_wpi[p,i])
    gv[p] = result

@wp.kernel
def update_particle_F(f: wp.array(dtype=wp.mat22),
                      gv: wp.array(dtype=wp.mat22),
                      dt: float) -> None:
    p = wp.tid()
    f[p] = f[p] + dt * wp.mul(gv[p], f[p])


@wp.func
def clamp_vec2(A: wp.vec2, lower: float, upper: float) -> wp.vec2:
    result = wp.vec2(0.0,0.0)
    for i in range(2):
        result[i] = wp.clamp(A[i], lower, upper)
    return result

@wp.kernel
def update_particle_FE_FP(fe: wp.array(dtype=wp.mat22),
                          fp: wp.array(dtype=wp.mat22),
                          f: wp.array(dtype=wp.mat22),
                          gv: wp.array(dtype=wp.mat22),
                          dt: float,
                          lower_bound: float,
                          upper_bound: float) -> None:
    p = wp.tid()
    fe_p = fe[p]
    f_p = f[p]
    outer = gv[p]
    fe_hat = fe_p + dt * wp.mul(outer, fe_p)
    AAT     = wp.mul(fe_hat, wp.transpose(fe_hat))
    theta   = 0.5 * wp.atan2(AAT[0,1] + AAT[1,0], AAT[0,0]-AAT[1,1])
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)
    U = wp.mat22(cos_theta,
                 -sin_theta,
                 sin_theta,
                 cos_theta)
    ATA     = wp.mul(wp.transpose(fe_hat), fe_hat)
    phi     = 0.5 * wp.atan2(ATA[0,1] + ATA[1,0], ATA[0,0]-ATA[1,1])
    cos_phi = wp.cos(phi)
    sin_phi = wp.sin(phi)
    trace_AAT = AAT[0,0] + AAT[1,1]
    diff_AAT = AAT[0,0] - AAT[1,1]
    stuff_AAT = wp.sqrt(diff_AAT*diff_AAT + 4.0*AAT[0,1]*AAT[1,0])
    s = wp.vec2(wp.sqrt(0.5 * (trace_AAT + stuff_AAT)),
                wp.sqrt(0.5 * (trace_AAT - stuff_AAT)))
    V = wp.mat22(cos_phi, -sin_phi, sin_phi, cos_phi)
    C = wp.mul(wp.transpose(U), wp.mul(fe_hat, V))
    C = wp.mat22(wp.sign(C[0,0]),0.0,0.0,wp.sign(C[1,1]), dtype=wp.float32)
    V = wp.mul(V, C)
    s = clamp_vec2(s, lower_bound, upper_bound)
    S = wp.diag(s)
    fe[p] = wp.mul(U, wp.mul(S, wp.transpose(V)))
    fp[p] = wp.mul(V, wp.mul(wp.inverse(S), wp.mul(wp.transpose(U), f[p])))

def update_FE_FP(fe: np.array, fp: np.array, f: np.array, gv: np.array, dt: float, theta_c: float, theta_s: float) -> None:
    for p in range(fe.shape[0]):
        fe_hat = fe[p] + dt * gv[p] @ fe[p]
        U, S, V = np.linalg.svd(fe_hat)
        S = np.clip(S, 1-theta_c, 1+theta_c)
        fe[p] = U @ np.diag(S) @ V
        fp[p] = np.transpose(V) @ np.linalg.inv(np.diag(S)) @ np.transpose(U) @ f[p]
