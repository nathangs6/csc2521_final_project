import numpy as np
import warp as wp
import linalg

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


def compute_vW(vW: wp.array(dtype=wp.vec2),
               vg: wp.array(dtype=wp.vec2),
               wpi: wp.array(dtype=wp.float32, ndim=2)) -> None:
    wp.launch(kernel=linalg.array_cw_sum_scalar_vector,
              dim=vW.shape[0],
              inputs=[vW, vg, wpi],
              device="cpu")

def update_grad_velocity(gv: wp.array(dtype=wp.mat22),
                         vi: wp.array(dtype=wp.vec2),
                         grad_wpi: wp.array(dtype=wp.vec2, ndim=2),
                         check: wp.array(dtype=wp.int8, ndim=2)) -> None:
    wp.launch(kernel=linalg.array_sum_array_outer_if,
              dim=gv.shape[0],
              inputs=[gv,vi,grad_wpi,check],
              device="cpu")

@wp.kernel
def shift_fe(fe_shifted: wp.array(dtype=wp.mat22),
             fe: wp.array(dtype=wp.mat22),
             gv: wp.array(dtype=wp.mat22),
             dt: float) -> None:
    p = wp.tid()
    fe_shifted[p] = fe[p] + dt * wp.mul(gv[p], fe[p])

@wp.kernel
def update_particle_F(f: wp.array(dtype=wp.mat22),
                      gv: wp.array(dtype=wp.mat22),
                      dt: float) -> None:
    p = wp.tid()
    f[p] = f[p] + dt * wp.mul(gv[p], f[p])


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
    s = linalg.clamp_vec2(s, lower_bound, upper_bound)
    S = wp.diag(s)
    fe[p] = wp.mul(U, wp.mul(S, wp.transpose(V)))
    fp[p] = wp.mul(V, wp.mul(wp.inverse(S), wp.mul(wp.transpose(U), f[p])))
