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
def shift_deformation(f: wp.array(dtype=wp.mat22),
                      gv: wp.array(dtype=wp.mat22),
                      dt: float) -> None:
    p = wp.tid()
    f[p] = f[p] + dt * wp.mul(gv[p], f[p])

@wp.kernel
def construct_deformations(fe: wp.array(dtype=wp.mat22),
                           fp: wp.array(dtype=wp.mat22),
                           f: wp.array(dtype=wp.mat22),
                           U: wp.array(dtype=wp.mat22),
                           s: wp.array(dtype=wp.vec2),
                           V: wp.array(dtype=wp.mat22)) -> None:
    p = wp.tid()
    S = wp.diag(s[p])
    fe[p] = wp.mul(U[p], wp.mul(S, wp.transpose(V[p])))
    fp[p] = wp.mul(V[p], wp.mul(wp.inverse(S), wp.mul(wp.transpose(U[p]), f[p])))




def update_deformations(fe: wp.array(dtype=wp.mat22),
                        fp: wp.array(dtype=wp.mat22),
                        f: wp.array(dtype=wp.mat22),
                        gv: wp.array(dtype=wp.mat22),
                        dt: float,
                        lower_bound: float,
                        upper_bound: float,
                        device="cpu") -> None:
    num_p = fe.shape[0]
    wp.launch(kernel=shift_deformation,
               dim=num_p,
               inputs=[f, gv, dt],
               device=device)
    wp.launch(kernel=shift_deformation,
               dim=num_p,
               inputs=[fe, gv, dt],
               device=device)
    U = wp.empty_like(fe)
    s = wp.empty(shape=num_p, dtype=wp.vec2, device=device)
    V = wp.empty_like(fe)
    wp.launch(kernel=linalg.array_svd2,
              dim=num_p,
              inputs=[fe,U,s,V],
              device=device)
    wp.launch(kernel=linalg.array_clamp_vec2,
              dim=num_p,
              inputs=[s, lower_bound, upper_bound],
              device=device)
    wp.launch(kernel=construct_deformations,
              dim=num_p,
              inputs=[fe, fp, f, U, s, V],
              device=device)
