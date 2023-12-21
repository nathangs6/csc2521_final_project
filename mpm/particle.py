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
    This is a sped up formulation of the PIC/FLIP scheme.
    Essentially, v = (1-a)*v_PIC + a*v_FLIP
    """
    p = wp.tid()
    vp[p] = new_viWi[p] + a*(vp[p] - viWi[p])


def compute_vW(vW: wp.array(dtype=wp.vec2),
               vg: wp.array(dtype=wp.vec2),
               wpi: wp.array(dtype=wp.float32, ndim=2),
               wpi_check: wp.array(dtype=wp.int8, ndim=2)) -> None:
    """
    Computes the useful intermediary values sum(vg * wp).
    """
    wp.launch(kernel=linalg.array_cw_sum_scalar_vector_if,
              dim=vW.shape[0],
              inputs=[vW, vg, wpi, wpi_check],
              device="cpu")

def update_grad_velocity(gv: wp.array(dtype=wp.mat22),
                         vi: wp.array(dtype=wp.vec2),
                         grad_wpi: wp.array(dtype=wp.vec2, ndim=2),
                         check: wp.array(dtype=wp.int8, ndim=2)) -> None:
    """
    Computes the useful intermediary values of grad(v_p^n+1).
    """
    wp.launch(kernel=linalg.array_sum_array_outer_if,
              dim=gv.shape[0],
              inputs=[gv,vi,grad_wpi,check],
              device="cpu")

@wp.kernel
def shift_deformation(f: wp.array(dtype=wp.mat22),
                      gv: wp.array(dtype=wp.mat22),
                      dt: float) -> None:
    """
    Shifts f with the matrix gv and the coefficient dt.
    """
    p = wp.tid()
    f[p] = f[p] + dt * wp.mul(gv[p], f[p])

@wp.kernel
def construct_deformations(fe: wp.array(dtype=wp.mat22),
                           fp: wp.array(dtype=wp.mat22),
                           f: wp.array(dtype=wp.mat22),
                           U: wp.array(dtype=wp.mat22),
                           s: wp.array(dtype=wp.vec2),
                           V: wp.array(dtype=wp.mat22)) -> None:
    """
    Constructs the updated elastic and plastic deformations.
    """
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
    """
    Updates the total, elastic, and plastic deformations.
    """
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

@wp.kernel
def construct_lame(mu: wp.array(dtype=wp.float32),
                   lam: wp.array(dtype=wp.float32),
                   JP: wp.array(dtype=wp.float32),
                   mu0: float,
                   lam0: float,
                   zeta: float) -> None:
    """
    Constructs the two lame parameters mu and lambda.
    """
    p = wp.tid()
    x = wp.exp(zeta * (1.0 - JP[p]))
    mu[p] = mu0 * x
    lam[p] = lam0 * x

@wp.kernel
def construct_stress(stress: wp.array(dtype=wp.mat22),
                     FE: wp.array(dtype=wp.mat22),
                     JE: wp.array(dtype=wp.float32),
                     RE: wp.array(dtype=wp.mat22),
                     mu: wp.array(dtype=wp.float32),
                     lam: wp.array(dtype=wp.float32)) -> None:
    p = wp.tid()
    stress[p] = 2.0*mu[p]*wp.mul(FE[p]-RE[p], wp.transpose(FE[p])) + lam[p]*(JE[p]-1.0)*JE[p]*wp.identity(n=2, dtype=wp.float32)



def get_stresses(stress: wp.array(dtype=wp.mat22),
                 FEH: wp.array(dtype=wp.mat22),
                 JE: wp.array(dtype=wp.float32),
                 FP: wp.array(dtype=wp.mat22),
                 JP: wp.array(dtype=wp.float32),
                 mu: wp.array(dtype=wp.float32),
                 lam: wp.array(dtype=wp.float32),
                 device="cpu") -> None:
    num_p = FEH.shape[0]
    U = wp.empty_like(FEH)
    s = wp.empty(shape=num_p, dtype=wp.vec2, device=device)
    V = wp.empty_like(FEH)
    wp.launch(kernel=linalg.array_svd2,
              dim=num_p,
              inputs=[FEH,U,s,V],
              device=device)
    wp.launch(kernel=linalg.array_polar_from_svd,
              dim=num_p,
              inputs=[U, U, V],
              device=device)
    wp.launch(kernel=construct_stress,
              dim=num_p,
              inputs=[stress, FEH, JE, U, mu, lam],
              device=device)
