import numpy as np
import warp as wp

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
def compute_vW(vg: wp.array(dtype=wp.vec2),
               wp: wp.array(dtype=wp.float32)) -> wp.vec2:
    result = wp.vec2(0.0, 0.0)
    for i in range(vg.shape[0]):
        result += vg[i] * wp[i]
    return result

@wp.kernel
def update_particle_velocity(vp: wp.array(dtype=wp.vec2),
                             new_vg: wp.array(dtype=wp.vec2),
                             old_vg: wp.array(dtype=wp.vec2),
                             wpi: wp.array(dtype=wp.float32, ndim=2),
                             a: float) -> None:
    """
    Update particle velocities using the grid velocities.
    """
    p = wp.tid()
    vp[p] = compute_vW(new_vg, wpi[p]) + a*(vp[p] - compute_vW(old_vg, wpi[p]))

@wp.func
def compute_outer_vi_gradwp(vi: wp.array(dtype=wp.vec2),
                            grad_wp: wp.array(dtype=wp.vec2)) -> wp.mat22:
    result = wp.mat22(0.0,0.0,0.0,0.0)
    for i in range(vi.shape[0]):
        result += wp.outer(vi[i], grad_wp[i])
    return result

@wp.kernel
def update_particle_F(f: wp.array(dtype=wp.mat22),
                      new_vi: wp.array(dtype=wp.vec2),
                      grad_wpi: wp.array(dtype=wp.vec2, ndim=2),
                      dt: float) -> None:
    p = wp.tid()
    f[p] = f[p] + dt * wp.mul(compute_outer_vi_gradwp(new_vi, grad_wpi[p]), f[p])


@wp.func
def clamp_vec2(A: wp.vec2, lower: float, upper: float) -> wp.vec2:
    result = wp.vec2(0.0,0.0)
    for i in range(2):
        result[i] = wp.clamp(A[i], lower, upper)
    return result

@wp.func
def svd2(A: wp.mat22, U: wp.mat22, S: wp.vec2, V: wp.mat22) -> None:
    AAT     = A * wp.transpose(A)
    ATA     = wp.transpose(A) * A
    theta   = 0.5 * wp.atan2(AAT[0][1], AAT[0][0]-AAT[1][1])
    U[0][0] = wp.cos(theta)
    U[0][1] = -wp.sin(theta)
    U[1][0] = wp.sin(theta)
    U[1][1] = wp.cos(theta)
    S[0]    = wp.sqrt(0.5*(AAT[0][0] + AAT[1][1] + wp.sqrt((AAT[0][0]-AAT[1][1])*(AAT[0][0]-AAT[1][1])+4.0*AAT[0][1])))
    S[1]    = wp.sqrt(0.5*(AAT[0][0] + AAT[1][1] - wp.sqrt((AAT[0][0]-AAT[1][1])*(AAT[0][0]-AAT[1][1])+4.0*AAT[0][1])))
    c0      = wp.sign(S[0])
    c1      = wp.sign(S[1])
    phi     = 0.5 * wp.atan2(ATA[0][1], ATA[0][0]-ATA[1][1])
    V[0][0] = c0*wp.cos(phi)
    V[0][1] = -c1*wp.sin(phi)
    V[1][0] = c0*wp.sin(phi)
    V[1][1] = c1*wp.cos(phi)

@wp.kernel
def update_particle_FE_FP(fe: wp.array(dtype=wp.mat22),
                          fp: wp.array(dtype=wp.mat22),
                          f: wp.array(dtype=wp.mat22),
                          new_vi: wp.array(dtype=wp.vec2),
                          grad_wpi: wp.array(dtype=wp.vec2, ndim=2),
                          dt: float,
                          theta_c: float,
                          theta_s: float) -> None:
    p = wp.tid()
    fe_p = fe[p]
    f_p = f[p]
    grad_wpi_p = grad_wpi[p]
    outer = compute_outer_vi_gradwp(new_vi, grad_wpi_p)
    fe_hat = fe_p + dt * wp.mul(outer, fe_p)
    AAT     = fe_hat * wp.transpose(fe_hat)
    ATA     = wp.transpose(fe_hat) * fe_hat
    theta   = 0.5 * wp.atan2(AAT[0,1], AAT[0,0]-AAT[1,1])
    U = wp.mat22(wp.cos(theta),
                 -wp.sin(theta),
                 wp.sin(theta),
                 wp.cos(theta))
    S = wp.vec2(wp.sqrt(0.5*(AAT[0,0] + AAT[1,1] + wp.sqrt((AAT[0,0]-AAT[1,1])*(AAT[0,0]-AAT[1,1])+4.0*AAT[0,1]))),
                wp.sqrt(0.5*(AAT[0,0] + AAT[1,1] - wp.sqrt((AAT[0,0]-AAT[1,1])*(AAT[0,0]-AAT[1,1])+4.0*AAT[0,1]))))
    c0      = wp.sign(S[0])
    c1      = wp.sign(S[1])
    phi     = 0.5 * wp.atan2(ATA[0,1], ATA[0,0]-ATA[1,1])
    V = wp.mat22(c0*wp.cos(phi),
                 -c1*wp.sin(phi),
                 c0*wp.sin(phi),
                 c1*wp.cos(phi))
    S = clamp_vec2(S, 1.0-theta_c, 1.0+theta_s)
    fe[p] = U * wp.diag(S) * wp.transpose(V)
    fp[p] = V * wp.inverse(wp.diag(S)) * wp.transpose(U) * f_p
