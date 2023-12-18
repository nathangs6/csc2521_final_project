import numpy as np
import numpy.linalg as npl
import warp as wp

@wp.func
def lame_parameter(c: float, zeta: float, JP: float):
    return c * wp.exp(zeta * (1.0 - JP))

@wp.func
def compute_stress(FE: wp.mat22, FP: wp.mat22, mu0: float, lam0: float, zeta: float) -> wp.mat22:
    JE      = wp.determinant(FE)
    JP      = wp.determinant(FP)
    mu      = lame_parameter(mu0, zeta, JP)
    lam     = lame_parameter(lam0, zeta, JP)
    AAT     = FE * wp.transpose(FE)
    ATA     = wp.transpose(FE) * FE
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
    RE      = U * wp.transpose(V)
    return (2.0*mu*(FE-RE)*wp.transpose(FE) + lam*(JE-1.0)*JE*wp.identity(n=2, dtype=wp.float32))/JP

@wp.kernel
def get_stresses(stress: wp.array(dtype=wp.mat22),
                 FE: wp.array(dtype=wp.mat22),
                 FP: wp.array(dtype=wp.mat22),
                 mu0: float,
                 lam0: float,
                 zeta: float) -> None:
    p = wp.tid()
    stress[p] = compute_stress(FE[p], FP[p], mu0, lam0, zeta)

@wp.func
def sum_stresses(volume: wp.array(dtype=wp.float32),
                 stress: wp.array(dtype=wp.mat22),
                 grad_wi: wp.array(dtype=wp.vec2)) -> wp.vec2:
    result = wp.vec2(0.0, 0.0)
    for p in range(volume.shape[0]):
        result += volume[p] * stress[p] * grad_wi[p]
    return result

@wp.kernel
def compute_grid_forces(
    grid_forces:    wp.array(dtype=wp.vec2),
    volume:         wp.array(dtype=wp.float32),
    grad_wip:       wp.array(dtype=wp.vec2, ndim=2),
    stress:         wp.array(dtype=wp.mat22)
):
    """
    Compute the grid forces.
    """
    i = wp.tid()
    grid_forces[i] = -1.0 * sum_stresses(volume, stress, grad_wip[i])

@wp.kernel
def add_force(force: wp.array(dtype=wp.vec2),
              new_force: wp.vec2) -> None:
    i = wp.tid()
    force[i] = force[i] + new_force

@wp.kernel
def update_grid_velocities_with_ext_forces(new_v: wp.array(dtype=wp.vec2),
                                           old_v: wp.array(dtype=wp.vec2),
                                           mass: wp.array(dtype=wp.float32),
                                           ext_f: wp.array(dtype=wp.vec2),
                                           dt: float) -> None:
    i = wp.tid()
    if mass[i] > 0:
        new_v[i] = old_v[i] + ext_f[i] * dt / mass[i]


@wp.kernel
def solve_grid_velocity_explicit(new_v: wp.array(dtype=wp.vec2),
                                 old_v: wp.array(dtype=wp.vec2)) -> None:
    i = wp.tid()
    new_v[i] = old_v[i]
