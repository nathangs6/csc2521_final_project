import numpy as np
import numpy.linalg as npl
from scipy.linalg import polar
import warp as wp

@wp.func
def lame_parameter(c: float, zeta: float, JP: float):
    return c * wp.exp(zeta * (1.0 - JP))

def lame_parameter_nonwp(c: float, zeta: float, JP: float):
    return c * wp.exp(zeta * (1.0 - JP))

@wp.func
def compute_stress(FE: wp.mat22, JE: wp.float32, FP: wp.mat22, JP: wp.float32, mu0: float, lam0: float, zeta: float) -> wp.mat22:
    mu      = lame_parameter(mu0, zeta, JP)
    lam     = lame_parameter(lam0, zeta, JP)
    AAT     = wp.mul(FE, wp.transpose(FE))
    theta   = 0.5 * wp.atan2(AAT[0,1] + AAT[1,0], AAT[0,0]-AAT[1,1])
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)
    U = wp.mat22(cos_theta,
                 -sin_theta,
                 sin_theta,
                 cos_theta)
    ATA     = wp.mul(wp.transpose(FE), FE)
    phi     = 0.5 * wp.atan2(ATA[0,1] + ATA[1,0], ATA[0,0]-ATA[1,1])
    cos_phi = wp.cos(phi)
    sin_phi = wp.sin(phi)
    V = wp.mat22(cos_phi, -sin_phi, sin_phi, cos_phi)
    C = wp.mul(wp.transpose(U), wp.mul(FE, V))
    C = wp.mat22(wp.sign(C[0,0]),0.0,0.0,wp.sign(C[1,1]), dtype=wp.float32)
    V = wp.mul(V, C)
    RE      = wp.mul(U, wp.transpose(V))
    return 2.0*mu*wp.mul(FE-RE, wp.transpose(FE)) + lam*(JE-1.0)*JE*wp.identity(n=2, dtype=wp.float32)

@wp.kernel
def get_stresses(stress: wp.array(dtype=wp.mat22),
                 FE: wp.array(dtype=wp.mat22),
                 JE: wp.array(dtype=wp.float32),
                 FP: wp.array(dtype=wp.mat22),
                 JP: wp.array(dtype=wp.float32),
                 mu0: float,
                 lam0: float,
                 zeta: float) -> None:
    p = wp.tid()
    stress[p] = compute_stress(FE[p], JE[p], FP[p], JP[p], mu0, lam0, zeta)

@wp.func
def sum_stresses(volume: wp.array(dtype=wp.float32),
                 stress: wp.array(dtype=wp.mat22),
                 grad_wi: wp.array(dtype=wp.vec2)) -> wp.vec2:
    result = wp.vec2(0.0, 0.0)
    for p in range(volume.shape[0]):
        result += volume[p] * wp.mul(stress[p], grad_wi[p])
    return result

@wp.kernel
def compute_grid_forces(
    grid_forces:    wp.array(dtype=wp.vec2),
    volume:         wp.array(dtype=wp.float32),
    grad_wip:       wp.array(dtype=wp.vec2, ndim=2),
    stress:         wp.array(dtype=wp.mat22),
    mass:           wp.array(dtype=wp.float32)
):
    """
    Compute the grid forces.
    """
    i = wp.tid()
    if mass[i] > 0:
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
                                           gravity: wp.vec2,
                                           dt: float) -> None:
    i = wp.tid()
    if mass[i] > 0:
        new_v[i] = old_v[i] + dt * (gravity + ext_f[i] / mass[i])


@wp.kernel
def solve_grid_velocity_explicit(new_v: wp.array(dtype=wp.vec2),
                                 old_v: wp.array(dtype=wp.vec2)) -> None:
    i = wp.tid()
    new_v[i] = old_v[i]

def grad_grad_psi(FE, JE, mu, lam):
    ddpsi = np.zeros(shape=(2,2,2,2), dtype=float)
    R, S = polar(FE)
    X = np.array([[-R[0,1], R[0,0]], [-R[1,1], R[1,0]]])
    T = np.trace(S)
    FIT = np.array([[FE[1,1], -FE[1,0]], [-FE[0,1], FE[0,0]]])
    ddpsi[0,0] = 2 * mu * np.array([[1,0],[0,0]], dtype=float) + \
        2 * mu * R[0,1] * X / T + \
        lam * FE[1,1] * JE * FIT + \
        lam * (JE - 1) * np.array([[0, 0], [0, 1]], dtype=float)
    ddpsi[0,1] = 2 * mu * np.array([[0,1],[0,0]], dtype=float) - \
        2 * mu * R[0,0] * X / T - \
        lam * FE[1,0] * JE * FIT + \
        lam * (JE - 1) * np.array([[0, 0], [-1, 0]], dtype=float)
    ddpsi[1,0] = 2 * mu * np.array([[0,0],[1,0]], dtype=float) + \
        2 * mu * R[1,1] * X / T - \
        lam * FE[0,1] * JE * FIT + \
        lam * (JE - 1) * np.array([[0, -1], [0, 0]], dtype=float)
    ddpsi[1,1] = 2 * mu * np.array([[0,0],[0,1]], dtype=float) - \
        2 * mu * R[1,0] * X / T + \
        lam * FE[0,0] * JE * FIT + \
        lam * (JE - 1) * np.array([[1, 0], [0, 0]], dtype=float)
    return ddpsi

def grad_grad_phi(V: np.array, grad_wi, grad_wj, FEP: np.array, ddpsi: np.array) -> np.array:
    ddphi = np.zeros(shape=(2,2), dtype=float)
    for p in range(V.shape[0]):
        ddphi_p = np.zeros(shape=(2,2), dtype=float)
        for alpha in range(2):
            for tau in range(2):
                for beta in range(2):
                    for sigma in range(2):
                        ddphi_p[alpha,tau] += ddpsi[p][alpha,tau][beta,sigma] * np.dot(grad_wj[p], FEP[p][:,sigma]) * np.dot(grad_wi[p], FEP[p][:,beta])
        ddphi += V[p] * ddphi_p
    return ddphi

def conjugate_residual(A: np.array, b: np.array, maxiter: int):
    """
    Uses the conjugate residual method to solve Ax = b.
    """
    xk = np.zeros(shape=A.shape[1], dtype=float)
    rk = b - np.matmul(A, xk)
    pk = rk
    for k in range(maxiter):
        a = np.matmul(np.transpose(rk), np.matmul(A, rk)) / np.linalg.norm(np.matmul(A, pk))**2
        xknew = xk + ak*pk
        rknew = rk - ak*np.matmul(A,pk)
        betak = np.matmul(np.transpose(rknew), np.matmul(A, rknew)) / np.matmul(rk, np.matmul(A,rk))
        xk = xknew
        rk = rknew
        pknew = rknew +betak*pk
    return xk

def solve_grid_velocity_implicit(v: np.array, m: np.array, grad_wip: np.array, FEP: np.array, FPP: np.array, V: np.array, dt: float, beta: float, mu0: float, lam0: float, zeta: float) -> np.array:
    num_particle = FEP.shape[0]
    num_grid = v.shape[0]
    RHS = v
    LHS = np.zeros(shape=(num_grid, num_grid, 2, 2), dtype=float)
    JEP = np.linalg.det(FEP)
    JPP = np.linalg.det(FPP)
    print("Computing Hessian of Psi")
    ddpsi = np.zeros(shape=(num_particle, 2,2,2,2), dtype=float)
    for p in range(num_particle):
        mu = lame_parameter_nonwp(mu0, zeta, JPP[p])
        lam = lame_parameter_nonwp(lam0, zeta, JPP[p])
        ddpsi[p] = grad_grad_psi(FEP[p], JEP[p], mu, lam)
    print("Computing Hessian of Phi")
    ddphi = np.zeros(shape=(num_grid, num_grid, 2,2), dtype=float)
    for i in range(num_grid):
        print("i = " + str(i) + " / " + str(num_grid))
        for j in range(i,num_grid):
            print("j = " + str(j) + " / " + str(num_grid))
            ddphi[i,j] = grad_grad_phi(V, grad_wip[i], grad_wip[j], FEP, ddpsi)
            ddphi[j,i] = ddphi[i,j]
    for i in range(num_grid):
        for j in range(num_grid):
            LHS[i,j] = np.eye(2, dtype=float) * (i == j)
            if m[i] > 0:
                LHS[i,j] += beta*dt*(1/m[i])*ddphi[i,j]
    print("Solving")
    return conjugate_residual(LHS, RHS, 30)
