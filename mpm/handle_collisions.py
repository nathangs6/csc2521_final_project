import numpy as np
import numpy.linalg as npl
from Body import Body

def compute_collision(p: np.array, v: np.array, body: Body) -> np.array:
    vco = body.get_velocity()
    n = body.get_normal(p)
    mu = body.get_mu()
    vrel = v - vco
    vn = np.dot(vrel, n)
    if vn >= 0:
        return v
    vt = vrel - vn*n
    if body.is_sticky():
        vrel = np.array([0.0,0.0])
    elif npl.norm(vt) <= -mu*vn:
        vrel = np.array([0.0,0.0])
    else:
        vrel = vt + mu*vn*vt/npl.norm(vt)
    return vrel + vco

def handle_all_collisions(x: np.array, v: np.array, dt: float, bodies: np.array, grid_masses=None) -> np.array:
    new_v = v
    for k in range(x.shape[0]):
        if grid_masses is None or grid_masses[k] > 0:
            for body in bodies:
                if body.check_collision(x[k] + dt*v[k]):
                    new_v[k] = compute_collision(x[k], v[k], body)
    return new_v
