import numpy as np
from Body import Wall, Plane, Box
import SnowShapes

class Scene:
    theta_c: float
    theta_s: float
    hardening_coefficient: float
    initial_density: float
    initial_young_modulus: float
    poission_ratio: float
    alpha: float
    spacing: float
    dt: float
    extents: np.array
    mass: np.array
    position: np.array
    velocity: np.array
    grid: np.array
    bodies: np.array

    def __init__(self, spacing, dt, extents, mass, position, velocity, bodies, theta_c=2.5e-2, theta_s=7.5e-3, hardening_coefficient=10.0, initial_density=1e2, initial_young_modulus=1e5, poisson_ratio=0.2, alpha=0.95, density=1e2):
        self.theta_c = theta_c
        self.theta_s = theta_s
        self.hardening_coefficient = hardening_coefficient
        self.initial_density = initial_density
        self.initial_young_modulus = initial_young_modulus
        self.poisson_ratio = poisson_ratio
        self.alpha = alpha
        self.spacing = spacing
        self.dt = dt
        self.mass = mass
        self.position = position
        self.velocity = velocity
        # Make grid
        gb = (extents / spacing).astype(int)
        grid = []
        for i in range(gb[0][0], gb[0][1]+1):
            for j in range(gb[1][0], gb[1][1]+1):
                grid.append([i,j])
        self.grid = np.array(grid, dtype=float)
        self.extents = extents
        # Make scene boundaries
        wall_mu = 1.0
        ## Make floor: fix y = y0
        c0 = [extents[0][0], extents[1][0]]
        c1 = [extents[0][1], extents[1][0]]
        x0 = np.array([extents[0][1]-extents[0][0], extents[1][0]], dtype=float)
        y = np.copy(x0)
        y[1] = extents[1][1]
        n = y - x0
        bodies = np.append(bodies, Wall("floor", mu=wall_mu, v=np.array([0.0,0.0]), n=np.array(n), x0=np.array(x0), corners=np.array([c0,c1]), visualize=False, is_sticky=False))
        ## Make ceiling: fix y = y1
        c0 = [extents[0][0], extents[1][1]]
        c1 = [extents[0][1], extents[1][1]]
        x0 = np.array([extents[0][1]-extents[0][0], extents[1][1]], dtype=float)
        y = np.copy(x0)
        y[1] = extents[1][0]
        n = y - x0
        bodies = np.append(bodies, Wall("ceiling", mu=wall_mu, v=np.array([0.0,0.0]), n=np.array(n), x0=np.array(x0), corners=np.array([c0,c1]), visualize=True))
        ## Make left wall: fix x = x0
        c0 = [extents[0][0], extents[1][0]]
        c1 = [extents[0][0], extents[1][1]]
        x0 = np.array([extents[0][0], extents[1][1]-extents[1][0]], dtype=float)
        y = np.copy(x0)
        y[0] = extents[0][1]
        n = y - x0
        bodies = np.append(bodies, Wall("wall0", mu=wall_mu, v=np.array([0.0,0.0]), n=np.array(n), x0=np.array(x0), corners=np.array([c0,c1]), visualize=True))
        ## Make right wall: fix x = x1
        c0 = [extents[0][1], extents[1][0]]
        c1 = [extents[0][1], extents[1][1]]
        x0 = np.array([extents[0][1], extents[1][1]-extents[1][0]], dtype=float)
        y = np.copy(x0)
        y[0] = extents[0][0]
        n = y - x0
        bodies = np.append(bodies, Wall("wall1", mu=wall_mu, v=np.array([0.0,0.0]), n=np.array(n), x0=np.array(x0), corners=np.array([c0,c1]), visualize=True))
        self.bodies=bodies


class BallDrop(Scene):
    def __init__(self, h=0.05, dt=1e-5, density=1e2):
        extents = np.array([[-1.0,1.0],[0.0,1.0]])
        position, velocity, mass = SnowShapes.make_snowball(0.25, np.array([0.0,0.4]), np.array([0, -9.81]), 0.05, density)
        bodies=np.array([])
        Scene.__init__(self,
                       spacing=h,
                       dt=dt,
                       density=density,
                       extents=extents,
                       mass=mass,
                       position=position,
                       velocity=velocity,
                       bodies=bodies,
                       initial_young_modulus=1.5e5)


class BallCrash(Scene):
    def __init__(self, h=0.1, dt=1e-3, density=1e2):
        extents = np.array([[-2.0,2.0],[0.0,4.1]])
        position, velocity, mass = SnowShapes.make_snowball(0.5, np.array([0.0,3.0]), np.array([0, -40.0]), 0.1, density)
        body = Box("box", mu=0.01, v=np.array([0.0,0.0]),
                   c=np.array([[0.0,2.0],[0.75,1.25],[0.0,0.5],[-0.75,1.25]]))
        bodies=np.array([body])
        Scene.__init__(self,
                       spacing=h,
                       dt=dt,
                       density=density,
                       extents=extents,
                       mass=mass,
                       position=position,
                       velocity=velocity,
                       bodies=bodies,
                       initial_young_modulus=1.5e5)


class TwoSnowBallCollide(Scene):
    def __init__(self, h=0.05, dt=1e-5, density=1e2):
        extents = np.array([[-2.0, 2.0], [0.0,3.0]])
        position1, velocity1, mass1 = SnowShapes.make_snowball(0.5, np.array([-1.0, 1.5]), np.array([20.0,0.0]), 0.5, density)
        position2, velocity2, mass2 = SnowShapes.make_snowball(0.5, np.array([1.0, 2.0]), np.array([-20.0,0.0]), 0.5, density)
        position = np.concatenate((position1, position2), axis=0)
        velocity = np.concatenate((velocity1, velocity2), axis=0)
        mass = np.concatenate((mass1, mass2), axis=0)
        bodies = np.array([])
        Scene.__init__(self,
                       spacing=h,
                       dt=dt,
                       density=density,
                       extents=extents,
                       mass=mass,
                       position=position,
                       velocity=velocity,
                       bodies=bodies)

class RollingBall(Scene):
    def __init__(self, h=0.1, dt=1e-5, density=1e2):
        extents=np.array([[-2.0,2.0],[0.0,4.0]])
        offset = 0.1
        c = np.array([[-2.0,3.1],[2.0,0.1],[2.0,0.0],[-2.0,3.0]])
        position1, velocity1, mass1 = SnowShapes.make_snowball(0.25, np.array([-1.5, 3.5]), np.array([0.0,-10.0]), 0.05, density)
        position2, velocity2, mass2 = SnowShapes.make_snow_quad(c=c, velocity=np.array([0.0,0.0]), mass=0.05, density=density)
        position = np.concatenate((position1, position2), axis=0)
        velocity = np.concatenate((velocity1, velocity2), axis=0)
        mass = np.concatenate((mass1, mass2), axis=0)
        body = Plane("ramp", mu=0.1, v=np.array([0.0,0.0]), extents=np.array([[-2.0,3.0], [2.0,0.0]]))
        bodies = np.array([body])
        Scene.__init__(self,
                       spacing=h,
                       dt=dt,
                       density=density,
                       extents=extents,
                       mass=mass,
                       position=position,
                       velocity=velocity,
                       bodies=bodies)
