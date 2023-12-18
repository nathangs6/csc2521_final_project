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

    def __init__(self, spacing, dt, extents, mass, position, velocity, bodies, theta_c=2.5e-2, theta_s=7.5e-3, hardening_coefficient=10, initial_density=4e2, initial_young_modulus=1e-1, poisson_ratio=0.2, alpha=0.95, density=4e2):
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


class TestScene(Scene):
    def __init__(self, h=0.5, dt=1e-5):
        Scene.__init__(self,
                       spacing=h,
                       dt=dt,
                       extents=np.array([[-1.0,1.0],[0.0,2.0]]),
                       mass=np.array([1.0]),
                       position=np.array([[0.0,1.5]]),
                       velocity=np.array([[0.0,0.0]]),
                       bodies = np.array([])
                       )

class FallingSnow(Scene):
    def __init__(self, h=0.5, dt=1e-5):
        n = 1000
        X = 2 * np.random.rand(n) - 1.0
        Y = np.random.rand(n) + 1.0
        position = np.array(list(zip(X,Y)))
        Scene.__init__(self,
                       spacing=h,
                       dt=dt,
                       extents=np.array([[-1.0,1.0],[0.0,2.0]]),
                       mass=np.ones(shape=position.shape[0], dtype=float),
                       position=position,
                       velocity=np.zeros_like(position, dtype=float),
                       bodies = np.array([])
                       )


class BallDrop(Scene):
    def __init__(self, h=0.1, dt=1e-3):
        position = []
        mass = []
        num_points = 500
        r_vals = np.random.uniform(0, 0.25, num_points)
        t_vals = np.random.uniform(0.01, 2*np.pi-0.01, num_points)
        for i in range(len(r_vals)):
            position.append([r_vals[i]*np.cos(t_vals[i]),
                             r_vals[i]*np.sin(t_vals[i]) + 1.0
                             ])
            mass.append(1.0)
        position = np.array(position)
        mass = np.array(mass)
        velocity = np.zeros_like(position)
        bodies=np.array([])
        Scene.__init__(self,
                       spacing=h,
                       dt=dt,
                       extents=np.array([[-1.0,1.0],[0.0,2.0]]),
                       mass=mass,
                       position=position,
                       velocity=velocity,
                       bodies=bodies)

class BallCrash(Scene):
    def __init__(self, h=0.1, dt=1e-3, density=2e3):
        extents = np.array([[-1.0,1.0],[0.0,2.0]])
        position = SnowShapes.make_snowball(0.25, np.array([0.0,1.5]), density)
        mass = np.ones(position.shape[0], dtype=float)
        velocity = np.zeros_like(position, dtype=float)
        for i in range(velocity.shape[0]):
            velocity[i][1] = -10.0
        body = Box("box", mu=0.01, v=np.array([0.0,0.0]),
                   c=np.array([[0.0,0.9],[0.2,0.3],[0.0,0.1],[-0.2,0.3]]))
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
                       theta_c=2.5e-1,
                       theta_s=7.5e-2,
                       poisson_ratio=0.9)


class TwoSnowBallCollide(Scene):
    def __init__(self, h=1.0, dt=1e-3, density=1e2):
        extents = np.array([[-5.0, 5.0], [0.0,6.0]])
        position1 = SnowShapes.make_snowball(1.0, np.array([-2.0, 2.5]), density)
        velocity1 = np.zeros_like(position1, dtype=float)
        for i in range(velocity1.shape[0]):
            velocity1[i][0] = 20.0
        position2 = SnowShapes.make_snowball(1.0, np.array([2.0, 3.5]), density)
        velocity2 = -np.copy(velocity1)
        position = np.concatenate((position1, position2), axis=0)
        print(position.shape)
        velocity = np.concatenate((velocity1, velocity2), axis=0)
        mass = np.ones(position.shape[0], dtype=float)
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
