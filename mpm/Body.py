import numpy as np
TOL = 1e-10

class Body:
    _name: str
    _velocity: np.array
    _friction_coefficient: float
    _is_sticky: bool
    _defining_function: callable
    _normal_generator: callable
    _mesh: dict

    def __init__(self, name, mu, v: np.array, defining_function: callable, normal_generator: callable, mesh: dict, is_sticky=False, visualize=True) -> None:
        self._name = name
        self._velocity = v
        self._friction_coefficient = mu
        self._is_sticky = is_sticky
        self._defining_function = defining_function
        self._normal_generator = normal_generator
        self._mesh = mesh
        self._visualize = visualize

    def get_name(self):
        return self._name

    def get_velocity(self):
        return np.copy(self._velocity)

    def set_velocity(self, v: np.array):
        self._velocity = v

    def get_mu(self):
        return self._friction_coefficient

    def is_sticky(self):
        return self._is_sticky

    def check_collision(self, p: np.array) -> bool:
        return self._defining_function(p) <= -TOL

    def get_normal(self, p: np.array) -> np.array:
        return self._normal_generator(p)

    def get_mesh(self):
        return self._mesh

    def is_visible(self):
        return self._visualize


class Wall(Body):
    def __init__(self, name: str, mu: float, v: np.array, n: np.array, x0: np.array, corners: np.array, is_sticky=True, visualize=False) -> None:
        """
        Returns a wall.
        """
        n /= np.linalg.norm(n)
        def wall_function(p):
            return np.dot(n, p-x0)

        def wall_normal(p):
            return n

        mesh = {
            "vertices": np.array([corners[0],
                                  corners[1]]),
            "faces": np.array([[0,1]])
        }
        Body.__init__(self, name, mu, v, wall_function, wall_normal, mesh, is_sticky=is_sticky, visualize=visualize)



class Plane(Body):
    def __init__(self, name, mu, v: np.array, n: np.array, x0: np.array, extent: float, is_sticky=False, visualize=True) -> None:
        """
        Returns the plane corresponding to the equation:
            q[0]x + q[1]y + q[2]z + d = 0
        """
        n /= np.linalg.norm(n)
        def plane_function(p):
            return np.dot(n, p-x0)

        def plane_normal(p):
            return n
        t1 = np.random.rand(2)
        t1 -= t1.dot(n) * n
        t1 /= np.linalg.norm(t1)
        mesh = {
            "vertices": np.array([x0+extent*t1,
                                  x0-extent*t1]),
            "faces": np.array([[0,1]])
        }
        Body.__init__(self, name, mu, v, plane_function, plane_normal, mesh, is_sticky=is_sticky, visualize=visualize)

def get_perp_distance(p, c0, c1):
    u = (c1 - c0) / np.linalg.norm(c1 - c0)
    x = (p - c0) / np.linalg.norm(c1 - c0)
    t = np.dot(u,x)
    nearest = t*u
    dist = np.linalg.norm(nearest-x)
    return dist

class Box(Body):
    def __init__(self, name, mu, v: np.array, c: np.array, is_sticky=False, visualize=True) -> None:
        """
        Returns a box with corners, in clockwise order, c[0], c[1], c[2], c[3]. That is,
        c[0]------c[1]
         |          |
        c[3]------c[2]
        """
        faces = np.array([[c[0], c[1]], [c[1], c[2]], [c[2], c[3]], [c[3], c[0]]])
        def box_function(p):
            u = c[0] - c[1]
            v = c[0] - c[3]
            up = np.dot(u,p)
            vp = np.dot(v,p)
            if up <= np.dot(u,c[1]):
                return 1
            if up >= np.dot(u,c[0]):
                return 1
            if vp <= np.dot(v,c[3]):
                return 1
            if vp >= np.dot(v,c[0]):
                return 1
            return -1
        def box_normal(p):
            """
            Gets edge closest to p and returns its normal vector.
            """
            distances = [get_perp_distance(p, c[0], c[1]),
                         get_perp_distance(p, c[1], c[2]),
                         get_perp_distance(p, c[2], c[3]),
                         get_perp_distance(p, c[3], c[0])]
            e = faces[np.argmin(distances)]
            e = e[1] - e[0]
            n = np.array([-e[1], e[0]])
            return n / np.linalg.norm(n)
        mesh = {
            "vertices": c,
            "faces": np.array([[0,1],[1,2],[2,3],[3,0]])
        }
        Body.__init__(self, name, mu, v, box_function, box_normal, mesh, is_sticky=is_sticky, visualize=visualize)
