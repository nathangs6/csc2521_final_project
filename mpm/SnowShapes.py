import numpy as np

def make_snowball(radius: float, center: np.array, velocity: np.array, mass: float, density: float) -> np.array:
    area = np.pi * radius**2
    num_points = int(density * area / mass)
    r_vals = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    t_vals = np.random.uniform(0.0, 2*np.pi, num_points)
    position = []
    for i in range(len(r_vals)):
        position.append([r_vals[i]*np.cos(t_vals[i]),
                         r_vals[i]*np.sin(t_vals[i])])
    position = np.array(position)
    position = position + center
    velocity = np.zeros_like(position) + velocity
    mass = np.ones(shape=num_points, dtype=float) * mass
    return position, velocity, mass

def make_snow_quad(c: np.array, velocity: np.array, mass: float, density: float):
    """
    Make a quadrilateral that looks like the following:
    c0 ---- c1
     |      |
    c3 ---- c2
    """
    v = c[1] - c[0]
    vn = np.linalg.norm(v)
    v /= vn
    w = c[3] - c[0]
    wn = np.linalg.norm(w)
    w /= wn
    area = vn*wn
    num_points = int(density * area / mass)
    a = np.linspace(0, vn, num_points)
    b = np.linspace(0, wn, num_points)
    position = []
    for i in range(num_points):
        position.append(c[0] + a[i]*v + b[i]*w)
    position = np.array(position)
    velocity = np.zeros_like(position) + velocity
    mass = np.ones(shape=num_points, dtype=float) * mass
    return position, velocity, mass
