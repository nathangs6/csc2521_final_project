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
