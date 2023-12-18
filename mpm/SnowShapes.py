import numpy as np

def make_snowball(radius: float, center: np.array, density: float) -> np.array:
    area = np.pi * radius**2
    num_points = int(density * area)
    r_vals = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    t_vals = np.random.uniform(0.0, 2*np.pi, num_points)
    position = []
    for i in range(len(r_vals)):
        position.append([r_vals[i]*np.cos(t_vals[i]),
                         r_vals[i]*np.sin(t_vals[i])])
    position = np.array(position)
    position = position + center
    return position
