import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import init_properties as src
import numpy as np
import warp as wp
TOL = 0.00001


def test_init_cell_density():
    h = 0.1
    m_grid = wp.array([1, 2, 3], dtype=wp.float32)
    cell_density = wp.empty(shape=(3), dtype=wp.float32)
    wp.launch(kernel=src.init_cell_density,
              dim=3,
              inputs=[cell_density, m_grid, h],
              device="cpu")
    actual = np.array(cell_density)
    expected = [1000, 2000, 3000]
    for i in range(len(expected)):
        assert (actual[i] - expected[i]) <= TOL

def test_init_particle_density():
    h = 0.1
    m_grid = wp.array([1, 2], dtype=wp.float32)
    wip = wp.array([
        [1, 2, 3],
        [0.5, 1.3, 2.2]], dtype=wp.float32).transpose()
    d = wp.empty(shape=(3), dtype=wp.float32)
    wp.launch(kernel=src.init_particle_density,
              dim=3,
              inputs=[d, m_grid, wip, h],
              device="cpu")
    actual = np.array(d)
    expected = [2000, 4600, 7400]
    for i in range(len(expected)):
        assert (actual[i] - expected[i]) <= TOL

def test_init_particle_volume():
    m = wp.array([1, 2, 3], dtype=wp.float32)
    d = wp.array([0.2, 4.3, 2.5], dtype=wp.float32)
    v = wp.empty(shape=(3), dtype=wp.float32)
    wp.launch(kernel=src.init_particle_volume,
              dim=3,
              inputs=[v, m, d],
              device="cpu")
    actual = np.array(v)
    expected = [1/0.2, 2/4.3, 3/2.5]
    for i in range(len(expected)):
        assert (actual[i] - expected[i]) <= TOL

