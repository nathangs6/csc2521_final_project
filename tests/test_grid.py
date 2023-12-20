import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import grid as src
import warp as wp
import numpy as np
TOL = 0.00001

def test_compute_grid_forces():
    wp.init()
    volume = wp.array([2.0,3.0], dtype=wp.float32)
    grad_wip = wp.array([[wp.vec2(1.0,2.0), wp.vec2(0.0,1.0)]], dtype=wp.vec2)
    stress = wp.array([wp.mat22(1.0,0.0,0.0,1.0),
                       wp.mat22(2.0,0.0,0.0,2.0)], dtype=wp.mat22)
    mass = wp.array([1, 1], dtype=wp.int8)
    grid_forces = wp.empty(shape=1, dtype=wp.vec2)
    wp.launch(kernel=src.compute_grid_forces,
              dim=1,
              inputs=[grid_forces, volume, grad_wip, stress, mass],
              device="cpu")
    actual = np.array(grid_forces)
    expected = np.array([[-2.0, -10.0]])
    assert np.linalg.norm(actual[0] - expected[0]) <= TOL

def test_add_force():
    force = wp.zeros(shape=2, dtype=wp.vec2)
    new_force = wp.vec2(0.0,-10.0)
    wp.launch(kernel=src.add_force,
              dim=2,
              inputs=[force, new_force],
              device="cpu")
    actual = np.array(force)
    expected = np.array([[0.0,-10.0],[0.0,-10.0]])
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL


def test_update_grid_velocities_with_ext_forces():
    old_v = wp.array([wp.vec2(1.0,0.0), wp.vec2(0.0,-2.0)], dtype=wp.vec2)
    mass = wp.array([1.0,2.0], dtype=wp.float32)
    ext_f = wp.array([wp.vec2(0.0,-10.0),wp.vec2(1.0,2.0)], dtype=wp.vec2)
    gravity = wp.vec2(0.0,-10.0)
    dt = 0.1
    new_v = wp.empty_like(old_v)
    wp.launch(kernel=src.update_grid_velocities_with_ext_forces,
              dim=2,
              inputs=[new_v, old_v, mass, ext_f, gravity, dt],
              device = "cpu")
    actual = np.array(new_v)
    expected = [[1.0,-2.0],[1/20,-29/10]]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL


def test_solve_grid_velocity_explicit():
    old_v = wp.array([wp.vec2(1.0,0.0), wp.vec2(0.0,-2.0)], dtype=wp.vec2)
    new_v = wp.empty_like(old_v)
    wp.launch(kernel=src.solve_grid_velocity_explicit,
              dim=2,
              inputs=[new_v, old_v],
              device="cpu")
    actual = np.array(new_v)
    expected = [[1.0,0.0], [0.0,-2.0]]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i]-expected[i]) <= TOL
