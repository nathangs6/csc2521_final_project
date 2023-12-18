import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import grid_updates as src
import warp as wp
import numpy as np
TOL = 0.00001

def test_get_stresses():
    wp.init()
    FE = wp.array([wp.mat33(2.0,0.0,0.0,
                            0.0,1.0,0.0,
                            0.0,0.0,1.0)], dtype=wp.mat33)
    FP = wp.array([wp.mat33(3.0,0.0,0.0,
                            0.0,1.0,0.0,
                            0.0,0.0,1.0)], dtype=wp.mat33)
    mu0 = 1.0
    lam0 = 1.0
    zeta = 1.0
    stress = wp.empty_like(FE)
    wp.launch(kernel=src.get_stresses,
              dim=1,
              inputs=[stress, FE, FP, mu0, lam0, zeta],
              device="cpu")
    actual = np.array(stress)
    print(actual)
    expected = [(2.0/3.0)*np.exp(-2)*np.array([[3.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])]
    assert np.linalg.norm(actual - expected) <= TOL

def test_compute_grid_forces():
    wp.init()
    volume = wp.array([2.0,3.0], dtype=wp.float32)
    grad_wip = wp.array([[wp.vec3(1.0,2.0,3.0), wp.vec3(0.0,0.0,1.0)]], dtype=wp.vec3)
    stress = wp.array([wp.mat33(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0),
                       wp.mat33(2.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0,2.0)], dtype=wp.mat33)
    grid_forces = wp.empty(shape=1, dtype=wp.vec3)
    wp.launch(kernel=src.compute_grid_forces,
              dim=1,
              inputs=[grid_forces, volume, grad_wip, stress],
              device="cpu")
    actual = np.array(grid_forces)
    expected = [[-2.0, -4.0, -12.0]]
    assert np.linalg.norm(actual[0] - expected[0]) <= TOL

def test_add_force():
    force = wp.zeros(shape=2, dtype=wp.vec3)
    new_force = wp.vec3(0.0,0.0,-10.0)
    wp.launch(kernel=src.add_force,
              dim=2,
              inputs=[force, new_force],
              device="cpu")
    actual = np.array(force)
    expected = np.array([[0.0,0.0,-10.0],[0.0,0.0,-10.0]])
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL


def test_update_grid_velocities_with_ext_forces():
    old_v = wp.array([wp.vec3(1.0,0.0,0.0), wp.vec3(0.0,-2.0,1.0)], dtype=wp.vec3)
    mass = wp.array([1.0,2.0], dtype=wp.float32)
    ext_f = wp.array([wp.vec3(0.0,0.0,-10.0),wp.vec3(1.0,2.0,3.0)], dtype=wp.vec3)
    dt = 0.1
    new_v = wp.empty_like(old_v)
    wp.launch(kernel=src.update_grid_velocities_with_ext_forces,
              dim=2,
              inputs=[new_v, old_v, mass, ext_f, dt],
              device = "cpu")
    actual = np.array(new_v)
    expected = [[1.0,0.0,-1.0],[0.05,-1.9,1.15]]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL


def test_solve_grid_velocity_explicit():
    old_v = wp.array([wp.vec3(1.0,0.0,0.0), wp.vec3(0.0,-2.0,1.0)], dtype=wp.vec3)
    new_v = wp.empty_like(old_v)
    wp.launch(kernel=src.solve_grid_velocity_explicit,
              dim=2,
              inputs=[new_v, old_v],
              device="cpu")
    actual = np.array(new_v)
    expected = [[1.0,0.0,0.0], [0.0,-2.0,1.0]]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i]-expected[i]) <= TOL
