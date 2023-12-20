import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import rasterize as src
import numpy as np
import warp as wp
TOL = 0.00001

##############
### Test N ###
##############
@wp.kernel
def evaluate_N(inputs: wp.array(dtype=wp.float32), outputs: wp.array(dtype=wp.float32)) -> None:
    i = wp.tid()
    outputs[i] = src.N(inputs[i])

def test_N_first_range():
    wp.init()
    test_input = [0.5, -0.5, 0.4]
    m = len(test_input)
    test_input = wp.array(test_input, dtype=wp.float32, device="cpu")
    test_output = wp.zeros(m, dtype=wp.float32, device="cpu")
    wp.launch(kernel=evaluate_N,
              dim=m,
              inputs=[test_input, test_output],
              device="cpu")
    actual = np.array(test_output)
    expected = [23/48, 23/48, 202/375]
    for i in range(m):
        assert abs(actual[i] - expected[i]) <= TOL

def test_N_second_range():
    wp.init()
    test_input = [1.5, -1.8, 1.1]
    m = len(test_input)
    test_input = wp.array(test_input, dtype=wp.float32, device="cpu")
    test_output = wp.zeros(m, dtype=wp.float32, device="cpu")
    wp.launch(kernel=evaluate_N,
              dim=m,
              inputs=[test_input, test_output],
              device="cpu")
    actual = np.array(test_output)
    expected = [1/48, 1/750, 243/2000]
    for i in range(m):
        assert abs(actual[i] - expected[i]) <= TOL

def test_N_third_range():
    wp.init()
    test_input = [4.0, -5.0, -2.5, -2.2]
    m = len(test_input)
    test_input = wp.array(test_input, dtype=wp.float32, device="cpu")
    test_output = wp.zeros(m, dtype=wp.float32, device="cpu")
    wp.launch(kernel=evaluate_N,
              dim=m,
              inputs=[test_input, test_output],
              device="cpu")
    actual = np.array(test_output)
    expected = [0.0, 0.0, 0.0, 0.0]
    for i in range(m):
        assert abs(actual[i] - expected[i]) <= TOL

###############
### Test dN ###
###############
@wp.kernel
def evaluate_dN(inputs: wp.array(dtype=wp.float32), outputs: wp.array(dtype=wp.float32)) -> None:
    i = wp.tid()
    outputs[i] = src.dN(inputs[i])

def test_dN_first_range():
    wp.init()
    test_input = [0.5, -0.5, -0.1, 0.9]
    m = len(test_input)
    test_input = wp.array(test_input, dtype=wp.float32, device="cpu")
    test_output = wp.zeros(m, dtype=wp.float32, device="cpu")
    wp.launch(kernel=evaluate_dN,
              dim=m,
              inputs=[test_input, test_output],
              device="cpu")
    actual = np.array(test_output)
    expected = [-5/8, 5/8, 37/200, -117/200]
    for i in range(m):
        assert abs(actual[i] - expected[i]) <= TOL

def test_dN_second_range():
    wp.init()
    test_input = [1.5, -1.5, -1.8, 1.1]
    m = len(test_input)
    test_input = wp.array(test_input, dtype=wp.float32, device="cpu")
    test_output = wp.zeros(m, dtype=wp.float32, device="cpu")
    wp.launch(kernel=evaluate_dN,
              dim=m,
              inputs=[test_input, test_output],
              device="cpu")
    actual = np.array(test_output)
    expected = [-1/8, 1/8, 1/50, -81/200]
    for i in range(m):
        assert abs(actual[i] - expected[i]) <= TOL

def test_dN_third_range():
    wp.init()
    test_input = [4.0, -5.0, -2.5, -2.2]
    m = len(test_input)
    test_input = wp.array(test_input, dtype=wp.float32, device="cpu")
    test_output = wp.zeros(m, dtype=wp.float32, device="cpu")
    wp.launch(kernel=evaluate_dN,
              dim=m,
              inputs=[test_input, test_output],
              device="cpu")
    actual = np.array(test_output)
    expected = [0.0, 0.0, 0.0, 0.0]
    for i in range(m):
        assert abs(actual[i] - expected[i]) <= TOL

##########################
### Test Interpolation ###
##########################
def test_construct_interpolation():
    wp.init()
    wip = wp.empty(shape=(3,4), dtype=float, device="cpu")
    gwip = wp.empty(shape=(3,4), dtype=wp.vec2, device="cpu")
    np_indices = np.array([
        [0,0],
        [2,4],
        [4,-3]], dtype=np.float32)
    i = wp.from_numpy(np_indices, dtype=wp.vec2, device="cpu")
    np_points = np.array([
        [0,0],
        [1,3],
        [0.1,0.2],
        [0.4,-0.2]], dtype=np.float32)
    p = wp.from_numpy(np_points, dtype=wp.vec2, device="cpu")
    h = 0.1
    wp.launch(kernel=src.construct_interpolations,
              dim=[3,4],
              inputs=[wip, gwip, i, p, h],
              device="cpu")
    actual = np.array(wip)
    expected = np.array([
        [4/9, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1/9]], dtype=np.float32)
    assert np.linalg.norm(actual - expected) <= TOL

##########################
### Test Rasterization ###
##########################
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

def test_basic1():
    # Test first range
    wp.init()
    m_p = wp.array([0.1, 0.4, 0.2], dtype=wp.float32)
    v_p = wp.array([
        wp.vec2(1.0, -2.0),
        wp.vec2(0.0, 0.0),
        wp.vec2(0.5, -5.5)], dtype=wp.vec2)
    wip = wp.array([[1,2,3], [4,5,6], [7,8,9], [0.1,0.2,0.3]], dtype=wp.float32)
    m_g = wp.empty(shape=4, dtype=wp.float32)
    v_g = wp.empty(shape=4, dtype=wp.vec2)
    wp.launch(kernel=src.rasterize_mass,
              dim=m_g.shape[0],
              inputs=[m_p, wip, m_g],
              device="cpu"
              )
    assert np.linalg.norm(m_g - np.array([1.5, 3.6, 5.7, 0.15])) <= TOL
    wp.launch(kernel=src.rasterize_velocity,
              dim=v_g.shape[0],
              inputs=[m_p, v_p, wip, m_g, v_g],
              device="cpu"
              )
    actual = np.array(v_g)
    expected = np.array([
        wp.vec2(4/15, -7/3),
        wp.vec2(5/18, -37/18),
        wp.vec2(16/57, -113/57),
        wp.vec2(4/15, -7/3)])
    for i in range(3, v_g.shape[0]):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL
