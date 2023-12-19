import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import rasterize as src
import numpy as np
import warp as wp
TOL = 0.00001

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
        print(expected[i], actual[i])
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL
