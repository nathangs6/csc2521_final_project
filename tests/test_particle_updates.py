import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import particle_updates as src
import warp as wp
import numpy as np
TOL = 0.00001

def test_update_particle_position():
    """
    Test update_particle_position.
    """
    p = [wp.vec3(0.0,0.0,0.0),
         wp.vec3(1.0,2.0,-3.0),
         wp.vec3(0.5,-2.2,4.3)]
    v = [wp.vec3(1.0,-1.0,0.0),
         wp.vec3(2.2,3.5,-6.0),
         wp.vec3(0.0,0.0,0.0)]
    p = wp.array(p, dtype=wp.vec3, device="cpu")
    v = wp.array(v, dtype=wp.vec3, device="cpu")
    dt = 0.1
    wp.launch(kernel=src.update_particle_position,
              dim=3,
              inputs=[p, v, dt],
              device="cpu")
    actual = np.array(p)
    expected = [np.array([0.1,-0.1,0.0]),
                np.array([1.22,2.35,-3.6]),
                np.array([0.5,-2.2,4.3])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

def test_update_particle_velocity():
    """
    Test update_particle_velocity.
    """
    vp = np.array([wp.vec3(1.0,0.0,2.0)])
    new_vg = np.array([wp.vec3(1.0,2.0,3.0),
                       wp.vec3(0.3,0.2,0.1),
                       wp.vec3(2.2,-3.5,1.3)])
    old_vg = np.array([wp.vec3(0.0,0.0,0.0),
                       wp.vec3(3.0,2.0,1.0),
                       wp.vec3(0.1,0.2,0.3)])
    wip = np.array([
        [1.0],
        [0.3],
        [2.2]])
    wpi = wip.transpose()
    a = 0.1

    vp = wp.array(vp, dtype=wp.vec3, device="cpu")
    new_vg = wp.array(new_vg, dtype=wp.vec3, device="cpu")
    old_vg = wp.array(old_vg, dtype=wp.vec3, device="cpu")
    wpi = wp.array(wpi, dtype=wp.float32, device="cpu")
    wp.launch(kernel=src.update_particle_velocity,
              dim=2,
              inputs=[vp, new_vg, old_vg, wpi, a],
              device="cpu")
    actual = np.array(vp)
    expected = [np.array([5.918,-5.744,5.994])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

def test_update_particle_F():
    f = np.array([
        wp.mat33(1.0,-2.0,0.5, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0),
        wp.mat33(-2.1, 1.2, 3.3, 0.0, 1.0, 2.0, 0.7, -1.3, -5.6)])
    new_vi = wp.array([wp.vec3(0.5,-1.1,3.4)], dtype=wp.vec3)
    grad_wpi = wp.array([
        [wp.vec3(1.0,2.0,3.0)],
        [wp.vec3(0.3,1.2,-2.7)]], dtype=wp.vec3, ndim=2)
    dt = 0.1
    f = wp.array(f, dtype=wp.mat33, device="cpu")
    wp.launch(kernel=src.update_particle_F,
              dim=2,
              inputs=[f, new_vi, grad_wpi, dt],
              device="cpu")
    actual = np.array(f)
    expected = [
        np.array([[1.2,-2.1,1.175],[-0.44,0.22,0.515],[2.36,-0.68,7.59]]),
                 np.array([[-2.226,1.4535,4.2255],[0.2772,0.4423,-0.0361],[-0.1568,0.4238,0.6934]])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

def test_update_particle_FE_FP():
    FE = wp.array([wp.mat33(0.1,0.0,0.0,0.0,2.0,0.0,0.0,0.0,3.0),
                   wp.mat33(2.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0,0.2)], dtype=wp.mat33)
    FP = wp.array([wp.mat33(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0),
                   wp.mat33(1.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0,3.0)], dtype=wp.mat33)
    F = wp.array([wp.mat33(1.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0,3.0),
                   wp.mat33(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)], dtype=wp.mat33)
    new_vi = wp.array([wp.vec3(1.0,0.0,0.0),
                       wp.vec3(0.0,2.0,0.0),
                       wp.vec3(0.0,0.0,3.0)], dtype=wp.vec3)
    grad_wpi = wp.array([[wp.vec3(1.0,0.0,0.0), wp.vec3(0.0,3.0,0.0), wp.vec3(0.0,0.0,2.0)],
                         [wp.vec3(1.0,2.0,0.0), wp.vec3(0.1,0.0,0.3), wp.vec3(1.0,2.0,3.0)]], dtype=wp.vec3)
    dt = 0.0 # for simple SVD
    theta_c = 0.1
    theta_s = 0.2
    wp.launch(kernel=src.update_particle_FE_FP,
              dim=2,
              inputs=[FE, FP, F, new_vi, grad_wpi, dt, theta_c, theta_s],
              device="cpu")
    actual_FE = np.array(FE)
    expected = [[[0.9,0,0],[0,1.2,0],[0,0,1.2]],[[1.2,0,0],[0,1.2,0],[0,0,0.9]]]
    for i in range(len(expected)):
        assert np.linalg.norm(actual_FE[i] - expected[i]) <= TOL

