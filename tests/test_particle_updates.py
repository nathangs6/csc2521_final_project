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
    p = [wp.vec2(0.0,0.0),
         wp.vec2(1.0,2.0),
         wp.vec2(0.5,-2.2)]
    v = [wp.vec2(1.0,-1.0),
         wp.vec2(2.2,3.5),
         wp.vec2(0.0,0.0)]
    p = wp.array(p, dtype=wp.vec2, device="cpu")
    v = wp.array(v, dtype=wp.vec2, device="cpu")
    dt = 0.1
    wp.launch(kernel=src.update_particle_position,
              dim=3,
              inputs=[p, v, dt],
              device="cpu")
    actual = np.array(p)
    expected = [np.array([0.1,-0.1]),
                np.array([1.22,2.35]),
                np.array([0.5,-2.2])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

def test_update_particle_velocity():
    """
    Test update_particle_velocity.
    """
    vp = np.array([wp.vec2(1.0,0.0)])
    new_vg = np.array([wp.vec2(1.0,2.0),
                       wp.vec2(0.3,0.2),
                       wp.vec2(2.2,-3.5)])
    old_vg = np.array([wp.vec2(0.0,0.0),
                       wp.vec2(3.0,2.0),
                       wp.vec2(0.1,0.2)])
    wip = np.array([
        [1.0],
        [0.3],
        [2.2]])
    wpi = wip.transpose()
    a = 0.1

    vp = wp.array(vp, dtype=wp.vec2, device="cpu")
    new_vg = wp.array(new_vg, dtype=wp.vec2, device="cpu")
    old_vg = wp.array(old_vg, dtype=wp.vec2, device="cpu")
    wpi = wp.array(wpi, dtype=wp.float32, device="cpu")
    wp.launch(kernel=src.update_particle_velocity,
              dim=2,
              inputs=[vp, new_vg, old_vg, wpi, a],
              device="cpu")
    actual = np.array(vp)
    expected = [np.array([5918/1000, -5744/1000])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

def test_update_particle_F():
    f = np.array([
        wp.mat22(1.0, 2.0, 0.3, 0.4),
        wp.mat22(0.1, 0.2, -0.4, 1.0)])
    new_vi = wp.array([wp.vec2(0.5,-1.1)], dtype=wp.vec2)
    grad_wpi = wp.array([
        [wp.vec2(1.0,2.0)],
        [wp.vec2(0.3,1.2)]], dtype=wp.vec2, ndim=2)
    dt = 0.1
    f = wp.array(f, dtype=wp.mat22, device="cpu")
    wp.launch(kernel=src.update_particle_F,
              dim=2,
              inputs=[f, new_vi, grad_wpi, dt],
              device="cpu")
    actual = np.array(f)
    expected = [
        np.array([[1080/1000, 2140/1000],[124/1000, 92/1000]]),
                 np.array([[775/10000, 2630/10000],[-3505/10000, 8614/10000]])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

def test_update_particle_FE_FP():
    FE = wp.array([wp.mat22(0.1,0.0,0.0,2.0),
                   wp.mat22(2.0,0.0,0.0,2.0)], dtype=wp.mat22)
    FP = wp.array([wp.mat22(1.0,0.0,0.0,1.0),
                   wp.mat22(1.0,0.0,0.0,2.0)], dtype=wp.mat22)
    F = wp.array([wp.mat22(1.0,0.0,0.0,2.0),
                  wp.mat22(1.0,0.0,0.0,1.0)], dtype=wp.mat22)
    new_vi = wp.array([wp.vec2(1.0,0.0),
                       wp.vec2(0.0,2.0),
                       wp.vec2(0.0,0.0)], dtype=wp.vec2)
    grad_wpi = wp.array([[wp.vec2(1.0,0.0), wp.vec2(0.0,3.0), wp.vec2(0.0,2.0)],
                         [wp.vec2(1.0,2.0), wp.vec2(0.1,0.0), wp.vec2(3.0,2.0)]], dtype=wp.vec2)
    dt = 0.0 # for simple SVD
    theta_c = 0.1
    theta_s = 0.2
    wp.launch(kernel=src.update_particle_FE_FP,
              dim=2,
              inputs=[FE, FP, F, new_vi, grad_wpi, dt, theta_c, theta_s],
              device="cpu")
    actual = np.array(FE)
    expected = np.array([
        [[9/10,0],[0,12/10]],
        [[12/10,0],[0,12/10]]
    ])
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL
    actual = np.array(FP)
    expected = np.array([
        [[10/9,0],[0,20/12]],
        [[10/12,0],[0,10/12]]
    ])
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

