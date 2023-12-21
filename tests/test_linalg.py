import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import linalg as src
import numpy as np
import warp as wp
TOL = 1e-8

def test_array_determinant():
    wp.init()
    F = wp.array([wp.mat22(1.0,0.0,0.0,1.0), wp.mat22(1.0,1.0,1.0,1.0), wp.mat22(1.0,2.0,2.0,-1.0)], dtype=wp.mat22)
    J = wp.empty(shape=3, dtype=wp.float32)
    wp.launch(kernel=src.array_determinant,
              dim=3,
              inputs=[J,F],
              device="cpu")
    actual = np.array(J)
    expected = [1.0, 0.0, -5.0]
    for i in range(3):
        assert abs(actual[i] - expected[i]) < TOL

def test_is_positive_1d():
    wp.init()
    data = wp.array([1.0,2.0,-3.0,-0.05,0.05], dtype=wp.float32)
    output = wp.zeros(shape=5, dtype=wp.int8)
    wp.launch(kernel=src.is_positive_1d,
              dim=5,
              inputs=[output, data],
              device="cpu")
    actual = np.array(output)
    expected = [1,1,-1,-1,1]
    for i in range(len(expected)):
        assert actual[i] == expected[i]

def test_is_positive_2d():
    wp.init()
    data = wp.array([[1.0,-1.0],[-0.5,2.0]], dtype=wp.float32)
    output = wp.zeros(shape=(2,2), dtype=wp.int8)
    wp.launch(kernel=src.is_positive_2d,
              dim=[2,2],
              inputs=[output,data],
              device="cpu")
    actual = np.array(output)
    expected = [[1,-1],[-1,1]]
    for i in range(2):
        for j in range(2):
            assert actual[i][j] == expected[i][j]
