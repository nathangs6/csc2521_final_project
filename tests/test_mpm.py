import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import MPM as src
import Scene as src2
import numpy as np
import warp as wp
TOL = 0.00001

def test_single_drop():
    grid = []
    for i in [-1.0,0.0,1.0]:
        for j in [-1.0,0.0,1.0]:
            for k in [0.0,1.0]:
                grid.append([i,j,k])
    grid = np.array(grid)
    scene = src2.Scene(theta_c=0.1,
                       theta_s=0.2,
                       hardening_coefficient=1.0,
                       mu0=1.0,
                       lam0=1.0,
                       initial_density=100.0,
                       initial_young_modulus=1e5,
                       poisson_ratio=0.2,
                       alpha=0.5,
                       spacing=1.0,
                       dt=1e-5,
                       extents=np.array([[-1,1],[-1,1],[0,1]]),
                       mass=np.array([1.0]),
                       position=np.array([[0.0,0.0,1.0]]),
                       velocity=np.array([[0.0,0.0,0.0]]),
                       bodies = np.array([])
                       )
    mpm = src.MPM(scene)
    mpm.init_animation()
    #print(str(mpm.frame) + ": " + str(mpm.get_position()))
    for i in range(1000):
        mpm.step()
        #print(str(mpm.frame) + ": " + str(mpm.get_position()))


if __name__ == "__main__":
    test_single_drop()
