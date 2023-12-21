# Import Packages
import os
import sys
curr_directory = os.path.dirname(__file__)
src_dir = os.path.join(curr_directory, "mpm")
sys.path.append(src_dir)
from MPM import MPM
import Scene
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Set Parameters
save = False
dt = 1e-5
implicit=False
step_per_frame = 30
num_frames = 100
radius = 0.05

# Define MPM and Scene
scene = Scene.BallDrop(h=0.01,dt=dt)
mpm = MPM(scene)
mpm.init_animation(device="cpu")

# Make data
data = [mpm.get_position()]
print("Number of particles: " + str(mpm.get_position().shape[0]))
print("Generating data")
for f in range(num_frames):
    start = time.time()
    for i in range(step_per_frame):
        mpm.step(implicit=implicit)
    data.append(mpm.get_position())
    print("Frame " + str(f) + " took " + str(round(time.time() - start,2)) + " seconds to run")

# Setup Plot
plt.figure()
fig, ax = plt.subplots()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim([scene.extents[0][0], scene.extents[0][1]])
ax.set_ylim([scene.extents[1][0]-1.0, scene.extents[1][1]])
ax.set_aspect('equal', 'box')
ax.set_title("Snow")
ax.set_facecolor("xkcd:navy")
moon_radius = 1.0
moon = plt.Circle((scene.extents[0][1], scene.extents[1][1]), moon_radius, color="xkcd:silver")
ax.add_patch(moon)
ax.fill([scene.extents[0][0], scene.extents[0][1], scene.extents[0][1], scene.extents[0][0]],
        [0.0, 0.0, -1.0, -1.0], color="xkcd:brown")


X = data[0]
points = ax.plot(X[:,0], X[:,1], 'o', color='xkcd:white')[0]
body_plots = []
for body in mpm.bodies:
    if body.is_visible():
        name = body.get_name()
        mesh = body.get_mesh()
        v = mesh["vertices"]
        if "box" in name:
            body_plots.append(ax.fill(v[:,0], v[:,1], 'r'))
        else:
            body_plots.append(ax.plot(v[:,0], v[:,1], '-', linewidth=2, color='xkcd:brown', label=body.get_name()))

# Define call back function: will run on every loop
def update(frame):
    X = data[frame]
    points.set_xdata(X[:,0])
    points.set_ydata(X[:,1])
    print(frame)

# Run animation
ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=1, repeat=False)
print("Animation done!")
writer = animation.PillowWriter(fps=30)
ani.save("output/output.gif", writer=writer)
print("Animation saved!")
