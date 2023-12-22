# MPM for 2D Snow in Warp
Submission for my CSC2521 final project. An implementation of the material point method for snow, in 2D, following the algorithm described in Stomakhin, 2013. It utilises the NVIDIA Warp package, which can be seen here: https://developer.nvidia.com/warp-python

## Demos
Here are some demos of the algorithm I implemented. Throughout, a time step of 1e-5 and mass 0.05 was used.
### Snowball Drop
For the simple snowball drop, a grid spacing of 0.025. We see some marshmallow and deflating.
![](https://github.com/nathangs6/MPMWarp_2D/blob/main/demo/drop_demo.gif)
### Snowball Crash
For the snowball crash, a grid spacing of 0.016 was used. We see some fracturing, which is desired.
![](https://github.com/nathangs6/MPMWarp_2D/blob/main/demo/crash_demo.gif)
### Snowball Collision
For the two snowballs colliding, a grid spacing of 0.05 and mass of 0.5 was used. We see some fracturing and wall sticking.
![](https://github.com/nathangs6/MPMWarp_2D/blob/main/demo/collide_demo.gif)
### Snowball on a Ramp
For the snowball on a ramp, a grid spacing of 0.05 was used. Snow deflating can be observed. 
![](https://github.com/nathangs6/MPMWarp_2D/blob/main/demo/ramp_demo.gif)
## Setup
To setup the project, do the following:
1. Clone repository to your computer.
2. Setup a python virtual environment, `python -m venv env`
3. Activate python virtual environment, `source env/bin/activate`
4. Install requirements, `python -m pip install -r requirements.txt`

## Future Improvements
* Finalize implementation of the implicit solver.
* Implement moveable collision objects.

## References
Jiang, C., Schroeder, C., Teran, J., Stomakhin, A., & Selle, A. (2016). The material point method for simulating continuum materials. ACM SIGGRAPH 2016 Courses. https://doi.org/10.1145/2897826.2927348

Nygaard, I., melissaEdge, & Oborn, J. (2021). Snow. https://github.com/Azmisov/snow/

Stomakhin, A., Schroeder, C., Chai, L., Teran, J., & Selle, A. (2013). A material point method for snow simulation. ACM Trans. Graph., 32 (4). https://doi.org/10.1145/2461912.2461948

Stomakhin, A., Schroeder, C., Chai, L., Teran, J., & Selle, A. (January 18, 2013). Material point method for snow simulation technical report. https://www.disneyanimation.com/publications/a-material-point-method-for-snow-simulation/
