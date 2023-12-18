# Import packages
import numpy as np
import warp as wp
# Import source
from Scene import Scene
from Body import Plane
import init_properties
import rasterize
import particle_updates
import grid_functions
import grid_updates
import handle_collisions

class MPM:
    scene: Scene
    p: dict     # parameters
    pp: dict    # particle properties
    num_p: float
    gp: dict    # grid properties
    num_g: float
    interp: dict
    frame: int

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        E0 = scene.initial_young_modulus
        nu = scene.poisson_ratio
        mu0 = E0 / (2 * (1 + nu))
        lam0 = E0 * nu / ((1 + nu) * (1 - nu))
        print("mu0: ", mu0, "lam0:", lam0)
        self.p = {
            "dt": scene.dt,
            "h": scene.spacing,
            "alpha": scene.alpha,
            "theta_c": scene.theta_c,
            "theta_s": scene.theta_s,
            "hardening_coeff": scene.hardening_coefficient,
            "mu0": mu0,
            "lam0": lam0,
            "init_density": scene.initial_density,
            "init_ym": scene.initial_young_modulus,
            "poisson_ratio": scene.poisson_ratio
        }
        self.num_p = len(scene.mass)
        self.pp = {
            "mass": scene.mass,
            "volume": np.empty(shape=self.num_p, dtype=np.float32),
            "density": np.empty(shape=self.num_p, dtype=np.float32),
            "position": scene.position,
            "velocity": scene.velocity,
            "force": np.zeros(shape=(self.num_p,2,2), dtype=np.float32),
            "force_elastic": np.zeros(shape=(self.num_p,2,2), dtype=np.float32),
            "force_plastic": np.zeros(shape=(self.num_p,2,2), dtype=np.float32),
        }
        for i in range(self.num_p):
            self.pp["force"][i] = np.eye(2, dtype=float)
            self.pp["force_elastic"][i] = np.eye(2, dtype=float)
            self.pp["force_plastic"][i] = np.eye(2, dtype=float)
        self.num_g = len(scene.grid)
        self.gp = {
            "position": scene.grid,
            "velocity": np.empty(shape=(self.num_g,2), dtype=np.float32),
            "mass": np.empty(shape=self.num_g, dtype=np.float32),
            "density": np.empty(shape=self.num_g, dtype=np.float32)
        }
        self.interp = {
            "wip": np.empty(shape=(self.num_g, self.num_p), dtype=np.float32),
            "gwip": np.empty(shape=(self.num_g, self.num_p, 2), dtype=np.float32)
        }
        self.bodies = scene.bodies
        self.frame = 0

    def get_position(self):
        return np.array(self.pp["position"])

    def init_animation(self, implicit=False, device="cpu"):
        wp.init()
        self.move_to_device(device=device)

    def move_to_device(self, device="cpu"):
        # Move particle properties
        self.pp["mass"] = wp.array(self.pp["mass"], dtype=wp.float32, device=device)
        self.pp["volume"] = wp.array(self.pp["volume"], dtype=wp.float32, device=device)
        self.pp["density"] = wp.array(self.pp["density"], dtype=wp.float32, device=device)
        self.pp["position"] = wp.array(self.pp["position"], dtype=wp.vec2, device=device)
        self.pp["velocity"] = wp.array(self.pp["velocity"], dtype=wp.vec2, device=device)
        self.pp["force"] = wp.array(self.pp["force"], dtype=wp.mat22, device=device)
        self.pp["force_elastic"] = wp.array(self.pp["force_elastic"], dtype=wp.mat22, device=device)
        self.pp["force_plastic"] = wp.array(self.pp["force_plastic"], dtype=wp.mat22, device=device)
        # Move grid properties
        self.gp["position"] = wp.array(self.gp["position"], dtype=wp.vec2, device=device)
        self.gp["velocity"] = wp.array(self.gp["velocity"], dtype=wp.vec2, device=device)
        self.gp["mass"] = wp.array(self.gp["mass"], dtype=wp.float32, device=device)
        self.gp["density"] = wp.array(self.gp["density"], dtype=wp.float32, device=device)
        # Move interpolation values
        self.interp["wip"] = wp.array(self.interp["wip"], dtype=wp.float32, device=device)
        self.interp["gwip"] = wp.array(self.interp["gwip"], dtype=wp.vec2, device=device)

    def reset_scratchpad(self, device="cpu"):
        self.gp["mass"] = wp.zeros_like(self.gp["mass"])
        self.gp["velocity"] = wp.zeros_like(self.gp["velocity"])

    def step(self, implicit=False, device="cpu") -> None:
        """
        Execute one time step.
        """
        # Step 0: reset scratchpad
        self.reset_scratchpad(device=device)
        # Step 1: rasterize particle data to the grid
        ## Update values
        wp.launch(kernel=grid_functions.construct_interpolation,
                  dim=[self.num_g, self.num_p],
                  inputs=[self.interp["wip"], self.gp["position"], self.pp["position"], self.p["h"]],
                  device=device)
        wp.launch(kernel=grid_functions.construct_interpolation_grad,
                  dim=[self.num_g, self.num_p],
                  inputs=[self.interp["gwip"], self.gp["position"], self.pp["position"], self.p["h"]],
                  device=device)
        wp.launch(kernel=rasterize.rasterize_mass,
                  dim=self.num_g,
                  inputs=[self.pp["mass"], self.interp["wip"], self.gp["mass"]],
                  device=device)
        wp.launch(kernel=rasterize.rasterize_velocity,
                  dim=self.num_g,
                  inputs=[self.pp["mass"], self.pp["velocity"], self.interp["wip"], self.gp["mass"], self.gp["velocity"]],
                  device=device)

        # Step 2: compute particle volumes and densities
        if self.frame == 0:
            wp.launch(kernel=init_properties.init_cell_density,
                      dim=self.num_g,
                      inputs=[self.gp["density"], self.gp["mass"], self.p["h"]],
                      device=device)
            wp.launch(kernel=init_properties.init_particle_density,
                      dim=self.num_p,
                      inputs=[self.pp["density"], self.gp["mass"], self.interp["wip"].transpose(), self.p["h"]],
                      device=device)
            wp.launch(kernel=init_properties.init_particle_volume,
                      dim=self.num_p,
                      inputs=[self.pp["volume"], self.pp["mass"], self.pp["density"]],
                      device=device)

        # Step 3: compute grid forces
        stresses = wp.zeros_like(self.pp["force"], device=device)
        wp.launch(kernel=grid_updates.get_stresses,
                  dim=self.num_p,
                  inputs=[stresses, self.pp["force_elastic"], self.pp["force_plastic"], self.p["mu0"], self.p["lam0"], self.p["hardening_coeff"]],
                  device=device)
        grid_forces = wp.zeros_like(self.gp["velocity"], device=device)
        wp.launch(kernel=grid_updates.compute_grid_forces,
                  dim=self.num_g,
                  inputs=[grid_forces, self.pp["volume"], self.interp["gwip"], stresses],
                  device=device)
        wp.launch(kernel=grid_updates.add_force,
                  dim=self.num_g,
                  inputs=[grid_forces, wp.vec2(0.0,-9.81)],
                  device=device)

        # Step 4: update velocities on grid
        vg_temp = wp.zeros_like(self.gp["velocity"])
        #grid_forces = wp.array(grid_forces, dtype=wp.vec2, device=device)
        wp.launch(kernel=grid_updates.update_grid_velocities_with_ext_forces,
                  dim=self.num_g,
                  inputs=[vg_temp, self.gp["velocity"], self.gp["mass"], grid_forces, self.p["dt"]],
                  device=device)

        # Step 5: handle grid-based body collisions
        ## Put necessary resources onto CPU
        grid_coords = self.p["h"] * self.gp["position"].numpy()
        velocities = vg_temp.numpy()
        velocities = handle_collisions.handle_all_collisions(grid_coords, velocities, self.p["dt"], self.bodies)


        # Step 6: solve the linear system
        vg_temp = wp.array(velocities, dtype=wp.vec2, device=device)
        new_vg = wp.zeros_like(vg_temp, device=device)
        if not implicit:
            wp.launch(kernel=grid_updates.solve_grid_velocity_explicit,
                      dim=self.num_g,
                      inputs=[new_vg, vg_temp],
                      device=device)
        else:
            raise NotImplementedError

        # Step 7: update deformation gradient
        wp.launch(kernel=particle_updates.update_particle_F,
                  dim=self.num_p,
                  inputs=[self.pp["force"], new_vg, self.interp["gwip"].transpose(), self.p["dt"]],
                  device=device)
        wp.launch(kernel=particle_updates.update_particle_FE_FP,
                  dim=self.num_p,
                  inputs=[self.pp["force_elastic"], self.pp["force_plastic"], self.pp["force"], new_vg, self.interp["gwip"].transpose(), self.p["dt"], self.p["theta_c"], self.p["theta_s"]],
                  device=device)

        # Step 8: update particle velocities
        wp.launch(kernel=particle_updates.update_particle_velocity,
                  dim=self.num_p,
                  inputs=[self.pp["velocity"], new_vg, self.gp["velocity"], self.interp["wip"].transpose(), self.p["alpha"]],
                  device=device)

        # Step 9: particle-based body collisions
        particle_positions = self.pp["position"].numpy()
        particle_velocities = self.pp["velocity"].numpy()
        new_vp = handle_collisions.handle_all_collisions(particle_positions, particle_velocities, self.p["dt"], self.bodies)
        self.pp["velocity"] = wp.array(new_vp, dtype=wp.vec2, device=device)

        # Step 10: update particle positions
        wp.launch(kernel=particle_updates.update_particle_position,
                  dim=self.num_p,
                  inputs=[self.pp["position"], self.pp["velocity"], self.p["dt"]],
                  device=device)

        # Clean up
        self.frame += 1

    def animate(self) -> None:
        """
        Animate current environment.
        """
        if type(self.pp["position"]) == np.ndarray:
            return self.pp["position"]
        return self.pp["position"].numpy()
