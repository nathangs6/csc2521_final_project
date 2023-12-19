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
import time

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
        self.p = {
            "dt": scene.dt,
            "h": scene.spacing,
            "alpha": scene.alpha,
            "tc": scene.theta_c,
            "ts": scene.theta_s,
            "zeta": float(scene.hardening_coefficient),
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
            "F": np.zeros(shape=(self.num_p,2,2), dtype=np.float32),
            "J": np.ones(shape=self.num_p, dtype=np.float32),
            "FE": np.zeros(shape=(self.num_p,2,2), dtype=np.float32),
            "JE": np.ones(shape=self.num_p, dtype=np.float32),
            "FP": np.zeros(shape=(self.num_p,2,2), dtype=np.float32),
            "JP": np.ones(shape=self.num_p, dtype=np.float32)
        }
        for i in range(self.num_p):
            self.pp["F"][i] = np.eye(2, dtype=float)
            self.pp["FE"][i] = np.eye(2, dtype=float)
            self.pp["FP"][i] = np.eye(2, dtype=float)
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
        self.pp["F"] = wp.array(self.pp["F"], dtype=wp.mat22, device=device)
        self.pp["FE"] = wp.array(self.pp["FE"], dtype=wp.mat22, device=device)
        self.pp["JE"] = wp.array(self.pp["JE"], dtype=wp.float32, device=device)
        self.pp["FP"] = wp.array(self.pp["FP"], dtype=wp.mat22, device=device)
        self.pp["JP"] = wp.array(self.pp["JP"], dtype=wp.float32, device=device)
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
        start = time.time()
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
        print("Step 1 took " + str(time.time() - start) + " to run")

        # Step 2: compute particle volumes and densities
        if self.frame == 0:
            start = time.time()
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
            print("Step 2 took " + str(time.time() - start) + " to run")

        # Step 3: compute grid forces
        start = time.time()
        ## Compute stresses
        stresses = wp.zeros_like(self.pp["F"], device=device)
        wp.launch(kernel=grid_updates.get_stresses,
                  dim=self.num_p,
                  inputs=[stresses, self.pp["FE"], self.pp["JE"], self.pp["FP"], self.pp["JP"], self.p["mu0"], self.p["lam0"], self.p["zeta"]],
                  device=device)
        ## Compute grid forces
        grid_forces = wp.zeros_like(self.gp["velocity"], device=device)
        wp.launch(kernel=grid_updates.compute_grid_forces,
                  dim=self.num_g,
                  inputs=[grid_forces, self.pp["volume"], self.interp["gwip"], stresses, self.gp["mass"]],
                  device=device)
        wp.launch(kernel=grid_updates.add_force,
                  dim=self.num_g,
                  inputs=[grid_forces, wp.vec2(0.0,-9.81)],
                  device=device)
        print("Step 3 took " + str(time.time() - start) + " to run")

        # Step 4: update velocities on grid
        start = time.time()
        vg_temp = wp.zeros_like(self.gp["velocity"])
        #grid_forces = wp.array(grid_forces, dtype=wp.vec2, device=device)
        wp.launch(kernel=grid_updates.update_grid_velocities_with_ext_forces,
                  dim=self.num_g,
                  inputs=[vg_temp, self.gp["velocity"], self.gp["mass"], grid_forces, self.p["dt"]],
                  device=device)
        print("Step 4 took " + str(time.time() - start) + " to run")

        # Step 5: handle grid-based body collisions
        start = time.time()
        ## Put necessary resources onto CPU
        grid_coords = self.p["h"] * self.gp["position"].numpy()
        grid_masses = self.gp["mass"].numpy()
        velocities = vg_temp.numpy()
        velocities = handle_collisions.handle_all_collisions(grid_coords, velocities,  self.p["dt"], self.bodies, grid_masses)
        print("Step 5 took " + str(time.time() - start) + " to run")

        # Step 6: solve the linear system
        start = time.time()
        if not implicit:
            vg_temp = wp.array(velocities, dtype=wp.vec2, device=device)
            new_vg = wp.zeros_like(vg_temp, device=device)
            wp.launch(kernel=grid_updates.solve_grid_velocity_explicit,
                      dim=self.num_g,
                      inputs=[new_vg, vg_temp],
                      device=device)
        else:
            velocities = grid_updates.solve_grid_velocity_implicit(velocities,
                                                      self.gp["mass"].numpy(),
                                                      self.interp["gwip"].numpy(),
                                                      self.pp["FE"].numpy(),
                                                      self.pp["FP"].numpy(),
                                                      self.pp["volume"].numpy(),
                                                      self.p["dt"],
                                                      1.0,
                                                      self.p["mu0"],
                                                      self.p["lam0"],
                                                      self.p["zeta"])
            new_vg = wp.array(velocities, dtype=wp.vec2, device=device)
        print("Step 6 took " + str(time.time() - start) + " to run")

        # Step 7: update deformation gradient
        start = time.time()
        ## Update matrices
        wp.launch(kernel=particle_updates.update_particle_F,
                  dim=self.num_p,
                  inputs=[self.pp["F"], new_vg, self.interp["gwip"].transpose(), self.p["dt"]],
                  device=device)
        wp.launch(kernel=particle_updates.update_particle_FE_FP,
                  dim=self.num_p,
                  inputs=[self.pp["FE"], self.pp["FP"], self.pp["F"], new_vg, self.interp["gwip"].transpose(), self.p["dt"], self.p["tc"], self.p["ts"]],
                  device=device)
        ## Update determinants
        wp.launch(kernel=particle_updates.array_determinant,
                  dim=self.num_p,
                  inputs=[self.pp["JE"], self.pp["FE"]],
                  device=device)
        wp.launch(kernel=particle_updates.array_determinant,
                  dim=self.num_p,
                  inputs=[self.pp["JP"], self.pp["FP"]],
                  device=device)
        print("Step 7 took " + str(time.time() - start) + " to run")

        # Step 8: update particle velocities
        start = time.time()
        wp.launch(kernel=particle_updates.update_particle_velocity,
                  dim=self.num_p,
                  inputs=[self.pp["velocity"], new_vg, self.gp["velocity"], self.interp["wip"].transpose(), self.p["alpha"]],
                  device=device)
        print("Step 8 took " + str(time.time() - start) + " to run")

        # Step 9: particle-based body collisions
        start = time.time()
        particle_positions = self.pp["position"].numpy()
        particle_velocities = self.pp["velocity"].numpy()
        new_vp = handle_collisions.handle_all_collisions(particle_positions, particle_velocities, self.p["dt"], self.bodies)
        self.pp["velocity"] = wp.array(new_vp, dtype=wp.vec2, device=device)
        print("Step 9 took " + str(time.time() - start) + " to run")

        # Step 10: update particle positions
        start = time.time()
        wp.launch(kernel=particle_updates.update_particle_position,
                  dim=self.num_p,
                  inputs=[self.pp["position"], self.pp["velocity"], self.p["dt"]],
                  device=device)
        print("Step 10 took " + str(time.time() - start) + " to run")

        # Clean up
        self.frame += 1
