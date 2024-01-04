import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import os
from tqdm import tqdm
import torch 

def mixture_of_gaussians_3D(x, means, covs, weights):
    D = x.shape[1]  # Number of dimensions
    density = torch.zeros(x.shape[0])
    for mean, cov, weight in zip(means, covs, weights):
        diff = x - mean
        exponent = -0.5 * torch.sum(torch.matmul(diff, torch.inverse(cov)) * diff, dim=1)
        normalizer = torch.sqrt(torch.pow(torch.tensor(2 * np.pi), D) * torch.det(cov))
        density += weight * torch.exp(exponent) / normalizer
    return density.numpy()

class ParticleSystem:
    def __init__(self, name, n_particles=None, dim=None, initial_particles=None):
        self.n_particles = n_particles if initial_particles is None else initial_particles.size(0)
        self.dim = dim
        self.particle_states = []
        self.name = name
        if initial_particles is not None:
            self.particles = initial_particles.clone()
        else:
            self.particles = torch.rand((self.n_particles, self.dim))
        self.record_state()

    def record_state(self):
        self.particle_states.append(self.particles.clone())
            
    def animate(self, fig=None, ax=None, trail_length=5):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        blue_dots = None  
        total_frames = len(self.particle_states)-1
        pbar = tqdm(total=total_frames) 
        
        # Label axes
        ax.set_xlabel('x')
        ax.set_ylabel('Concentration')

        trails = []  # List to hold trail plot objects

        def update_plot(i):
            nonlocal blue_dots  
            nonlocal trails 

            pbar.update(1)  
            if blue_dots is not None:
                blue_dots.remove()

            # Remove the oldest trail if the length exceeds 5
            if len(trails) > 5:
                for trail in trails[0]:
                    trail.remove()
                del trails[0]

            current_state = self.particle_states[i].detach().cpu().numpy()
            blue_dots = ax.scatter(current_state[:, 0], current_state[:, 1], c='blue')

            current_trails = []
            for particle_idx in range(self.n_particles):
                x_coords = [self.particle_states[j].detach().cpu().numpy()[particle_idx, 0] for j in range(max(0, i - trail_length), i + 1)]
                y_coords = [self.particle_states[j].detach().cpu().numpy()[particle_idx, 1] for j in range(max(0, i - trail_length), i + 1)]
                trail, = ax.plot(x_coords, y_coords, c='black', alpha=.1)  # The comma is to unpack the returned list into a single object
                current_trails.append(trail)
            
            x_position = 0.95 - (len(str(i)) - 1) * 0.02
            for text_annotation in ax.texts:
                text_annotation.remove()
            # Add the current epoch number to the top right corner
            ax.text(x_position, 0.95, f'Epoch: {i}', transform=ax.transAxes, fontsize=12, ha='right', va='top')
            trails.append(current_trails)

        ani = animation.FuncAnimation(fig, update_plot, frames=total_frames, interval=200)
        unique_id = datetime.datetime.now().strftime('%m-%d-%Y_%H:%M')
        directory = "./gifs/" + str(self.name) + "/"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, str(self.name) + '_animation_' + unique_id + '.gif')
        ani.save(path, writer='Pillow', dpi=400)
        pbar.close()
        print("Gif Created with Trails")
    
    def animate_3d(self, logp, locs_3d, cov_mats_3d, probs_3d, fig=None, ax=None, trail_length=5):
        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        if logp is None or locs_3d is None or cov_mats_3d is None or probs_3d is None:
            raise ValueError("One or more Gaussian Mixture Model parameters are None")

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Concentration')

        blue_dots = None
        total_frames = len(self.particle_states) - 1
        pbar = tqdm(total=total_frames)
        trails = []

        def update_plot(i):
            nonlocal blue_dots, trails

            pbar.update(1)
            if blue_dots is not None:
                blue_dots.remove()

            if len(trails) > trail_length:
                for trail in trails[0]:
                    trail.remove()
                del trails[0]
            
            current_state = self.particle_states[i].detach().cpu().numpy()
            #density_current = torch.exp(logp(torch.tensor(current_state, dtype=torch.float32), locs_3d, cov_mats_3d, probs_3d)).detach()
            blue_dots = ax.scatter(current_state[:, 0], current_state[:, 1], current_state[:, 2], c='blue', alpha=0.2)

            current_trails = []
            for particle_idx in range(current_state.shape[0]):
                x_coords, y_coords, z_coords = [], [], []
                for j in range(max(0, i - trail_length), i + 1):
                    past_state = self.particle_states[j].detach().cpu().numpy()[particle_idx]
                    density_past = torch.exp(logp(torch.tensor(past_state.reshape(1, -1), dtype=torch.float32), locs_3d, cov_mats_3d, probs_3d)).detach().cpu().numpy()
                    
                    x_coords.append(past_state[0])
                    y_coords.append(past_state[1])
                    z_coords.append(density_past[0])  # Assuming density_past is a 1D array

                trail, = ax.plot(x_coords, y_coords, z_coords, c='black', alpha=0.9)
                current_trails.append(trail)

            trails.append(current_trails)

            for text_annotation in ax.texts:
                text_annotation.remove()
            ax.text(0.9, 0.9, 1.0, f'Epoch: {i}', transform=ax.transAxes, fontsize=12, ha='right', va='top')

        ani = animation.FuncAnimation(fig, update_plot, frames=total_frames, interval=200)
        unique_id = datetime.datetime.now().strftime('%m-%d-%Y_%H:%M')
        directory = "./gifs/" + str(self.name) + "/"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, str(self.name) + '_animation_3d_' + unique_id + '.gif')
        ani.save(path, writer='Pillow', dpi=400)
        pbar.close()
        print("Gif Created with Trails")