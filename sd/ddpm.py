import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator:torch.Generator, num_training_steps:int=1000,
                 beta_start:float=0.00085, beta_end:float=0.0120
                 ):
        
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, self.num_train_steps)[::-1].copy())

        


    
    