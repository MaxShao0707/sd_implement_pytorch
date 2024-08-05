import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(prompt,
             uncond_prompt=None,
             input_image=None,
             strength=0.8,
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):
    
    with torch.no_grad():
        if not 0 < strength < 1:
            raise ValueError("strength must between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).inputs_ids

            #batch_size, seq_len
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).inputs_ids

            #batch_size, seq_len 
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            
            #batch_size, seq_len -> batch_size, seq_len, dim
            uncond_context = clip(uncond_context)

            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).inputs_ids

            #batch, seq_len
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            #batch, seq_len -> batch, seq_len, dim
            context = clip(tokens)

        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("unknown sampler!")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((LATENT_HEIGHT, LATENT_WIDTH))

            #(height, width, channels)
            input_image_tensor = np.array(input_image_tensor)

            #height, width, channels -> height, width, channels
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

            #height, width, channels -> height, width, channels
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1, 1))

            #height, width, channels -> batch, height, width, chhannels
            input_image_tensor = input_image_tensor.unsqueeze(0)

            #batch, height, width, channels -> batch, channels, height, width
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            #batch, 4, latent_height, latent_width
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            #batch, 4, latent_height, latent_width
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            #batch_size, 4, latent_height, latent_width
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):

            #(1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            #(batch, 4, latent_height, latent_width)
            model_inputs = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_inputs = model_inputs.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_inputs, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)

                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1,), (0, 255), clamp=True)

        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        mages = images.to("cpu", torch.uint8).numpy()

        return images[0]
    

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    #(160,)
    freqs = torch.pow(10000, -torch.range(start=0, end=160, dtype=torch.float32) / 160)

    #shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    #shape (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)





























        
    


