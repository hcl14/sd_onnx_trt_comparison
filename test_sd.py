from diffusers import StableDiffusionPipeline

from diffusers import DPMSolverMultistepScheduler

from compel import Compel
import torch
import time
import numpy as np

'''
pip install diffusers torch omegaconf compel onnx onnxruntime-gpu accelerate

'''


model_id = "checkpoints/realisticVisionV60B1_v60B1VAE.safetensors"


"""
Euler A or DPM++ SDE Karras

CFG Scale 3,5 - 7

Hires. fix with 4x-UltraSharp upscaler

Denoising strength 0.25-0.45

Upscale by 1.1-2.0

Clip Skip 1-2

ENSD 31337
"""

def safe_check(images, **kwargs):
    return images, [False] * len(images)


# https://huggingface.co/stabilityai/stable-diffusion-2-1/discussions/23
pipe = StableDiffusionPipeline.from_single_file(model_id, dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

#pipe.safety_checker.forward = safe_check #lambda images, **kwargs: (images, [False] * len(images))
pipe.safety_checker = None #lambda images, **kwargs: (images, [False] * len(images))


prompt = "closeup portrait photo of beautiful 24 y.o goth woman, makeup, 8k uhd, high quality, dramatic, cinematic"

'''
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream"
'''
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

times = []
# Avg prompt encoding time: 0.1272261142730713
for i in range(5):
    start = time.time()

    prompt_embed = compel_proc(prompt)
    neg_prompt_embed = compel_proc(prompt)
    times.append(time.time() - start)


print("Avg prompt encoding time:", np.mean(times[2:]))

times = []

for i in range(5):

    generator = torch.Generator(device="cuda").manual_seed(2325031155)
    start = time.time()

    image = pipe(prompt_embeds=prompt_embed, negative_prompt_embeds=neg_prompt_embed, height=640, width=512, num_inference_steps = 25, guidance_scale = 7, generator = generator).images[0]

    times.append(time.time() - start)

print("Avg generation time:", np.mean(times[2:]))


image.save("test1.png")

# Without Compel:
# Avg generation time:  4.801786740620931


