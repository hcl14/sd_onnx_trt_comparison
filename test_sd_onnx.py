from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline
import torch

from diffusers import DPMSolverMultistepScheduler

from compel import Compel

import time
import numpy as np


'''
# conda create -n onnx_env -c conda-forge cudatoolkit=11.4
#pip install --force-reinstall nvidia-cudnn-cu11==8.5.0.96 torch onnxruntime-gpu omegaconf compel onnx accelerate
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/


TRT (does not work):
pip install tensorrt==8.6.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/anaconda3/envs/sd_env/lib/python3.11/site-packages/tensorrt_libs/
'''


model_path = "onnx"


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
pipe = OnnxStableDiffusionPipeline.from_pretrained(model_path, provider="CUDAExecutionProvider") #'TensorrtExecutionProvider'
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#pipe = pipe.to("cuda")

#pipe.safety_checker.forward = safe_check #lambda images, **kwargs: (images, [False] * len(images))
pipe.safety_checker = None #lambda images, **kwargs: (images, [False] * len(images))


prompt = "closeup portrait photo of beautiful 24 y.o goth woman, makeup, 8k uhd, high quality, dramatic, cinematic"

'''
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream"
'''
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
'''
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

times = []


for i in range(5):
    start = time.time()

    prompt_embed = compel_proc(prompt)
    neg_prompt_embed = compel_proc(prompt)
    times.append(time.time() - start)


print("Avg prompt encoding time:", np.mean(times[2:]))
'''

times = []

for i in range(5):

    generator = np.random.RandomState(2325031155) # does not work torch.Generator(device="cpu").manual_seed(2325031155)
    start = time.time()

    image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=640, width=512, num_inference_steps = 25, guidance_scale = 7, generator = generator).images[0]

    times.append(time.time() - start)

print("Avg generation time:", np.mean(times[2:]))


image.save("test2.png") # is different

# Without Compel:
# Avg generation time: 3.8234673341115317



