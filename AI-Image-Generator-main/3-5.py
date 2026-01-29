import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"  # public, ungated, CPU-safe

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

pipe.enable_attention_slicing()
pipe = pipe.to("cpu")

prompt = "a programmer touching grass, realistic, detailed"

result = pipe(
    prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512
)

image = result.images[0]
image.save("image_0.png")

print("Saved image_0.png")
