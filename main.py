from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import StableVideoDiffusionPipeline

app = FastAPI()

model_id = "stabilityai/stable-video-diffusion-1"

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: Request):
    prompt = req.prompt

    video = pipe(
        prompt=prompt,
        num_inference_steps=25,
        height=512,
        width=512,
        num_frames=16
    ).videos[0]

    out_path = "/tmp/output.mp4"
    video.save(out_path)

    return {"video_path": out_path}
