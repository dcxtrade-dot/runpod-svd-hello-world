from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import DiffusionPipeline

app = FastAPI()
pipe = None   # ⚠️ NIC nenačítat při importu

class Request(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup():
    global pipe
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.0",
        torch_dtype=torch.float16
    )
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")

@app.post("/generate")
async def generate(req: Request):
    video = pipe(prompt=req.prompt).videos[0]
    path = "/tmp/out.mp4"
    video.save(path)
    return {"path": path}
