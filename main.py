from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import DiffusionPipeline

app = FastAPI()

model_id = "damo-vilab/text-to-video-ms-1.0"

# načtení modelu při startu aplikace
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: Request):
    prompt = req.prompt

    video = pipe(prompt=prompt).videos[0]

    out_path = "/tmp/output.mp4"
    video.save(out_path)

    return {"video_path": out_path}
