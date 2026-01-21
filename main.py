from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import DiffusionPipeline

app = FastAPI()

# lehčí model CogVideo (THUDM/CogVideoX1.5-5B-I2V)
model_id = "THUDM/CogVideoX1.5-5B-I2V"

# načtení modelu
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: Request):
    prompt = req.prompt

    # generování videa
    video = pipe(prompt=prompt).videos[0]

    # uložení
    out_path = "/tmp/output.mp4"
    video.save(out_path)

    return {"video_path": out_path}
