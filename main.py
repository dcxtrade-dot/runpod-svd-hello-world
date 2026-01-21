from fastapi import FastAPI
from pydantic import BaseModel
import torch

# Import procesu, když je pipeline dostupná
try:
    from diffusers import WanPipeline
except ImportError:
    # fallback error
    WanPipeline = None

app = FastAPI()

# WAN 2.1
model_id = "wangkanai/wan21-fp8-i2v-gguf"  # GGUF kvantizovaná verze
pipe = None

@app.on_event("startup")
async def load_model():
    global pipe
    if WanPipeline is None:
        raise RuntimeError("WanPipeline not found; make sure diffusers>=0.24.1 is installed")

    # načtení modelu
    # (pokud je v GGUF formátu, diffusers může vyžadovat WanPipeline.from_pretrained)
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: Request):
    result = pipe(prompt=req.prompt)
    # Výstup může mít jinou strukturu než standardní .videos
    # proto to uložíme přímo
    out_path = "/tmp/output.mp4"
    result.save_video(out_path)
    return {"video_path": out_path}
