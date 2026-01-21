import runpod

def handler(event):
    return {
        "message": "Hello World from RunPod!",
        "input": event.get("input", {})
    }

runpod.serverless.start({"handler": handler})
