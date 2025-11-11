import torch
from diffusers import StableDiffusionPipeline
import os
import time


MODEL_CACHE_DIR = "./models/stable-diffusion"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

def download_model():
    print("=" * 20)
    print("Downloading Stable Diffusion model...")

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    try:
        # Завантаження моделі з кешуванням без GPU
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16,
            # revision="fp16", # fp32 для GPU з меншою пам'яттю, fp16 для сучасних СPU
            safety_checker=None,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            variant="fp16"
        )

        save_path = os.path.join(MODEL_CACHE_DIR, "saved_model")
        pipe.save_pretrained(save_path)

        print(f"Model downloaded and saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def load_model():
    print("=" * 20)
    print("Loading Stable Diffusion model from cache...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if os.path.exists(MODEL_CACHE_DIR):
        print('Loading model from cache...')
        local = True
    else:
        print("Model cache not found.")
        local = False

    pipe =  StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR if local else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        local_files_only=local,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        variant="fp16" if device == "cuda" else "fp16"
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe

def generate_image(pipe, prompt, out_paths=f"generated_images/", file_name=f"british_gen_{int(time.time())}", file_type=".png", steps=30, guidance= 7.5):
    print("=" * 20)
    print(f"Generating image for prompt: {prompt}")

    # os.makedirs(os.path.dirname("./generated_images"), exist_ok=True)

    image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0] #TODO: check same temp in JS
    image.save(f"{out_paths}{file_name}_{steps}{file_type}")

    print(f"Image saved to: {out_paths}{file_name}_{steps}{file_type}")
    return image


def main():
    prompt = """
    british short hair cat sleeping in sunlight on wooden floor, warm tone, peaceful cozy atmosphere, cinematic detail,32k
    """
    try:
        pipe = load_model()
        generate_image(pipe, prompt, steps=55, guidance=7.5)
    except Exception  as e:
        print(f"Model loading failed: {e}")


if __name__ == "__main__":
    # download_model()
    main()
