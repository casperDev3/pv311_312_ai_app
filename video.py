import torch
from diffusers import DiffusionPipeline
import os
import time
from PIL import Image
import numpy as np

# Використовуємо модель, яка підтримує safetensors
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"  # або іншу сумісну модель
MODEL_CACHE_DIR = "./models/video-diffusion"


def download_model():
    print("=" * 20)
    print("Downloading Video Diffusion model...")

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    try:
        # Завантаження моделі без специфікації variant
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )

        save_path = os.path.join(MODEL_CACHE_DIR, "saved_model")
        pipe.save_pretrained(save_path)

        print(f"Model downloaded and saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Trying without safetensors...")
        return download_model_without_safetensors()


def download_model_without_safetensors():
    """Альтернативна функція завантаження без вимоги safetensors"""
    try:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            use_safetensors=False,  # Дозволяємо використання .bin файлів
            low_cpu_mem_usage=True
        )

        save_path = os.path.join(MODEL_CACHE_DIR, "saved_model")
        pipe.save_pretrained(save_path)

        print(f"Model downloaded and saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model without safetensors: {e}")
        return False


def load_model():
    print("=" * 20)
    print("Loading Video Diffusion model from cache...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cache_exists = os.path.exists(os.path.join(MODEL_CACHE_DIR, "saved_model"))

    if cache_exists:
        print('Loading model from cache...')
        # Завантаження з локального кешу
        pipe = DiffusionPipeline.from_pretrained(
            os.path.join(MODEL_CACHE_DIR, "saved_model"),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            low_cpu_mem_usage=True
        ).to(device)
    else:
        print("Model cache not found, downloading...")
        # Завантаження з Hugging Face
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            use_safetensors=True,
            low_cpu_mem_usage=True
        ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        # Для відео часто потрібне додаткове оптимізації пам'яті
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()

    return pipe


def generate_video(pipe, prompt, out_path=None, num_frames=16, steps=30, guidance=7.5, fps=8):
    print("=" * 20)
    print(f"Generating video for prompt: {prompt}")
    print(f"Frames: {num_frames}, FPS: {fps}")

    # Генерація відео
    result = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_frames=num_frames
    )

    # Отримання кадрів (залежить від структури результату)
    if hasattr(result, 'frames'):
        video_frames = result.frames[0]
    elif hasattr(result, 'images'):
        video_frames = result.images
    else:
        video_frames = result[0]  # Спробуємо перший елемент

    # Збереження результатів
    if out_path is None:
        timestamp = int(time.time())
        out_path = f"generated_videos/video_{timestamp}.gif"

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Збереження як GIF
    if isinstance(video_frames, list) and len(video_frames) > 0:
        video_frames[0].save(
            out_path,
            save_all=True,
            append_images=video_frames[1:],
            duration=1000 // fps,
            loop=0
        )
        print(f"GIF video saved to {out_path}")
    else:
        print("Error: No frames generated")

    return video_frames


def save_video_as_mp4(frames, out_path, fps=8):
    """Альтернативний спосіб збереження як MP4"""
    try:
        import cv2

        if not isinstance(frames, list) or len(frames) == 0:
            print("No frames to save as MP4")
            return

        # Конвертація PIL images до numpy array
        frames_np = [np.array(frame) for frame in frames]

        # Отримання розмірів відео
        height, width = frames_np[0].shape[:2]

        # Створення VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for frame in frames_np:
            # Конвертація RGB до BGR для OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"MP4 video saved to {out_path}")

    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
    except Exception as e:
        print(f"Error saving MP4: {e}")


def main():
    prompt = "A astronaut riding a horse on mars"

    try:
        pipe = load_model()

        # Генерація відео
        frames = generate_video(
            pipe,
            prompt,
            num_frames=16,  # Менше кадрів для швидшої генерації
            steps=20,
            guidance=7.5,
            fps=8
        )

        # Додаткове збереження як MP4
        timestamp = int(time.time())
        mp4_path = f"generated_videos/video_{timestamp}.mp4"
        save_video_as_mp4(frames, mp4_path, fps=8)

    except Exception as e:
        print(f"Video generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Спершу завантажуємо модель
    if download_model():
        print("Model downloaded successfully!")
        # Потім генеруємо відео
        main()
    else:
        print("Failed to download model")