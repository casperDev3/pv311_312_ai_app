import torch
from diffusers import DiffusionPipeline
import os
import time
from PIL import Image
import numpy as np

# Використовуємо модель, яка підтримує safetensors
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
MODEL_CACHE_DIR = "./models/video-diffusion"


def download_model():
    print("=" * 20)
    print("Downloading Video Diffusion model...")

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    try:
        # Завантаження моделі
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
        return False


def load_model():
    print("=" * 20)
    print("Loading Video Diffusion model from cache...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        pipe = DiffusionPipeline.from_pretrained(
            os.path.join(MODEL_CACHE_DIR, "saved_model"),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            low_cpu_mem_usage=True
        ).to(device)

        if device == "cuda":
            pipe.enable_attention_slicing()
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()

        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_frames_simple(result):
    """Проста функція для витягування кадрів з TextToVideoSDPipelineOutput"""
    try:
        print(f"Result type: {type(result)}")

        # Безпосередній доступ до frames атрибуту
        if hasattr(result, 'frames'):
            frames_array = result.frames
            print(f"Frames array type: {type(frames_array)}")
            print(f"Frames array shape: {frames_array.shape}")

            # frames_array має форму: (batch_size, num_frames, height, width, channels)
            # Зазвичай: (1, 16, 256, 256, 3) або подібну

            if len(frames_array.shape) == 5:
                # Видаляємо batch dimension і отримуємо (num_frames, height, width, channels)
                frames_array = frames_array[0]
                print(f"After removing batch dimension: {frames_array.shape}")

            # Конвертуємо кожен кадр в PIL Image
            frames = []
            for i in range(frames_array.shape[0]):
                frame = frames_array[i]

                # Переконуємося, що значення в правильному діапазоні [0, 255]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

                # Створюємо PIL Image
                pil_img = Image.fromarray(frame)
                frames.append(pil_img)

            print(f"Successfully converted {len(frames)} frames to PIL images")
            return frames
        else:
            print("No 'frames' attribute found in result")
            print(f"Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            return None

    except Exception as e:
        print(f"Error in extract_frames_simple: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_video(pipe, prompt, out_path=None, num_frames=16, steps=30, guidance=7.5, fps=8):
    print("=" * 20)
    print(f"Generating video for prompt: {prompt}")
    print(f"Frames: {num_frames}, FPS: {fps}")

    try:
        # Генерація відео
        result = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_frames=num_frames
        )

        # Отримання кадрів простою функцією
        video_frames = extract_frames_simple(result)

        if not video_frames:
            print("Failed to extract frames")
            return None

        # Створюємо папку для результатів
        os.makedirs("generated_videos", exist_ok=True)

        # Збереження результатів
        if out_path is None:
            timestamp = int(time.time())
            out_path = f"generated_videos/video_{timestamp}.gif"

        # Збереження як GIF
        if len(video_frames) > 0:
            # Перевіряємо розміри першого кадру
            print(f"First frame size: {video_frames[0].size}")
            print(f"First frame mode: {video_frames[0].mode}")

            video_frames[0].save(
                out_path,
                save_all=True,
                append_images=video_frames[1:],
                duration=1000 // fps,  # мілісекунди на кадр
                loop=0,
                optimize=True
            )
            print(f"✅ GIF video saved to {out_path}")
            print(f"✅ Generated {len(video_frames)} frames")

            return video_frames
        else:
            print("❌ Error: No frames in video_frames list")
            return None

    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_video_as_mp4(frames, out_path, fps=8):
    """Збереження як MP4"""
    try:
        import cv2

        if not isinstance(frames, list) or len(frames) == 0:
            print("No frames to save as MP4")
            return

        # Конвертація PIL images до numpy array
        frames_np = [np.array(frame) for frame in frames]

        # Отримання розмірів відео
        height, width = frames_np[0].shape[:2]
        print(f"Video dimensions: {width}x{height}")

        # Створення VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for frame in frames_np:
            # Конвертація RGB до BGR для OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"✅ MP4 video saved to {out_path}")

    except ImportError:
        print("❌ OpenCV not available. Install with: pip install opencv-python")
    except Exception as e:
        print(f"❌ Error saving MP4: {e}")


def debug_model_output(pipe, prompt):
    """Функція для дебагінгу виводу моделі"""
    print("=" * 20)
    print("DEBUG: Testing model output...")

    try:
        # Генерація одного кадру для тесту
        result = pipe(
            prompt,
            num_inference_steps=5,  # Мінімальна кількість для тесту
            num_frames=4
        )

        print("DEBUG: Result analysis:")
        print(f"  Type: {type(result)}")
        print(f"  Attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

        if hasattr(result, 'frames'):
            frames = result.frames
            print(f"  Frames type: {type(frames)}")
            print(f"  Frames shape: {frames.shape}")
            print(f"  Frames dtype: {frames.dtype}")
            print(f"  Frames range: [{frames.min()}, {frames.max()}]")

            # Спробуємо зберегти перший кадр як зображення для перевірки
            if len(frames.shape) >= 4:
                test_frame = frames[0, 0] if len(frames.shape) == 5 else frames[0]
                if test_frame.max() <= 1.0:
                    test_frame = (test_frame * 255).astype(np.uint8)
                test_img = Image.fromarray(test_frame)
                test_img.save("debug_test_frame.jpg")
                print("  ✅ Debug frame saved as debug_test_frame.jpg")

        return True
    except Exception as e:
        print(f"DEBUG Error: {e}")
        return False


def main():
    prompt = "A astronaut riding a horse on mars"

    try:
        pipe = load_model()

        if pipe is None:
            print("❌ Failed to load model")
            return

        # Спочатку запустимо дебаг
        debug_model_output(pipe, prompt)

        print("=" * 20)
        print("Starting video generation...")

        # Генерація відео
        frames = generate_video(
            pipe,
            prompt,
            num_frames=8,  # Менше кадрів для швидшої генерації
            steps=200,  # Менше кроків
            guidance=7.5,
            fps=4
        )

        if frames:
            # Додаткове збереження як MP4
            timestamp = int(time.time())
            mp4_path = f"generated_videos/video_{timestamp}.mp4"
            save_video_as_mp4(frames, mp4_path, fps=4)
            print("🎉 Video generation completed successfully!")
        else:
            print("❌ Video generation failed - no frames produced")

    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Перевіряємо, чи модель вже завантажена
    if os.path.exists(os.path.join(MODEL_CACHE_DIR, "saved_model")):
        print("Model found in cache, skipping download...")
        main()
    else:
        print("Downloading model...")
        if download_model():
            print("Model downloaded successfully!")
            main()
        else:
            print("Failed to download model")