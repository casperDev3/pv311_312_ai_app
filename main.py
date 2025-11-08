"""
Розпізнавання об'єктів на відео в реальному часі з виділенням рамкою
Використовуємо веб-камеру, YOLOv5 для детекції та ResNet50 для класифікації
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import requests
import time
import numpy as np

# Завантажуємо YOLOv5 для детекції об'єктів
print("🔄 Завантаження моделі YOLOv5...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

# Завантажуємо ResNet50 для класифікації
print("🔄 Завантаження моделі ResNet50...")
model = models.resnet50(pretrained=True)
model.eval()

# Переміщуємо моделі на GPU якщо доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
yolo_model = yolo_model.to(device)
print(f"✅ Моделі завантажені на: {device}")

# Завантажуємо мітки класів ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Підготовка трансформацій для кадрів відео
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_frame(frame):
    """
    Розпізнає об'єкт на одному кадрі відео

    Args:
        frame: кадр з відео (numpy array)

    Returns:
        top_class: назва класу з найвищою ймовірністю
        confidence: впевненість у відсотках
    """
    # Конвертуємо BGR (OpenCV) в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Застосовуємо трансформації
    img_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    # Робимо передбачення
    with torch.no_grad():
        outputs = model(img_tensor)

    # Отримуємо ймовірності
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Знаходимо клас з найвищою ймовірністю
    top_prob, top_idx = torch.max(probabilities, 0)

    top_class = labels[top_idx.item()]
    confidence = top_prob.item() * 100

    return top_class, confidence


def draw_text_with_background(frame, text, position, font_scale=0.8, thickness=2):
    """
    Малює текст з напівпрозорим фоном для кращої читабельності
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Отримуємо розмір тексту
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Малюємо напівпрозорий прямокутник
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - text_height - 15),
                  (x + text_width + 10, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Малюємо текст
    cv2.putText(frame, text, (x, y - 5), font, font_scale, (0, 255, 0), thickness)


def draw_text_with_background(frame, text, position, font_scale=0.8, thickness=2):
    """
    Малює текст з напівпрозорим фоном для кращої читабельності
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Отримуємо розмір тексту
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Малюємо напівпрозорий прямокутник
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - text_height - 15),
                  (x + text_width + 10, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Малюємо текст
    cv2.putText(frame, text, (x, y - 5), font, font_scale, (0, 255, 0), thickness)


def run_realtime_detection(camera_id=0, confidence_threshold=0.3):
    """
    Запускає розпізнавання об'єктів в реальному часі з виділенням рамками

    Args:
        camera_id: ID камери (зазвичай 0 для вбудованої веб-камери)
        confidence_threshold: мінімальна впевненість для відображення (0-1)
    """
    # Відкриваємо відео потік
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("❌ Помилка: Не вдалося відкрити камеру")
        return

    # Встановлюємо розмір кадру
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("🎥 РОЗПІЗНАВАННЯ ОБ'ЄКТІВ В РЕАЛЬНОМУ ЧАСІ")
    print("=" * 60)
    print("📌 Натисніть 'q' для виходу")
    print("📌 Натисніть 's' для збереження скріншоту")
    print("=" * 60 + "\n")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    screenshot_counter = 0

    # Кольори для різних класів об'єктів
    colors = [
        (0, 255, 0),  # Зелений
        (255, 0, 0),  # Синій
        (0, 0, 255),  # Червоний
        (255, 255, 0),  # Блакитний
        (255, 0, 255),  # Пурпурний
        (0, 255, 255),  # Жовтий
    ]

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Помилка читання кадру")
            break

        # Детекція об'єктів за допомогою YOLO
        try:
            results = yolo_model(frame)
            detections = results.pandas().xyxy[0]  # Отримуємо результати у форматі pandas

            # Обробляємо кожен виявлений об'єкт
            for idx, detection in detections.iterrows():
                confidence = detection['confidence']

                # Пропускаємо об'єкти з низькою впевненістю
                if confidence < confidence_threshold:
                    continue

                # Отримуємо координати рамки
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                    int(detection['xmax']), int(detection['ymax'])

                # Отримуємо назву класу
                class_name = detection['name']

                # Вибираємо колір для рамки
                color = colors[idx % len(colors)]

                # Малюємо рамку навколо об'єкта
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Підготовка тексту з назвою та впевненістю
                label = f"{class_name}: {confidence:.2f}"

                # Малюємо фон для тексту
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                # Фон для тексту
                cv2.rectangle(frame, (x1, y1 - text_height - 10),
                              (x1 + text_width + 5, y1), color, -1)

                # Текст
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        except Exception as e:
            print(f"Помилка детекції: {e}")

        # Обчислюємо FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Відображаємо FPS з фоном
        fps_text = f"FPS: {fps}"
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (5, frame.shape[0] - 30),
                      (15 + text_width, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Відображаємо інструкції
        cv2.putText(frame, "Natysni 'q' - vyhid, 's' - screenshot",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        # Показуємо кадр
        cv2.imshow('Rozpiznavannya ob\'yektiv z vydilennyam', frame)

        # Обробка клавіш
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n👋 Завершення роботи...")
            break
        elif key == ord('s'):
            screenshot_counter += 1
            filename = f"screenshot_{screenshot_counter}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Скріншот збережено: {filename}")

    # Звільняємо ресурси
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Робота завершена")


# ============================================
# ЗАПУСК ПРОГРАМИ
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  РОЗПІЗНАВАННЯ ОБ'ЄКТІВ З ВЕБ-КАМЕРИ З ВИДІЛЕННЯМ")
    print("=" * 60)

    try:
        # Запуск з веб-камерою
        run_realtime_detection(camera_id=4, confidence_threshold=0.3)

    except KeyboardInterrupt:
        print("\n\n👋 Програма перервана користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
