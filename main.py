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

# завантаження моделі YOLOv5
print("Завантаження моделі YOLOv5...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

# завантаження моделі ResNet50
print("Завантаження моделі ResNet50...")
model = models.resnet50(pretrained=True)
model.eval()

# Переміщуємо модель на GPU, якщо доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
yolo_model = yolo_model.to(device)
print('Використовується пристрій:', device)

# Завантаження міток ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL)

# Підготовка трансформацій для кадрів відео
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_frame(frame):
    # конвертуємо BGR в RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Застосування трансформацій
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Робимо передбачення
    with torch.no_grad():
        outputs = model(img_tensor)

    # Отримую ймовірності
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Отримуємо топ передбачення
    top_prob, top_idx = torch.max(probabilities, 0)
    # TODO: вивести формат top_idx

    top_class = labels[top_idx.item()]
    confidence = top_prob.item() * 100

    return top_class, confidence


def draw_text_with_background(frame, text, position, font_scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Отримуємо розміри тексту
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    # Малюємо фон (напівпрозорий прямокутник)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y + baseline), (0, 0, 0), -1)
    alpha = 0.6  # прозорість фону
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Малюємо текст поверх фону
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def run_realtime_detection(camera_id=0, confidence_threshold=0.3):
    # Відкриваємо відеопотік з камери
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Не вдалося відкрити камеру")
        return

    # Встановлюємо роздільну здатність відео
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Стартові налаштування для FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

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
            print("Не вдалося отримати кадр з камери")
            break

        # Детекція об'єктів за допомогою YOLOv5
        try:
            results = yolo_model(frame)
            print(results)
            detections = results.pandas().xyxy[0]  # отримуємо детекції у форматі pandas DataFrame
            print(detections)
            # for idx, detection in detections.iterrows():
            #     confidence = detection['confidence']
            #     if confidence < confidence_threshold:
            #         continue
            #     # Координати  рамки
            #     x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(
            #         detection['ymax'])
            #     # Клас об'єкта
            #     class_name = detection['name']
            #     color = colors[idx % len(colors)]
            #
            #     # Малюємо рамку навколо об'єкта
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #     label = f"{class_name} {confidence:.2f}"
            #     (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            #     cv2.rectangle(
            #         frame,
            #         (x1, y1 - text_height - baseline),
            #         (x1 + text_width, y1),
            #         color,
            #         -1
            #     )
            #     cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            break
        except Exception as e:
            print("Помилка під час детекції об'єктів:", e)
            continue

        #  Оновлення FPS
        fps_counter += 1
        if (time.time() - fps_start_time) >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Відображення FPS накадрі
        draw_text_with_background(frame, f"FPS: {fps}", (10, 30))
        # Відображення кадру
        cv2.imshow("Detect in realtime", frame)

        # Вихід при натисканні клавіші 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Звільнення ресурсів
    cap.release()
    cv2.destroyAllWindows()
    print("Завершено роботу. Ресурси звільнено.")

def main():
    try:
        run_realtime_detection(camera_id=3, confidence_threshold=0.2)
    except KeyboardInterrupt:
        print("Програма завершена користувачем.")
    except Exception as e:
        print("Виникла помилка:", e)


if __name__ == "__main__":
    main()
