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


def main():
    print("Hello, World!")


if __name__ == "__main__":
    main()
