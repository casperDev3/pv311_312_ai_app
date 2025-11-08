import cv2
import numpy as np
import time
import os
from pathlib import Path
import face_recognition
import traceback

from face_recognition import face_encodings, face_distance
from ultralytics import YOLO
import threading

class FrameGrabber(threading.Thread):
    def __init__(self, src=0, width=1920, height=1080):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        if not self.capture.isOpened():
            raise ValueError("Не вдалось відкрити камеру з ID:", src)

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                print("Не вдалося отримати кадр з камери.")
                continue

    def read(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        return frame

    def stop(self):
        self.running = False
        self.capture.release()

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.people_boxes = []

        print("Завантаження моделі YOLOv8n...")
        self.person_detector = YOLO("yolov8n.pt")
        self.person_detector.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

        self.frame_count = 0
        self.lock = threading.Lock()

    def load_known_faces(self, photo_folder="photos"):
        print("Завантаження відомих облич...")
        if not os.path.exists(photo_folder):
            print(f"Папка '{photo_folder}' не існує.")
            return False

        person_folders = [f for f in Path(photo_folder).iterdir() if f.is_dir()]
        if not person_folders:
            print(f"Папка '{photo_folder}' порожня.")
            return False

        total_photos = 0
        for person_folder in person_folders:
            person_name = person_folder.name
            photo_files = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.jpeg")) + list(person_folder.glob("*.png"))
            for photo_path in photo_files:
                try:
                    image = face_recognition.load_image_file(photo_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(person_name)
                        total_photos += 1
                    else:
                        print(f"Обличчя не знайдено на фото: {photo_path}")
                except Exception as e:
                    print(f"Помилка при обробці фото {photo_path}: {e}")

        print(f"Завантажено {total_photos} фото з {len(person_folders)} осіб.")
        return total_photos > 0

    def process_frame(self, frame, face_interval=3, scale_factor=0.25):
        people_boxes = []

        # Виявлення людей за допомогою YOLOv8n на зменшеному кадрі
        small_for_yolo = cv2.resize(frame, (640, 360))
        results = self.person_detector.predict(small_for_yolo, classes=[0], conf=0.5, verbose=False) # Клас 0 - людина (person)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_ratio =  frame.shape[0] / 360
                w_ratio =  frame.shape[1] / 640
                people_boxes.append((int(x1 * w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)))

        # Розпізнавання облич на кожному n-му кадрі
        if self.frame_count % face_interval == 0:
            small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 0.0
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                face_names.append((name, confidence))

            face_locations = [
                (int(t / scale_factor), int(r / scale_factor), int(b / scale_factor), int(l / scale_factor))
                for (t, r, b, l) in face_locations
            ]

            with self.lock:
                self.face_locations = face_locations
                self.face_encodings = face_encodings

        with self.lock:
            self.people_boxes = people_boxes

        self.frame_count +=  1

def main():
    print("Hello, World!")
    FaceRecognitionSystem().load_known_faces()

if __name__ == "__main__":
    main()