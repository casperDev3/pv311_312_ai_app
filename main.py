import cv2
import numpy as np
import time
import os
from pathlib import Path
import face_recognition
import traceback
from ultralytics import YOLO
import threading


class FrameGrabber(threading.Thread):
    """Потік для асинхронного зчитування кадрів з камери"""
    def __init__(self, src=0, width=1920, height=1080):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        if not self.capture.isOpened():
            raise RuntimeError("Не вдалося відкрити камеру!")

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        """Повертає останній доступний кадр"""
        with self.lock:
            frame_copy = self.latest_frame.copy() if self.latest_frame is not None else None
        return frame_copy

    def stop(self):
        self.running = False
        self.capture.release()


class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.people_boxes = []
        self.phone_boxes = []

        print("🧠 Завантаження моделі YOLOv8n для детекції людей...")
        self.person_detector = YOLO("yolov8n.pt")
        self.person_detector.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

        self.frame_count = 0
        self.lock = threading.Lock()

    def load_known_faces(self, photo_folder="photos"):
        print("Завантаження фотографій для навчання...")
        if not os.path.exists(photo_folder):
            print(f"!!! Папка не існує {photo_folder}")
            return False

        person_folders = [f for f in Path(photo_folder).iterdir() if f.is_dir()]
        if not person_folders:
            print(f"Немає підпапок з працівниками у {photo_folder}")
            return False

        total_photos = 0
        for person_folder in person_folders:
            person_name = person_folder.name
            photo_files = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.jpeg")) + list(person_folder.glob("*.png"))
            for photo_path in photo_files:
                try:
                    image = face_recognition.load_image_file(str(photo_path))
                    enc = face_recognition.face_encodings(image)
                    if len(enc) == 0:
                        continue
                    self.known_faces_encodings.append(enc[0])
                    self.known_faces_names.append(person_name)
                    total_photos += 1
                except Exception as err:
                    print("Помилка при обробці:", err)
        print(f"✅ Завантажено {total_photos} фото облич")
        return total_photos > 0

    def process_frame(self, frame, face_interval=1, scale_factor=0.25):
        """Фонова обробка кадру — виконується в окремому потоці"""
        people_boxes = []
        phone_boxes = []

        # YOLO на зменшеному кадрі
        small_for_yolo = cv2.resize(frame, (640, 360))
        results = self.person_detector.predict(small_for_yolo, classes=[0, 67], conf=0.5, verbose=False)
        for r in results:
            for box in r.boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h_ratio = frame.shape[0] / 360
                    w_ratio = frame.shape[1] / 640
                    people_boxes.append((int(x1 * w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)))
                elif box.cls  == 67:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h_ratio = frame.shape[0] / 360
                    w_ratio = frame.shape[1] / 640
                    phone_boxes.append((int(x1 * w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)))

        # Face Recognition не кожен кадр
        if self.frame_count % face_interval == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                confidence = 0
                face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_faces_names[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100
                face_names.append((name, confidence))

            face_locations = [
                (int(t / scale_factor), int(r / scale_factor), int(b / scale_factor), int(l / scale_factor))
                for (t, r, b, l) in face_locations
            ]

            # Безпечне оновлення спільних змінних
            with self.lock:
                self.face_locations = face_locations
                self.face_names = face_names

        # Оновлення списку тіл
        with self.lock:
            self.people_boxes = people_boxes
            self.phone_boxes = phone_boxes

        self.frame_count += 1

    def draw_results(self, frame):
        with self.lock:
            people_boxes = self.people_boxes.copy()
            face_locations = self.face_locations.copy()
            face_names = self.face_names.copy()

        for (x1, y1, x2, y2) in people_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        for (x1, y1, x2, y2)  in self.phone_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, "Phone", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
            # Якщо координати обличчя виходять в координати телефона то підсвчуємо червоним рамку
            face_box_color = (0, 255, 0)
            for (px1, py1, px2, py2) in self.phone_boxes:
                if left < px2 and right > px1 and top < py2 and bottom > py1:
                    face_box_color = (0, 0, 255)
                    break
            cv2.rectangle(frame, (left, top), (right, bottom), face_box_color, 2)
            cv2.putText(frame, f"{name} ({confidence:.1f}%)", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_box_color, 2)


def run_face_recognition(camera_id=0, photos_folder="photos"):
    system = FaceRecognitionSystem()
    if not system.load_known_faces(photos_folder):
        print("Не вдалось завантажити обличчя!")
        return

    grabber = FrameGrabber(camera_id)
    grabber.start()

    print("🚀 Асинхронне розпізнавання облич і тіл запущено!")

    fps_start = time.time()
    fps_counter = 0
    fps = 0

    process_thread = None

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                continue

            # Якщо попередній потік завершився — запускаємо новий для обробки
            if process_thread is None or not process_thread.is_alive():
                process_thread = threading.Thread(target=system.process_frame, args=(frame,))
                process_thread.start()

            # Малюємо результати
            system.draw_results(frame)

            # FPS
            fps_counter += 1
            if time.time() - fps_start >= 1:
                fps = fps_counter
                fps_counter = 0
                fps_start = time.time()

            cv2.rectangle(frame, (5, 5), (120, 35), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Heimdall — Async Face+Body Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Зупинено користувачем")
    finally:
        grabber.stop()
        grabber.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_face_recognition(camera_id=0, photos_folder="photos")
    except Exception as e:
        print("❌ Помилка:", e)
        traceback.print_exc()
