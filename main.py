from pyexpat import features

import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import pickle
import sounddevice as sd
import soundfile as sf

class VoiceIdentifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(kernel="linear", probability=True)
        self.users = {}
        self.is_trained = False

    def extract_features(self, audio_path, duration=3):
        """Витягнення MFCC особливостей з аудіо"""
        try:
            y, sr = librosa.load(audio_path, duration=duration)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Додаткові особливості
            special_centroid = np.mean(librosa.feature.special_centroid(y=y, sr=sr))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))

            features = np.concatenate(
                [
                    mfcc_mean,
                    mfcc_std,
                    [special_centroid, zero_crossing]
                ]
            )

            return features
        except Exception as err:
            print("Помилка при опрацюванні аудіо", err)
            return None

    def record_voice(self, file_name, duration=3, sample_rate=22050):
        print(f"Запис голосу {duration} секунд...")
        recording = sd.rec(int(duration * sample_rate), sample_rate=sample_rate, channels=1)
        sd.wait()
        sd.write(file_name, recording, sample_rate)
        print(f"Записано у файл: {file_name}")

    def add_user(self, user_id, num_samples=3):
        print(f"Додавання користувача {user_id}. Потрібно записати {num_samples} зразків!")
        samples = []

        for i in range(num_samples):
            input(f"Натисність Enter для запису зразка {i + 1} ...")
            filename = f"temp_sample_{i}.wav"
            self.record_voice(filename)

            features = self.extract_features(filename)
            if features is not None:
                samples.append(features)

            if os.path.exists(filename):
                os.remove(filename)

        if samples:
            self.users[user_id] = samples
            print(f"Користувач {user_id} доданий успішно!")
            return True

        return False

    def prepare_training_data(self):
        X = []
        y = []

        for user_id, samples in self.users.item():
            for sample in samples:
                X.append(sample)
                y.append(user_id)

        return  np.array(X), np.array(y)


    def train_model(self):
        if len(self.users) < 2:
            print("Потрібно мінімум 2 користувача")

            X, y = self.prepare_training_data()

            X_scaled = self.scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            self.is_trained = True
            print(f"Модкль навчена! Точність: {acc:2f} %")
            return True

    def indentify_voice(self, audio_path=None):
        if not self.is_trained:
            print("Модель ще не навчена!")
            return None

        if audio_path is None:
            audio_path = "test_sample.wav"
            self.record_voice(audio_path)

        features = self.extract_features(audio_path)
        if features is None:
            return None

        features_scaled = self.scaler.transform([features])
        probabilities = self.model.predict_proba(features_scaled)[0]

        results = {}
        for user_id, prob  in zip(self.model.classes_, probabilities):
            results[user_id] = prob

        if audio_path == "test_sample.wav" and os.path.exists(audio_path):
            os.remove(audio_path)

        return results




def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()