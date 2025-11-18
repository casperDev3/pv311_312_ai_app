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
        self.model = SVC(kernel='linear', probability=True)
        self.users = {}
        self.is_trained = False

    def extract_features(self, audio_path, duration=3):
        """Витягнення MFCC особливостей з аудіо"""
        try:
            # Завантаження аудіо
            y, sr = librosa.load(audio_path, duration=duration)

            # MFCC особливості
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Додаткові особливості
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))

            # Комбінування особливостей
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, zero_crossing]
            ])

            return features
        except Exception as e:
            print(f"Помилка обробки аудіо: {e}")
            return None

    def record_voice(self, filename, duration=3, sample_rate=22050):
        """Запис голосу"""
        print(f"Запис голосу {duration} секунд...")
        recording = sd.rec(int(duration * sample_rate),
                           samplerate=sample_rate,
                           channels=1)
        sd.wait()
        sf.write(filename, recording, sample_rate)
        print(f"Записано у файл: {filename}")

    def add_user(self, user_id, num_samples=3):
        """Додавання нового користувача"""
        print(f"Додавання користувача {user_id}. Запис {num_samples} зразків...")

        samples = []
        for i in range(num_samples):
            input(f"Натисніть Enter для запису зразка {i + 1}...")
            filename = f"temp_sample_{i}.wav"
            self.record_voice(filename)

            features = self.extract_features(filename)
            if features is not None:
                samples.append(features)

            # Видалення тимчасового файлу
            if os.path.exists(filename):
                os.remove(filename)

        if samples:
            self.users[user_id] = samples
            print(f"Користувач {user_id} доданий успішно!")
            return True
        return False

    def prepare_training_data(self):
        """Підготовка даних для навчання"""
        X = []
        y = []

        for user_id, samples in self.users.items():
            for sample in samples:
                X.append(sample)
                y.append(user_id)

        return np.array(X), np.array(y)

    def train_model(self):
        """Навчання моделі"""
        if len(self.users) < 2:
            print("Потрібно мінімум 2 користувача для навчання")
            return False

        X, y = self.prepare_training_data()

        # Масштабування ознак
        X_scaled = self.scaler.fit_transform(X)

        # Розділення на тренувальну та тестову вибірки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Навчання моделі
        self.model.fit(X_train, y_train)

        # Оцінка точності
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.is_trained = True
        print(f"Модель навчена! Точність: {accuracy:.2f}")
        return True

    def identify_voice(self, audio_path=None):
        """Ідентифікація голосу"""
        if not self.is_trained:
            print("Модель не навчена!")
            return None

        if audio_path is None:
            # Запис нового зразка
            audio_path = "test_sample.wav"
            self.record_voice(audio_path)

        features = self.extract_features(audio_path)
        if features is None:
            return None

        # Масштабування та прогноз
        features_scaled = self.scaler.transform([features])
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Результати
        results = {}
        for user_id, prob in zip(self.model.classes_, probabilities):
            results[user_id] = prob

        # Видалення тимчасового файлу
        if audio_path == "test_sample.wav" and os.path.exists(audio_path):
            os.remove(audio_path)

        return results

    def save_model(self, filename="voice_model.pkl"):
        """Збереження моделі"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'model': self.model,
                'users': self.users,
                'is_trained': self.is_trained
            }, f)
        print(f"Модель збережена у файл: {filename}")

    def load_model(self, filename="voice_model.pkl"):
        """Завантаження моделі"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.scaler = data['scaler']
            self.model = data['model']
            self.users = data['users']
            self.is_trained = data['is_trained']
            print(f"Модель завантажена з файлу: {filename}")
            return True
        except Exception as e:
            print(f"Помилка завантаження моделі: {e}")
            return False


# Приклад використання
def main():
    identifier = VoiceIdentifier()

    while True:
        print("\n=== Система ідентифікації по голосу ===")
        print("1. Додати користувача")
        print("2. Навчити модель")
        print("3. Ідентифікувати голос")
        print("4. Зберегти модель")
        print("5. Завантажити модель")
        print("6. Вийти")

        choice = input("Оберіть опцію: ")

        if choice == '1':
            user_id = input("Введіть ID користувача: ")
            identifier.add_user(user_id)

        elif choice == '2':
            identifier.train_model()

        elif choice == '3':
            if identifier.is_trained:
                results = identifier.identify_voice()
                if results:
                    print("\nРезультати ідентифікації:")
                    for user_id, confidence in results.items():
                        print(f"Користувач {user_id}: {confidence:.2%}")
            else:
                print("Спочатку навчіть модель!")

        elif choice == '4':
            identifier.save_model()

        elif choice == '5':
            identifier.load_model()

        elif choice == '6':
            break

        else:
            print("Невірний вибір!")


if __name__ == "__main__":
    main()