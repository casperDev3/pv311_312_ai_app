import ssl
import numpy as np
import pickle
import gzip
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate

        # Ініціалізація ваг з випадковими значеннями
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Для числової стабільності
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, X):
        # Пряме поширення
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y, Y_hat):
        # Перехресна ентропія
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
        return loss

    def backward(self, X, Y, Y_hat):
        m = X.shape[1]

        # Зворотне поширення
        dZ2 = Y_hat - Y
        dW2 = np.dot(dZ2, self.A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Оновлення ваг
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, Y, epochs=1000, batch_size=32):
        losses = []
        accuracies = []
        m = X.shape[1]

        for epoch in range(epochs):
            # Міні-батч навчання
            for i in range(0, m, batch_size):
                end = min(i + batch_size, m)
                X_batch = X[:, i:end]
                Y_batch = Y[:, i:end]

                # Пряме та зворотне поширення
                Y_hat = self.forward(X_batch)
                self.backward(X_batch, Y_batch, Y_hat)

            # Обчислення втрат кожні 30 епох
            if epoch % 30 == 0:
                Y_hat_full = self.forward(X)
                loss = self.compute_loss(Y, Y_hat_full)
                accuracy = self.accuracy(X, Y)
                losses.append(loss)
                accuracies.append(accuracy)
                print(f"Епоха {epoch}, Втрати: {loss:.4f}, Точність: {accuracy:.4f}")

        return losses, accuracies

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=0)

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=0)
        return np.mean(predictions == true_labels)


class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def load_mnist_data(self):
        """Завантаження даних MNIST без TensorFlow"""
        url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
        filename = 'mnist.pkl.gz'

        if not os.path.exists(filename):
            print("Завантаження MNIST даних...")
            # Вимкнути перевірку SSL
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(url, filename)

        with gzip.open(filename, 'rb') as f:
            train_data, val_data, test_data = pickle.load(f, encoding='latin1')

        # Підготовка даних
        X_train = train_data[0].T
        Y_train = self.one_hot_encode(train_data[1]).T
        X_test = test_data[0].T
        Y_test = self.one_hot_encode(test_data[1]).T

        return (X_train, Y_train), (X_test, Y_test)

    def one_hot_encode(self, y, num_classes=10):
        """One-hot кодування міток"""
        return np.eye(num_classes)[y]

    def create_and_train_model(self, hidden_units=128, epochs=1000, learning_rate=0.1):
        """Створення та навчання моделі"""
        print("Завантаження даних...")
        (X_train, Y_train), (X_test, Y_test) = self.load_mnist_data()

        input_size = X_train.shape[0]
        output_size = Y_train.shape[0]

        print(f"Розмірність даних: {X_train.shape}")
        print(f"Розмірність міток: {Y_train.shape}")

        # Створення моделі
        self.model = NeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_units,
            output_size=output_size,
            learning_rate=learning_rate
        )

        print("Початок навчання...")
        losses, accuracies = self.model.train(X_train, Y_train, epochs=epochs)

        # Оцінка на тестових даних
        test_accuracy = self.model.accuracy(X_test, Y_test)
        print(f"\nТочність на тестових даних: {test_accuracy:.4f}")

        self.is_trained = True

        # Графік втрат та точності
        plt.figure(figsize=(12, 4))

        # Графік втрат
        plt.subplot(1, 2, 1)
        epochs_recorded = range(0, epochs, 30)
        plt.plot(epochs_recorded, losses)
        plt.title('Графік втрат під час навчання')
        plt.xlabel('Епоха')
        plt.ylabel('Втрати')
        plt.grid(True)

        # Графік точності
        plt.subplot(1, 2, 2)
        plt.plot(epochs_recorded, accuracies)
        plt.title('Графік точності під час навчання')
        plt.xlabel('Епоха')
        plt.ylabel('Точність')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return test_accuracy

    def visualize_predictions(self, X_test, Y_test, num_examples=5):
        """Візуалізація прогнозів"""
        indices = np.random.choice(X_test.shape[1], num_examples, replace=False)

        plt.figure(figsize=(12, 3))
        for i, idx in enumerate(indices):
            x = X_test[:, idx:idx + 1]
            true_label = np.argmax(Y_test[:, idx])
            prediction = self.model.predict(x)[0]

            plt.subplot(1, num_examples, i + 1)
            plt.imshow(x.reshape(28, 28), cmap='gray')
            plt.title(f'True: {true_label}\nPred: {prediction}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def predict_digit(self, image_path):
        """Передбачення цифри з зображення"""
        if not self.is_trained:
            print("Модель не навчена! Спочатку викличте create_and_train_model().")
            return None

        try:
            # Завантаження та обробка зображення
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0

            # Інвертуємо кольори (MNIST має білі цифри на чорному фоні)
            img_array = 1 - img_array

            # Вирівнюємо та підготовлюємо дані
            img_flat = img_array.reshape(784, 1)

            # Передбачення
            prediction_probs = self.model.forward(img_flat)
            digit = np.argmax(prediction_probs)
            confidence = np.max(prediction_probs)

            # Візуалізація
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(img_array, cmap='gray')
            plt.title(f"Вхідне зображення")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            bars = plt.bar(range(10), prediction_probs.flatten())
            bars[digit].set_color('red')
            plt.title(f"Результат: {digit}\nВпевненість: {confidence:.2f}")
            plt.xlabel('Цифри')
            plt.ylabel('Ймовірність')
            plt.xticks(range(10))

            plt.tight_layout()
            plt.show()

            print(f"Розпізнана цифра: {digit} (впевненість: {confidence:.2%})")

            return digit, confidence

        except Exception as e:
            print(f"Помилка при обробці зображення: {e}")
            return None

    def save_model(self, filename='digit_model.pkl'):
        """Збереження моделі"""
        if self.is_trained:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Модель збережена як {filename}")

    def load_model(self, filename='digit_model.pkl'):
        """Завантаження моделі"""
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Модель завантажена з {filename}")


def create_test_image(digit, filename='test_digit.png'):
    """Створення тестового зображення для перевірки"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('L', (28, 28), color=0)  # Чорний фон
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Малюємо білу цифру
    draw.text((10, 4), str(digit), fill=255, font=font)
    img.save(filename)
    print(f"Тестове зображення створено: {filename}")
    return filename


# Приклад використання
def main():
    recognizer = DigitRecognizer()

    print("=== НАВЧАННЯ МОДЕЛІ ===")
    # Навчання моделі (закоментуйте після першого запуску)
    accuracy = recognizer.create_and_train_model(
        hidden_units=128,
        epochs=60,  # Зменшено для швидшого тестування
        learning_rate=0.1
    )

    # Збереження моделі
    recognizer.save_model()

    print("\n=== ТЕСТУВАННЯ НА ВЛАСНИХ ЗОБРАЖЕННЯХ ===")
    # Створення тестового зображення
    test_digit = 1
    test_image_path = create_test_image(test_digit)

    # Розпізнавання
    result = recognizer.predict_digit(test_image_path)

    if result:
        digit, confidence = result
        print(f"Цифра {digit} розпізнана з впевненістю {confidence:.2%}")


if __name__ == "__main__":
    main()