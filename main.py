"""
🧠 ПРОСТИЙ ПЕРЦЕПТРОН З НУЛЯ
Автор: Igorich
Мета: Навчити перцептрон розпізнавати, чи лежить точка (x, y)
      вище або нижче діагоналі y = x.
Мова: Python 3
Бібліотеки: numpy, matplotlib

🧩 Що буде зроблено:
1. Згенеруємо дані (точки)
2. Реалізуємо перцептрон з нуля
3. Навчимо його класифікувати
4. Візуалізуємо результат
"""
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_data(n=1000):
    X = np.random.uniform(-1, 1, (n, 2))
    y = np.array([1 if x[1] > x[0] else 0 for x in X])
    return X, y

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=50):
        """
        Ініціалізація перцептрона.
        :param input_size:  - розмірність вхідних даних (кількість ознак)
        :param learning_rate: - швидкість навчання
        :param epochs:  - кількість епох навчання
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # +1 для зсуву (bias)

    def activation(self,  x):
        return np.where(x >= 0, 1, 0) # Softmax активація

    def predict(self, x):
        x_with_bias = np.insert(x, 0, 1) # Додаємо зсув
        z = np.dot(self.weights, x_with_bias) # Лінійна комбінація
        return self.activation(z)

def main():
    data = generate_data(100)
    X, y = data
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=20)
    print(data)

if __name__ == "__main__":
    main()