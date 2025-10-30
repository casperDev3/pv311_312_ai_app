import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score # імпортуємо модулі для розділення даних та крос-валідації
from sklearn.preprocessing import StandardScaler, LabelEncoder # імпортуємо модулі для масштабування та кодування міток
from sklearn.linear_model import LogisticRegression # імпортуємо логістичну регресію
from sklearn.tree import DecisionTreeClassifier # імпортуємо класифікатор дерев рішень
from sklearn.ensemble import RandomForestClassifier # імпортуємо випадковий ліс
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # імпортуємо метрики для оцінки моделей: точність, матриця плутанини, звіт класифікації


def generate_data():
    # ======== 1. Створення синтетичних датасету ========
    print("1. Створення синтетичних даних...")
    np.random.seed(42)  # для відтворюваності результатів, тому що ми використовуємо випадкові числа

    n_samples = 1000
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'account_balance': np.random.randint(-5000, 100000, n_samples),
        'num_products': np.random.randint(1, 5, n_samples),
        'has_credit_card': np.random.choice([0, 1], n_samples),
        'is_active_member': np.random.choice([0, 1], n_samples),
        'country': np.random.choice(['Ukraine', 'Poland', 'Germany'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    print(f"Синтетичні дані створено.\n, {df.shape[0]} записів та {df.shape[1]} стовпців.\n")

    # Збереження даних у CSV, XLSX та JSON
    # df.to_csv('files/synthetic_data.csv', index=False)
    # df.to_excel('files/synthetic_data.xlsx', index=False)
    # df.to_json('files/synthetic_data.json', orient='records', lines=True)





def main():
    generate_data()

if __name__ == "__main__":
    main()