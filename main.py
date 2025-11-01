import pandas as pd
import numpy as np
import seaborn as sns # імпортуємо seaborn для візуалізації градієнтів
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score # імпортуємо модулі для розділення даних та крос-валідації
from sklearn.preprocessing import StandardScaler, LabelEncoder # імпортуємо модулі для масштабування та кодування міток
from sklearn.linear_model import LogisticRegression # імпортуємо логістичну регресію
from sklearn.tree import DecisionTreeClassifier # імпортуємо класифікатор дерев рішень
from sklearn.ensemble import RandomForestClassifier # імпортуємо випадковий ліс
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score
) # імпортуємо метрики для оцінки моделей: точність, матриця плутанини, звіт класифікації


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
    return df

# ================= 2. Огляд даних (EDA)  =================
def eda(df):
    print("Перші 5 рядків даних:")
    print(df.head(), "\n")

    print("Описова статистика:")
    print(df.describe(), "\n")

    print("Інформація про дані:")
    print(df.info(), "\n")

    print("Розподіл цільової змінної 'churn':")
    print(df['churn'].value_counts(), "\n")

# ================= 3. Обробка даних =================
def data_preprocessing(data):
    # add lose value
    data.loc[data.sample(50).index, 'income'] = np.nan  # додавання пропущених значень для демонстрації
    data.loc[data.sample(30).index, 'credit_score'] = np.nan  # додавання пропущених значень для демонстрації
    print(f"Пропущені значення у кожному стовпці:\n{data.isnull().sum()}\n")

    # Заповнення пропущених значень медіаною для числових стовпців
    data['income'] = data['income'].fillna(data['income'].median())
    data['credit_score'] = data['credit_score'].fillna(data['credit_score'].median())
    print("Пропущені значення заповнено медіаною для числових стовпців.\n")
    return data

# ================= 4. Інженерія ознак =================
def feature_engineering(data):
    # Створення нових ознак
    data['income_per_product'] = data['income'] / data['num_products'] # дохід на продукт
    data['balance_to_income_ratio'] = data['account_balance'] / (data['income'] + 1)# співвідношення балансу до доходу і уникнення ділення на нуль
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Old']) # вікові групи
    data['is_high_value'] = (data['account_balance'] > 50000).astype(int) # високий баланс

    # Кодування категоріальних змінних
    data = pd.get_dummies(data, columns=['country'], prefix='country', drop_first=True)
    le = LabelEncoder()
    data['age_group_encoded'] = le.fit_transform(data['age_group'])
    print(f'Нові колонки після інженерії ознак: {data.columns.tolist()}\n')

    return data

# ================= 5. Підготовка даних для моделювання =================
def prepare_data(data):
    data_ml = data.drop(['customer_id', 'age_group'], axis=1) # видалення непотрібних стовпців

    # Розділення на ознаки та цільову змінну
    X = data_ml.drop('churn', axis=1)
    y = data_ml['churn']

    # Розділення на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# ================= Тренування моделей та оцінка =================
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'MSE': mse,
            'MAE': mae,
            'R2 Score': r2,
            'ROC AUC': roc_auc
        })

        print("=" * 50)
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Виведення матриці плутанини та звіту класифікації
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for ' + name)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    results_df  = pd.DataFrame(results)
    print("\nПорівняння моделей:")
    display(results_df)

    # Візуалізація порівняння моделей
    results_df.set_index("Model")[["Accuracy"]].plot(kind='bar', figsize=(7, 4))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.show()


def main():
    df = generate_data()
    # eda(df)
    df = data_preprocessing(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()