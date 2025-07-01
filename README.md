## Описание 

Streamlit-приложение для классификации отзывов кинокритиков на два класса:
- положительный 
- отрицательный 

Для анализа текста отзывов используется ML-модель (TF-IDF + SVD + CatBoost).

## Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone https://github.com/andrey-ermilov/text-classification.git
cd /text-classification
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите приложение:
```bash
streamlit run app.py
```
