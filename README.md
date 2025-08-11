
# Sentiment Analysis Baseline

Простой baseline для **анализа тональности текста** на основе датасета [UCI Sentiment Labeled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences).  
Модель классифицирует предложения на **positive** или **negative**.

---

## Структура проекта

```
.
├── data/                   
│   ├── amazon_cells_labelled.txt
│   ├── imdb_labelled.txt
│   └── yelp_labelled.txt
├── preprocessing.py          
├── data_loader.py            
├── model.py                  
├── utils.py                  
├── train.py                  
├── requirements.txt
```

---

## Установка и запуск

1. **Клонировать репозиторий**
    
```bash
git clone https://github.com/username/sentiment-baseline.git
cd sentiment-baseline
```

2. **Создать виртуальное окружение и установить зависимости**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Скачать и положить данные в папку `data/`**  
    [Ссылка на датасет](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)  
    Положите три файла:
    ```
    amazon_cells_labelled.txt
    imdb_labelled.txt
    yelp_labelled.txt
    ```
    
4. **Запустить обучение**
```bash
python train.py \
  --data_dir data \
  --vectorizer bow \
  --out_dir artifacts \
  --test_size 0.2
```

5. **Пример запуска с TF-IDF**

```bash
python train.py --data_dir data --vectorizer tfidf --out_dir artifacts
```

---

## Результат

После запуска в папке `artifacts/` появятся:
- **`sentiment_benchmark.joblib`** — сохранённая модель (векторизатор + классификатор)
- **`report.txt`** — текстовый отчёт с метриками
- **`metrics.json`** — метрики в JSON

