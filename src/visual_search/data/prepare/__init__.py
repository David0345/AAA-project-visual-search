"""Офлайн-подготовка данных (out-of-loop): EDA-фильтрация и сборка train.parquet.

Сюда переезжает существующий код (отдельным PR, см. §4 PROJECT_STRUCTURE):
  * ./eda                              -> prepare/eda/
  * ./train_dataset/prepare_train_data.py -> prepare/build_train.py

Запускается редко и вручную; результат (train.parquet) — в data/processed/.
"""
