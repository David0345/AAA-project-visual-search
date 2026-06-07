import pandas as pd
from pathlib import Path

# Пути (убедитесь, что train.parquet уже существует)
input_path = Path("data/processed/train.parquet")
output_path = Path("data/processed/train_mini.parquet")

if not input_path.exists():
    print("Ошибка: data/processed/train.parquet не найден. Сначала запустите scripts/prepare_data.py")
else:
    # Читаем и берем первые 256 строк (или случайные 256 через df.sample(256))
    df = pd.read_parquet(input_path)
    df_mini = df.head(256).copy()

    df_mini.to_parquet(output_path, index=False)
    print(f"✅ Успешно создан мини-датасет: {output_path}")
    print(f"   Количество строк: {len(df_mini)}")
    print(f"   Колонки: {list(df_mini.columns)}")