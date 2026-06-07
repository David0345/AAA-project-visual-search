import pandas as pd
from pathlib import Path
from visual_search.common.io import PROCESSED_DIR

input_path = Path(PROCESSED_DIR / "train.parquet")
output_path = Path(PROCESSED_DIR / "train_mini.parquet")


if not input_path.exists():
    print("Ошибка: data/processed/train.parquet не найден. Сначала запустите scripts/prepare_data.py")
else:
    df = pd.read_parquet(input_path)
    df_mini = df.sample(256).copy()

    df_mini.to_parquet(output_path, index=False)
    print(f"   Успешно создан мини-датасет: {output_path}")
    print(f"   Количество строк: {len(df_mini)}")
    print(f"   Колонки: {list(df_mini.columns)}")
