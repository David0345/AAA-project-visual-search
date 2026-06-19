"""training — оркестрация одного прогона обучения.

Берёт DataLoader'ы из data/, модель из models.build_model(config), гоняет
train-loop, логирует метрики и пишет чекпойнт в experiments/<run_id>/.
"""
