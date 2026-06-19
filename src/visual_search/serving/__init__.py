"""serving — веб-сервис поиска.

Опирается на ТОТ ЖЕ query pipeline, что и оценка: encode -> index.ann.search ->
hydrate метаданных. Деплой добавляется поверх (Dockerfile в корне), без правок
в обучении/оценке. Re-ranker, если понадобится, вставляется в search.py после ANN.
"""
