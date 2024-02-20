# Topic 1. One Stage Object detection

[Запись занятия]()

Команде студентов необходимо реализовать и обучить модель YOLO первой версии, используя фреймворк pytorch, на датасете [COCO-2017](https://docs.voxel51.com/user_guide/dataset_zoo/index.html)

Критерии оценки:
- Обучение модели 3 балла
- Добавить самописную реализацию NMS и якорей(Anchors) 4 балла
- Добавить реализацию метрики mAP 5 баллов

Решение необходимо продоставить в формате Pull Request в данную ветку, код рещения должен быть написан в скриптах, без использования Jupyter.
Зависимости зафиксированы в файле requirements.txt.

### Оформление Pull Request
название PR доожно быть в виде: "team #номер или название команды"
в описании PR указать:
- состав команды: фамилии имена участников команды
- что из критериев сделали
- метрики получившейся модели на сплите `validation`
