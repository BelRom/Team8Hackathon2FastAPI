## Шаги обработки данных

Ниже представлен пошаговый разбор действий, выполненных в предоставленном файле.

---

- ссылка на докер image
  https://drive.google.com/file/d/1nQV2-l1u5sgz6fN9leuHeqbititRGyf6/view?usp=sharing
  
- для запуска контейнера нужно выолнить
- docker run -it --rm --name=container_name -p=8000:8000 team_8 
- для открытия сайта перейти по url
- [localhost:8000](http://localhost:8000)

- для сборки модели через проект нужно выполнить команду 
- python src/pipeline_pack_model.py train 

- для запуска сервера для сайта 
- uvicorn main:app --port 8000 --app-dir src


### 1. Обработка датасета df_hits

#### Что сделано? 

1. Целевые события хранятся в колонке event_action. Целевые события в этой колонке заменили на 1, остальное - 0

---

### 2. Обработка датасета `df_sessions`

#### Что сделано?
1. **Уменьшение числа колонок**:  
   - Убраны некоторые избыточные или малозначимые столбцы, такие как идентификатор клиента, модель устройства и разрешение экрана.
   
2. **Заполнение пропусков**:  
   - Незначащие элементы в ряде столбцов, таких как рекламные кампании и источник трафика, заменены на стандартизированное значение `"unknown"`.
   
3. **Разбиение временного параметра**:  
   - Дата и время посещения разделены на составляющие (год, месяц, день, день недели, рабочий или выходной день, час посещения).
   
4. **Классификация устройств**:  
   - Используются только три самые распространенные версии браузеров и операционных систем, остальные маркируются как "другое".
   
5. **One-Hot encoding**:  
   - Некоторые ключевые признаки типа устройства и операционной системы представлены в виде двоичных индикаторов (наличие-отсутствие).
   
6. **Топовые UTM-метки**:  
   - Только наиболее часто встречающиеся источники, каналы и другие рекламно-трекинговые параметры оставлены, остальные сгруппированы под одним названием "other".
   
7. **OrdinalEncoding остальных признаков**:  
   - Остальные категориальные признаки были переведены в числовой вид с помощью одного общего подхода (OrdinalEncoder).

---

### 3. Объединение датасетов `df_hits` и `df_sessions`

1. Столбец с целевой переменной из датасетв df_hots добавляем в обработанный датасет df_sessions
---

### 4. Подготовка итогового датасета

Итоговая версия объединяет подготовленный объединенный датасет и сохраняет его для последующей работы. Здесь проведен финальный контроль наличия пропущенных значений и сохранение результата в CSV-файл.

---

### 5. Попытка снизить размерность данных

Хотя попытка была сделана для снижения размерности с помощью алгоритма UMAP, в связи с большими размерами данных выполнение привело к ошибке. Это решение пока отложено для дальнейшей оптимизации ресурсов.

---

Таким образом, рассмотренный скрипт охватывает весь цикл обработки данных, начиная от загрузки и первичного анализа и заканчивая созданием унифицированного набора данных, готового для анализа и построения моделей машинного обучения.

### 6. Разведочный анализ (EDA)

1. Построили диаграмму рассеяния.
2. Гистограммы распределения признаков.
3. Статистический анализ.
4. Матрица корреляции признаков с целевой переменной - y.
5. Анализ ключевых числовых признаков, временных характеристик и поведенческих метрик.
