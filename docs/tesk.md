SigLIPv2 на практике
Рассмотрим возможности SigLIPv2 на примере модели google/siglip2-base-patch16-224 из HuggingFace. Чтобы начать работать с моделью, инициализируйте:
Саму модель через класс AutoModel.
Обработчик входных данных через AutoProcessor.from_pretrained. Передайте в него аргумент с именем модели.
Сам обработчик, аналогично токенизатору, принимает на вход тексты и соответствующие им картинки. Дополнительно можно задать формат возвращаемых тензоров и способ выполнения пэддинга.

from transformers import AutoProcessor, AutoModel

# загружаем новую модель
siglip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

# подготовка данных
inputs = processor(
    text= # ваши тексты
    images= # картинка как PIL.Image
    padding="max_length",
    return_tensors="pt"
) 
Может возникнуть вопрос: а как формировать текст для картинки? Как и в CLIP, авторы SigLIPv2 передавали в label желаемый класс с помощью промпта формата 'This is a photo of {label}.'. Для одной картинки можно создать несколько текстовых описаний и передать их на вход модели, а результирующим вектором будут логиты предсказания для каждой пары «промпт-картинка».

# инференс
outputs = siglip_model(**inputs)
logits_per_image = outputs.logits_per_image  # логиты для каждого текста по изображению 
Максимальное значение логита — лучшее совпадение текста и картинки.
Задание 1
Скачайте фото. 
Вам дано изображение car.jpeg и потенциальные классы candidate_labels, к которым оно может относиться. Используя модель SigLIPv2, определите наиболее вероятный класс для фото. Выведите int-индекс наиболее вероятного класса.
Выполните задание локально, а затем сверьтесь с авторским решением и ожидаемым выводом. 
Прекод

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

image = # загрузите картинку
candidate_labels = ["a car", "a bicycle", "a truck"]

# инициализируйте и выполните инференс модели siglipv2
 


Подсказка