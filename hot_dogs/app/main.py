import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from tf_keras.models import load_model

model_dir = Path(__file__).resolve().parents[1] / 'models' / 'connectors'
model = load_model(model_dir / 'keras_model.h5', compile=False)
labels = [line.split(' ', 1)[1] for line in
          (model_dir / 'labels.txt').read_text(encoding='utf-8').splitlines()]

app = FastAPI(title='Распознавание разъёмов')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(file: UploadFile):
    if file.size is not None and file.size > 10 * 1024 * 1024:
        raise HTTPException(413, 'Файл больше 10 МБ.')

    try:
        with Image.open(file.file, formats=['JPEG', 'PNG']) as photo:
            image = ImageOps.fit(photo.convert('RGB'), (224, 224), Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(422, 'Не удалось открыть изображение. Загрузите JPG или PNG.') from error

    pixels = np.asarray(image, dtype=np.float32)[None, ...]
    scores = model(pixels / 127.5 - 1, training=False).numpy()[0]
    index = scores.argmax()
    return {'class': labels[index], 'confidence_percent': round(float(scores[index]) * 100, 2)}
