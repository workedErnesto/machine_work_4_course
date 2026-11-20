from io import BytesIO

import requests
import streamlit as st

api_url = 'http://127.0.0.1:8010/predict'

st.set_page_config(page_title='Распознавание разъёмов')
st.markdown('''
<style>
.stApp * { border-radius: 0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stFileUploaderDropzone"] button[aria-label^="Remove "] {
    display: inline-grid;
    place-items: center;
    min-width: 1.5rem;
    min-height: 1.5rem;
}
[data-testid="stFileUploaderDropzone"] button[aria-label^="Remove "] svg { display: none; }
[data-testid="stFileUploaderDropzone"] button[aria-label^="Remove "]::after { content: "×"; }
</style>
''', unsafe_allow_html=True)

st.title('Распознавание разъёмов', anchor=False)
st.write('По фотографии определяем тип штекера USB-A, USB-C или HDMI.')

with st.form('connector_photo'):
    photo = st.file_uploader(
        'Фотография штекера в JPG или PNG, до 10 МБ',
        type=['jpg', 'jpeg', 'png'],
        max_upload_size=10,
    )
    submitted = st.form_submit_button('Распознать', type='primary')

if not submitted:
    st.stop()

if photo is None:
    st.warning('Выберите фотографию.')
    st.stop()

try:
    response = requests.post(api_url, files={'file': (photo.name, photo.getvalue())}, timeout=15)
except requests.RequestException:
    st.error('Нет ответа от API. Проверьте, что сервер запущен на порту 8010.')
    st.stop()

if response.status_code in (413, 422):
    detail = response.json().get('detail')
    st.error(detail if isinstance(detail, str) else 'Проверьте формат файла.')
    st.stop()

if not response.ok:
    st.error('API вернул ошибку. Попробуйте ещё раз.')
    st.stop()

result = response.json()
st.subheader('Результат', anchor=False)
preview, prediction = st.columns(2)
with preview:
    st.image(BytesIO(photo.getvalue()))
with prediction:
    st.metric('Тип штекера', result['class'])
    st.metric('Уверенность модели', f"{result['confidence_percent']:.2f}%")
