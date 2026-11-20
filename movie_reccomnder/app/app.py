from html import escape
from pathlib import Path

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Рекомендации фильмов", layout="wide", initial_sidebar_state="expanded")

SERVER = "http://127.0.0.1:8000"
API_URL = SERVER + "/api"

st.html(Path(__file__).with_name("style.css"))

COLUMNS = {'title': 'Фильм', 'year': 'Год', 'genres': 'Жанры', 'rating': 'Оценка',
           'similarity': 'Сходство', 'text_similarity': 'По описанию', 'genre_similarity': 'По жанрам',
           'common': 'Общих зрителей', 'common_films': 'Общих фильмов',
           'user_id': 'ID пользователя', 'mean_rating': 'Средняя оценка'}


def get(path, **params):
    response = requests.get(API_URL + path, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def show_table(items, columns, empty="Ничего не найдено.", column_config=None, **options):
    if not items:
        st.write(empty)
        return
    table = pd.DataFrame(items)[columns].rename(columns=COLUMNS)
    config = {'Средняя оценка': st.column_config.NumberColumn(format="%.2f")}
    config.update(column_config or {})
    return st.dataframe(table, hide_index=True, width="stretch", column_config=config, **options)


def show_ranking(items, score_column, empty="Ничего не найдено."):
    if not items:
        st.write(empty)
        return
    rows = []
    for place, film in enumerate(items, start=1):
        if score_column == 'score':
            votes = f"{film['votes']:,}".replace(',', ' ')
            detail = f"Средняя {film['average']:.2f}<br>Оценок {votes}"
        else:
            detail = f"Соседей {film['neighbors']}"
        rows.append(
            f'<li><span class="movie-number">{place}</span>'
            f'<div><strong class="movie-name">{escape(film["title"])}</strong>'
            f'<small class="movie-detail">{film["year"]}, {escape(film["genres"])}</small></div>'
            f'<div class="movie-score"><span class="movie-value">{film[score_column]:.2f}</span>'
            f'<small class="movie-detail">{detail}</small></div></li>'
        )
    st.html('<ol class="movie-list" role="list">' + ''.join(rows) + '</ol>')


def choose_movie(key):
    query = st.text_input("Название фильма", key=f"{key}_query", placeholder="Например, Девятаев")
    if not query.strip():
        return None
    found = get("/search", q=query, limit=20)
    if not found:
        st.write("Фильм не найден.")
        return None
    return st.selectbox("Фильм", found, key=f"{key}_movie", format_func=lambda m: f"{m['title']} ({m['year']})")


def describe_param(name, schema, description):
    limits = []
    if 'enum' in schema:
        limits.append('варианты: ' + ', '.join(schema['enum']))
    if 'minimum' in schema:
        limits.append(f"от {schema['minimum']:g}" + (f" до {schema['maximum']:g}" if 'maximum' in schema else ''))
    if 'default' in schema:
        limits.append(f"по умолчанию {schema['default']}")
    return f"{name}: {description}" + (f" ({', '.join(limits)})" if limits else '')


def enough_ratings(ratings):
    return len(ratings) >= 3 and len({item['rating'] for item in ratings}) >= 2


try:
    stats = get("/status")
except requests.exceptions.RequestException:
    st.error("Нет соединения с API. Запустите его из папки app командой: uvicorn main:app")
    st.stop()

first_new = stats['last_dataset_user'] + 1
user_id = st.session_state.get("active_user")

if user_id is None or st.session_state.get("choosing_user"):
    with st.container(key="entry"):
        intro, profile = st.columns([1.2, 1], vertical_alignment="center")
        with intro:
            movies_count = f"{stats['movies']:,}".replace(',', ' ')
            profiles_count = f"{first_new - 1:,}".replace(',', ' ')
            st.html(
                '<section class="entry-intro">'
                '<div><h1>Рекомендации фильмов</h1><p>Подборки по данным Kion</p></div>'
                '<dl class="entry-stats">'
                f'<div><dt>Фильмов в каталоге</dt><dd>{movies_count}</dd></div>'
                f'<div><dt>Профилей Kion</dt><dd>{profiles_count}</dd></div>'
                '</dl></section>'
            )
        with profile, st.container(key="profile-form"), st.form("choose_user", border=False):
            st.subheader("Выберите профиль", anchor=False)
            chosen_id = st.number_input("ID пользователя", min_value=1, max_value=10 ** 9,
                                        value=user_id or 1642, step=1)
            st.caption(f"В профилях 1-{first_new - 1} уже есть история Kion. "
                       f"Чтобы начать с пустого профиля, введите ID от {first_new}.")
            if st.form_submit_button("Продолжить", type="primary", width="stretch"):
                st.session_state.active_user = chosen_id
                st.session_state.choosing_user = False
                st.rerun()
    st.stop()

sections = ["Популярное", "Персональные", "Похожие фильмы", "Похожие пользователи", "Оценить фильм", "Справка"]
section = st.session_state.get("section", sections[0])

# сохраняем фильтры при переходе между разделами
for key in ("popular_count", "personal_method", "similar_method", "similar_query", "similar_movie"):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

with st.sidebar:
    st.subheader("Рекомендательная система", anchor=False)
    with st.container(key="navigation"):
        for index, name in enumerate(sections):
            if st.button(name, key=f"nav_{index}", width="stretch",
                         type="primary" if name == section else "secondary"):
                st.session_state.section = name
                st.rerun()
    st.divider()
    st.markdown(f"**Профиль {user_id}**")
    st.caption("История Kion и свои оценки" if user_id < first_new else "Свои оценки")
    if st.button("Сменить пользователя", width="stretch"):
        st.session_state.choosing_user = True
        st.rerun()

try:
    if section in ("Похожие пользователи", "Персональные", "Оценить фильм"):
        user_list = get(f"/users/{user_id}/ratings")

    # популярные фильмы
    if section == "Популярное":
        heading, filter_box = st.columns([2.5, 1.1], vertical_alignment="bottom")
        with heading:
            st.title("Популярные фильмы", anchor=False)
            st.caption("По взвешенной оценке с учётом числа зрителей")
        with filter_box:
            count = st.number_input("Фильмов в списке", min_value=1, max_value=50, value=10, key="popular_count")
        show_ranking(get("/popular", limit=count), 'score')

    # похожие фильмы
    elif section == "Похожие фильмы":
        st.title("Похожие фильмы", anchor=False)
        method_column, search_column = st.columns(2, gap="medium")
        with method_column:
            method = st.selectbox("Способ", ["По контенту (описание и жанры)", "По оценкам пользователей"], key="similar_method")
        with search_column:
            movie = choose_movie("similar")
        if movie:
            st.subheader(movie['title'])
            st.caption(f"{movie['year']}, {movie['genres']}")
            with st.expander("Описание фильма"):
                st.write(movie['overview'] or "Описания в каталоге нет.")
            selected_description = st.empty()
            if method.startswith("По контенту"):
                kind = "similar"
                columns = ['title', 'year', 'genres', 'similarity', 'text_similarity', 'genre_similarity']
                empty = "Похожие фильмы не найдены."
            else:
                kind = "similar-by-ratings"
                columns = ['title', 'year', 'genres', 'similarity', 'common']
                empty = "Нет фильмов, которые смотрели хотя бы 10 тех же пользователей."
            similar = get(f"/movies/{movie['movie_id']}/{kind}", limit=10)
            if similar:
                st.caption("Нажмите на фильм в таблице, чтобы прочитать его описание.")
            selection = show_table(similar, columns, empty, key=f"{kind}_{movie['movie_id']}",
                                   on_select="rerun", selection_mode="single-cell")
            if selection and selection.selection.cells:
                row, _ = selection.selection.cells[0]
                selected_movie = similar[row]
                with selected_description.container():
                    st.subheader(f"Описание фильма «{selected_movie['title']}»", anchor=False)
                    st.write(selected_movie['overview'] or "Описания в каталоге нет.")

    # похожие пользователи
    elif section == "Похожие пользователи":
        st.title("Похожие пользователи", anchor=False)
        if enough_ratings(user_list):
            st.caption(f"До 10 пользователей с оценками, похожими на оценки профиля {user_id}")
            neighbors = get(f"/users/{user_id}/neighbors", limit=10)
            show_table([{**person, 'common_films': person['common']} for person in neighbors],
                       ['user_id', 'similarity', 'common_films', 'mean_rating'], "Похожие пользователи не найдены.")
            if neighbors:
                st.caption("Сходство выше, когда оба пользователя оценили общие фильмы выше или ниже своей средней. "
                           "Чем меньше общих фильмов, тем меньше вес совпадений.")
                own_mean = sum(film['rating'] for film in user_list) / len(user_list)
                for person in neighbors:
                    neighbor_id = person['user_id']
                    details = st.expander(f"Сравнить оценки с пользователем {neighbor_id}",
                                          key=f"compare_{user_id}_{neighbor_id}", on_change="rerun")
                    if details.open:
                        neighbor_ratings = get(f"/users/{neighbor_id}/ratings")
                        scores = {film['movie_id']: film['rating'] for film in neighbor_ratings}
                        own_column = f"Вы ({user_id})"
                        neighbor_column = f"ID {neighbor_id}"
                        comparison = [
                            {'title': film['title'], 'year': film['year'], own_column: film['rating'],
                             neighbor_column: scores[film['movie_id']]}
                            for film in user_list if film['movie_id'] in scores
                        ]
                        with details:
                            st.caption(f"Средняя по всем фильмам у профиля {user_id}: {own_mean:.2f}. "
                                       f"У профиля {neighbor_id}: {person['mean_rating']:.2f}.")
                            show_table(comparison, ['title', own_column, neighbor_column, 'year'],
                                       "Общих оценок нет.", column_config={
                                           'Фильм': st.column_config.TextColumn(width=145),
                                           own_column: st.column_config.NumberColumn(width=90),
                                           neighbor_column: st.column_config.NumberColumn(width=70)
                                       })
        else:
            st.write("Для поиска похожих пользователей нужно минимум 3 оценки и хотя бы 2 разные.")
        with st.expander(f"Лучшие оценки пользователя {user_id}"):
            show_table(user_list[:10], ['title', 'year', 'genres', 'rating'], "Оценок пока нет.")

    # персональная подборка
    elif section == "Персональные":
        st.title("Персональная подборка", anchor=False)
        st.caption(f"Прогноз оценок для профиля {user_id}")
        method = st.selectbox("Способ", ["По похожим пользователям", "По похожим фильмам"], key="personal_method")
        if enough_ratings(user_list):
            api_method = 'users' if method == "По похожим пользователям" else 'items'
            show_ranking(get(f"/users/{user_id}/recommendations", method=api_method, limit=10), 'predicted',
                         "Не хватает данных о похожих пользователях или фильмах.")
            st.caption("В истории Kion оценки получены из доли просмотра. Прогноз сглажен с учётом числа соседей.")
        else:
            st.write("Оцените хотя бы 3 фильма в разделе «Оценить фильм» и поставьте хотя бы 2 разные оценки.")

    # оценка фильма
    elif section == "Оценить фильм":
        st.title("Оценить фильм", anchor=False)
        if st.session_state.pop("rating_saved", False):
            st.success("Оценка сохранена. Персональные рекомендации обновлены.")
        movie = choose_movie("rate")
        if movie:
            movie_id = movie['movie_id']
            previous_score = next((film['rating'] for film in user_list if film['movie_id'] == movie_id), None)
            with st.form("movie_rating", border=False):
                score = st.pills("Оценка от 1 до 10", range(1, 11), default=previous_score,
                                 format_func=lambda value: f":material/star: {value}",
                                 key=f"rating_{user_id}_{movie_id}")
                save = st.form_submit_button("Сохранить оценку", type="primary")
            if save and score is None:
                st.write("Выберите оценку.")
            elif save:
                response = requests.post(API_URL + "/ratings", timeout=30,
                                         json={'user_id': user_id, 'movie_id': movie_id, 'rating': score})
                response.raise_for_status()
                st.session_state.rating_saved = True
                st.rerun()
        st.subheader("Оценки пользователя")
        show_table(user_list, ['title', 'rating', 'year', 'genres'], "Оценок пока нет.")

    # справка
    else:
        st.title("Справка", anchor=False)
        st.subheader("Разделы")
        st.markdown(
            "1. Популярное: топ фильмов по взвешенной оценке. Параметр: сколько фильмов показать, от 1 до 50.\n"
            "2. Персональные: 10 фильмов с самым высоким прогнозом оценки. Параметры: ID пользователя и способ.\n"
            "3. Похожие фильмы: подборка по контенту или по оценкам. Нажмите на фильм в таблице, чтобы открыть описание.\n"
            "4. Похожие пользователи: кнопки сравнения открывают оценки обоих профилей по общим фильмам.\n"
            "5. Оценить фильм: выберите оценку от 1 до 10 и сохраните её. Повторная оценка заменяет прежнюю, "
            "а персональная подборка учитывает любую сохранённую оценку."
        )

        st.subheader("Команды API")
        spec = requests.get(SERVER + "/openapi.json", timeout=30).json()
        rows = []
        for path, methods in spec['paths'].items():
            for http_method, info in methods.items():
                params = [describe_param(p['name'], p['schema'], p.get('description', '')) for p in info.get('parameters', [])]
                body = info.get('requestBody', {}).get('content', {}).get('application/json', {}).get('schema', {})
                if '$ref' in body:
                    model = spec['components']['schemas'][body['$ref'].split('/')[-1]]
                    params += [describe_param(name, field, field.get('description', '')) for name, field in model['properties'].items()]
                rows.append({'Команда': f"{http_method.upper()} {path}", 'Что делает': info.get('description', ''),
                             'Параметры': '; '.join(params) or 'нет'})
        st.table(pd.DataFrame(rows).set_index('Команда'))
        st.write(f"Все команды можно вызвать вручную на странице {SERVER}/docs.")

        st.subheader("Запуск")
        st.code("uvicorn main:app", language="bash")
        st.code("streamlit run app.py", language="bash")
except requests.exceptions.RequestException:
    st.error("Не удалось получить данные от API. Проверьте, что сервер запущен, и обновите страницу.")
