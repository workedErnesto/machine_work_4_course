from typing import Literal
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field
from recommender import Recommender

app = FastAPI(
    title="Movie Recommender API",
    description="API рекомендательной системы фильмов на данных Kion",
    version="1.0.0"
)

engine = Recommender()


class RatingItem(BaseModel):
    user_id: int = Field(..., ge=1, example=1642, description="ID пользователя, который ставит оценку")
    movie_id: int = Field(..., example=9728, description="ID фильма из поиска")
    rating: int = Field(..., ge=1, le=10, example=8, description="Оценка от 1 до 10, повторная оценка заменяет прежнюю")


def find_movie(func, movie_id, limit):
    try:
        return func(movie_id, limit)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0])


@app.get("/health", summary="Проверка работы", description="Возвращает ok, если сервер запущен.")
def healthcheck():
    return {"status": "ok"}


@app.get("/api/status", summary="Размер данных",
         description="Число фильмов, пользователей и оценок, последний ID пользователя из датасета.")
def status():
    with engine.lock:
        return engine.status()


@app.get("/api/search", summary="Поиск фильма", description="Ищет фильмы по части названия.")
def search(q: str = Query(..., min_length=1, description="Часть названия фильма"),
           limit: int = Query(20, ge=1, le=50, description="Сколько фильмов вернуть")):
    return engine.search(q, limit)


@app.get("/api/popular", summary="Популярные фильмы",
         description="Топ фильмов по взвешенной оценке, учитывает среднюю оценку и число оценок.")
def popular(limit: int = Query(10, ge=1, le=50, description="Сколько фильмов вернуть")):
    with engine.lock:
        return engine.popular(limit)


@app.get("/api/movies/{movie_id}/similar", summary="Похожие фильмы по контенту",
         description="Фильтрация на контенте: TF-IDF описаний и жанров, косинусное сходство, описание 70%, жанры 30%.")
def similar_content(movie_id: int = Path(..., description="ID фильма"),
                    limit: int = Query(10, ge=1, le=50, description="Сколько фильмов вернуть")):
    return find_movie(engine.similar_content, movie_id, limit)


@app.get("/api/movies/{movie_id}/similar-by-ratings", summary="Похожие фильмы по оценкам",
         description="Коллаборативная фильтрация по фильмам: косинусное сходство центрированных оценок, "
                     "не меньше 10 общих пользователей.")
def similar_by_ratings(movie_id: int = Path(..., description="ID фильма"),
                       limit: int = Query(10, ge=1, le=50, description="Сколько фильмов вернуть")):
    with engine.lock:
        return find_movie(engine.similar_by_ratings, movie_id, limit)


@app.get("/api/users/{user_id}/ratings", summary="Оценки пользователя",
         description="Все фильмы, которые оценил пользователь, от высокой оценки к низкой.")
def user_ratings(user_id: int = Path(..., ge=1, description="ID пользователя")):
    with engine.lock:
        return engine.user_ratings(user_id)


@app.get("/api/users/{user_id}/neighbors", summary="Похожие пользователи",
         description="Коллаборативная фильтрация по пользователям. Нужно минимум 3 оценки и хотя бы 2 разные.")
def neighbors(user_id: int = Path(..., ge=1, description="ID пользователя"),
              limit: int = Query(10, ge=1, le=30, description="Сколько пользователей вернуть")):
    with engine.lock:
        return engine.neighbors(user_id, limit)


@app.get("/api/users/{user_id}/recommendations", summary="Персональные рекомендации",
         description="Фильмы, которые пользователь ещё не оценил, со сглаженным прогнозом. "
                     "Нужно минимум 3 оценки и хотя бы 2 разные.")
def recommendations(user_id: int = Path(..., ge=1, description="ID пользователя"),
                    method: Literal['users', 'items'] = Query('users', description="users: по похожим пользователям, items: по похожим фильмам"),
                    limit: int = Query(10, ge=1, le=50, description="Сколько фильмов вернуть")):
    with engine.lock:
        return engine.personal(user_id, method, limit)


@app.post("/api/ratings", summary="Оценить фильм",
          description="Сохраняет оценку в базу и сразу пересчитывает рекомендации.")
def rate(item: RatingItem):
    try:
        engine.rate(item.user_id, item.movie_id, item.rating)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0])
    return {"message": "Оценка сохранена"}
