import sqlite3
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "recommender.sqlite3"

# те же параметры, что в ноутбуке
NEIGHBORS = 30
MIN_COMMON = 10
TEXT_WEIGHT = 0.7
SMOOTHING = 3


def divide_or_zero(numerator, denominator):
    # делим там, где знаменатель положительный, в остальных местах оставляем ноль
    result = np.zeros_like(numerator, dtype=np.float32)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def create_db(db_path):
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS movies (movie_id INTEGER PRIMARY KEY, title TEXT, genres TEXT, "
                   "year INTEGER, overview TEXT, text TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS ratings (user_id INTEGER, movie_id INTEGER REFERENCES movies(movie_id), "
                   "rating INTEGER CHECK (rating BETWEEN 1 AND 10), source TEXT DEFAULT 'app', "
                   "PRIMARY KEY (user_id, movie_id))")
        db.execute("CREATE TABLE IF NOT EXISTS personal_ratings (user_id INTEGER, "
                   "movie_id INTEGER REFERENCES movies(movie_id), rating INTEGER CHECK (rating BETWEEN 1 AND 10), "
                   "PRIMARY KEY (user_id, movie_id))")
        movie_count = db.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        if movie_count > 0:
            return
        # переносим каталог и оценки из csv в базу
        movies = pd.read_csv(DATA_DIR / "movies.csv").fillna('')
        ratings = pd.read_csv(DATA_DIR / "ratings.csv")
        ratings['source'] = 'kion'
        movies[['movie_id', 'title', 'genres', 'year', 'overview', 'text']].to_sql('movies', db, if_exists='append', index=False)
        ratings[['user_id', 'movie_id', 'rating', 'source']].to_sql('ratings', db, if_exists='append', index=False)


class Recommender:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        create_db(db_path)
        with sqlite3.connect(db_path) as db:
            self.movies = pd.read_sql("SELECT * FROM movies ORDER BY rowid", db)
            # для одной пары пользователь и фильм личная оценка важнее исходной
            ratings_query = (
                "SELECT user_id, movie_id, rating FROM personal_ratings "
                "UNION ALL "
                "SELECT user_id, movie_id, rating FROM ratings AS base "
                "WHERE NOT EXISTS (SELECT 1 FROM personal_ratings AS personal "
                "WHERE personal.user_id = base.user_id AND personal.movie_id = base.movie_id)"
            )
            ratings = pd.read_sql(ratings_query, db)
            self.last_dataset_user = db.execute("SELECT MAX(user_id) FROM ratings WHERE source = 'kion'").fetchone()[0]
        self.movie_ids = self.movies['movie_id'].tolist()
        self.movie_pos = {movie_id: i for i, movie_id in enumerate(self.movie_ids)}

        # матрица пользователь-фильм, 0 значит, что фильм не смотрели
        table = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')
        table = table.reindex(columns=self.movie_ids).fillna(0)
        self.user_ids = table.index.tolist()
        self.R = table.to_numpy(dtype=np.float32, copy=True)
        self.update_stats()

        # похожие фильмы по контенту: TF-IDF отдельно для описаний и жанров
        text_tfidf = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(self.movies['text'])
        self.genre_vectorizer = TfidfVectorizer(tokenizer=lambda s: s.split(', '), token_pattern=None, lowercase=False)
        genre_tfidf = self.genre_vectorizer.fit_transform(self.movies['genres'])
        self.text_sim = cosine_similarity(text_tfidf)
        self.genre_sim = cosine_similarity(genre_tfidf)

    def update_stats(self):
        ratings = self.R
        has_rating = ratings > 0
        movie_counts = has_rating.sum(axis=0)
        movie_totals = ratings.sum(axis=0)
        self.item_means = divide_or_zero(movie_totals, movie_counts)

        # похожие фильмы по оценкам: вычитаем среднюю каждого фильма и считаем косинус
        movie_vectors = np.where(has_rating, ratings - self.item_means, 0).astype(np.float32)
        movie_lengths = np.linalg.norm(movie_vectors, axis=0)
        movie_vectors = divide_or_zero(movie_vectors, movie_lengths)
        self.item_sim = movie_vectors.T @ movie_vectors
        rating_mask = has_rating.astype(np.float32)
        self.common = rating_mask.T @ rating_mask

        # для сходства пользователей вычитаем среднюю оценку пользователя
        user_counts = has_rating.sum(axis=1)
        self.user_means = ratings.sum(axis=1) / np.maximum(user_counts, 1)
        self.centered = np.where(has_rating, ratings - self.user_means[:, None], 0).astype(np.float32)

        # при малом числе оценок приближаем среднюю фильма к общей средней
        global_mean = ratings[has_rating].mean()
        rated_movies = movie_counts > 0
        prior_votes = np.quantile(movie_counts[rated_movies], 0.75)
        weighted_total = movie_counts * self.item_means + prior_votes * global_mean
        scores = weighted_total / (movie_counts + prior_votes)
        popularity = pd.DataFrame({
            'pos': np.arange(len(movie_counts)),
            'average': self.item_means,
            'votes': movie_counts,
            'score': scores,
        })
        self.popularity = popularity[rated_movies].sort_values('score', ascending=False)

    def movie(self, i):
        row = self.movies.iloc[i]
        return {'movie_id': int(row['movie_id']), 'title': row['title'], 'year': int(row['year']),
                'genres': row['genres'], 'overview': row['overview']}

    def get_pos(self, movie_id):
        if movie_id not in self.movie_pos:
            raise KeyError(f"Фильм {movie_id} не найден")
        return self.movie_pos[movie_id]

    def user_vector(self, user_id):
        if user_id in self.user_ids:
            return self.R[self.user_ids.index(user_id)]
        return np.zeros(len(self.movie_ids), dtype=np.float32)

    def status(self):
        return {'movies': len(self.movie_ids), 'users': len(self.user_ids), 'ratings': int((self.R > 0).sum()),
                'last_dataset_user': int(self.last_dataset_user)}

    def search(self, query, limit=20):
        query = query.strip().lower().replace('ё', 'е')
        if not query:
            return []
        titles = self.movies['title'].str.lower().str.replace('ё', 'е')
        found = self.movies.index[titles.str.contains(query, regex=False)]
        return [self.movie(i) for i in found[:limit]]

    # популярные фильмы
    def popular(self, limit=10):
        rows = self.popularity.head(limit)
        return [{**self.movie(int(row.pos)), 'average': round(float(row.average), 2), 'votes': int(row.votes),
                 'score': round(float(row.score), 2)} for row in rows.itertuples()]

    def similar_content(self, movie_id, limit=10):
        i = self.get_pos(movie_id)
        scores = TEXT_WEIGHT * self.text_sim[i] + (1 - TEXT_WEIGHT) * self.genre_sim[i]
        scores[i] = -1
        best = np.argsort(-scores, kind='stable')[:limit]
        return [{**self.movie(j), 'similarity': round(float(scores[j]), 3),
                 'text_similarity': round(float(self.text_sim[i, j]), 3),
                 'genre_similarity': round(float(self.genre_sim[i, j]), 3)} for j in best]

    def similar_by_ratings(self, movie_id, limit=10):
        i = self.get_pos(movie_id)
        scores = self.item_sim[i].copy()
        scores[i] = -1
        scores[self.common[i] < MIN_COMMON] = -1
        best = [j for j in np.argsort(-scores, kind='stable') if scores[j] > 0][:limit]
        return [{**self.movie(j), 'similarity': round(float(scores[j]), 3), 'common': int(self.common[i, j])}
                for j in best]

    def user_ratings(self, user_id):
        r = self.user_vector(user_id)
        rated = np.flatnonzero(r > 0)
        rated = sorted(rated, key=lambda j: (-r[j], self.movies['title'].iat[j]))
        return [{**self.movie(j), 'rating': int(r[j])} for j in rated]

    def enough_ratings(self, r):
        rated = r[r > 0]
        return len(rated) >= 3 and len(set(rated)) >= 2

    # похожие пользователи
    def user_similarity(self, user_id):
        ratings = self.user_vector(user_id)
        has_rating = ratings > 0
        user_mean = ratings[has_rating].mean()
        user_deviations = np.where(has_rating, ratings - user_mean, 0).astype(np.float32)
        user_mask = has_rating.astype(np.float32)
        all_masks = (self.R > 0).astype(np.float32)

        dot_products = self.centered @ user_deviations
        user_sum_squares = all_masks @ (user_deviations ** 2)
        other_sum_squares = (self.centered ** 2) @ user_mask
        length_products = np.sqrt(user_sum_squares * other_sum_squares)
        similarities = divide_or_zero(dot_products, length_products)

        common_counts = all_masks @ user_mask
        similarities = similarities * common_counts / (common_counts + 10)
        if user_id in self.user_ids:
            user_index = self.user_ids.index(user_id)
            similarities[user_index] = 0
        return ratings, similarities, common_counts

    def neighbors(self, user_id, limit=10):
        ratings = self.user_vector(user_id)
        if not self.enough_ratings(ratings):
            return []
        _, similarities, common_counts = self.user_similarity(user_id)
        order = np.argsort(-similarities, kind='stable')
        best = [i for i in order if similarities[i] > 0][:limit]
        result = []
        for i in best:


            result.append({
                'user_id': int(self.user_ids[i]),
                'similarity': round(float(similarities[i]), 3),
                'common': int(common_counts[i]),
                'mean_rating': round(float(self.user_means[i]), 2),
            })
        return result

    def predict_user_based(self, user_id):
        ratings, similarities, _ = self.user_similarity(user_id)
        order = np.argsort(-similarities, kind='stable')
        neighbors = order[similarities[order] > 0]
        neighbor_scores = similarities[neighbors]
        has_rating = self.R[neighbors] > 0

        within_limit = np.cumsum(has_rating, axis=0) <= NEIGHBORS
        selected = has_rating & within_limit
        weights = selected * neighbor_scores[:, None]

        weighted_deviations = (weights * self.centered[neighbors]).sum(axis=0)
        total_weights = weights.sum(axis=0)
        shift = divide_or_zero(weighted_deviations, total_weights)
        user_mean = ratings[ratings > 0].mean()
        predictions = user_mean + shift
        neighbor_counts = selected.sum(axis=0)
        return ratings, predictions, neighbor_counts

    # прогноз по похожим фильмам
    def predict_item_based(self, user_id):
        ratings = self.user_vector(user_id)
        rated_movies = np.flatnonzero(ratings > 0)
        weights = self.item_sim[:, rated_movies].copy()
        too_few_viewers = self.common[:, rated_movies] < MIN_COMMON
        weights[(weights < 0) | too_few_viewers] = 0

        if len(rated_movies) > NEIGHBORS:
            order = np.argsort(-weights, axis=1, kind='stable')
            outside_top = order[:, NEIGHBORS:]
            np.put_along_axis(weights, outside_top, 0, axis=1)

        deviations = ratings[rated_movies] - self.item_means[rated_movies]
        total_weights = weights.sum(axis=1)
        weighted_deviations = weights @ deviations
        shift = divide_or_zero(weighted_deviations, total_weights)
        predictions = self.item_means + shift
        neighbor_counts = (weights > 0).sum(axis=1)
        return ratings, predictions, neighbor_counts

    # персональные рекомендации
    def personal(self, user_id, method='users', limit=10):
        if not self.enough_ratings(self.user_vector(user_id)):
            return []
        if method == 'items':
            ratings, predictions, neighbor_counts = self.predict_item_based(user_id)
        else:
            ratings, predictions, neighbor_counts = self.predict_user_based(user_id)

        user_mean = ratings[ratings > 0].mean()
        baseline = (self.item_means + user_mean) / 2
        predictions = np.clip(predictions, 1, 10)
        weighted_total = neighbor_counts * predictions + SMOOTHING * baseline
        smoothed = weighted_total / (neighbor_counts + SMOOTHING)

        allowed = (ratings == 0) & (neighbor_counts >= 3)
        order = np.argsort(-smoothed, kind='stable')
        best = [j for j in order if allowed[j]][:limit]
        result = []
        for j in best:
            movie = self.movie(j)
            movie['predicted'] = round(float(smoothed[j]), 2)
            movie['neighbors'] = int(neighbor_counts[j])
            result.append(movie)
        return result

    # сохранение оценки и пересчёт рекомендаций
    def rate(self, user_id, movie_id, rating):
        j = self.get_pos(movie_id)
        with self.lock:
            with sqlite3.connect(self.db_path) as db:
                db.execute("INSERT INTO personal_ratings (user_id, movie_id, rating) VALUES (?, ?, ?) "
                           "ON CONFLICT(user_id, movie_id) DO UPDATE SET rating = excluded.rating",
                           (user_id, movie_id, rating))
            if user_id not in self.user_ids:
                self.user_ids.append(user_id)
                self.R = np.vstack([self.R, np.zeros((1, len(self.movie_ids)), dtype=np.float32)])
            self.R[self.user_ids.index(user_id), j] = rating
            self.update_stats()
