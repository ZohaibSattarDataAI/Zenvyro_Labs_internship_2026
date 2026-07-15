"""
mlJudge.py
----------
A small "ML judge" that scores debate arguments on a 1-10 persuasiveness
scale using a RandomForestRegressor trained on a mock historical dataset.

Pipeline:
    raw text -> extract_NLP_features() -> numeric feature vector
             -> RandomForestRegressor -> predicted persuasiveness score

Feature choices (why these three):
    - word_count: longer, more developed arguments tend to correlate with
      higher perceived persuasiveness, up to a point of diminishing
      returns — the model learns that relationship from data rather than
      us hard-coding it.
    - complexity_score: average word length, used as a cheap proxy for
      vocabulary sophistication / rhetorical complexity without needing
      a heavyweight NLP library.
    - sentiment: a lightweight polarity score (via NLTK's VADER, since it
      is fast and works well on short, punchy debate-style text). Falls
      back to 0.0 if NLTK/VADER isn't installed so the app never crashes
      just because an optional dependency is missing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# VADER sentiment is optional (bonus feature). We degrade gracefully if
# it isn't installed rather than blowing up the whole judge.
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None


FEATURE_COLUMNS = ["word_count", "complexity_score", "sentiment"]


class DebateRegressionJudge:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def extract_NLP_features(self, text: str) -> dict:
        """
        Converts raw debate text into the numeric feature dict the
        regression model expects.
        """
        words = text.split()
        word_count = len(words)

        # Average word length as a stand-in for "complexity". Guard
        # against empty text to avoid a divide-by-zero.
        if word_count > 0:
            complexity_score = sum(len(w.strip(".,!?;:\"'")) for w in words) / word_count
        else:
            complexity_score = 0.0

        if _VADER is not None and text.strip():
            sentiment = _VADER.polarity_scores(text)["compound"]
        else:
            sentiment = 0.0

        return {
            "word_count": word_count,
            "complexity_score": round(complexity_score, 3),
            "sentiment": round(sentiment, 3),
        }

    def train_model(self, dataset_path: str) -> dict:
        """
        Trains a RandomForestRegressor on historical_debates.csv.
        Expected columns: word_count, complexity_score, sentiment,
        human_persuasiveness_score.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Training dataset not found at '{dataset_path}'. "
                "Run generate_mock_dataset() first or supply your own CSV."
            )

        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)

        missing = [c for c in FEATURE_COLUMNS + ["human_persuasiveness_score"] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        X = df[FEATURE_COLUMNS].values
        y = df["human_persuasiveness_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # RandomForest over a plain LinearRegression: debate persuasiveness
        # is unlikely to be a purely linear function of these features
        # (e.g. word_count probably helps up to a point, then hurts), and
        # a small forest handles that non-linearity without much tuning.
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        predictions = self.model.predict(X_test)
        mse = float(mean_squared_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        print(f"Model trained! MSE={mse:.4f}, R2={r2:.4f}")
        return {"mse": round(mse, 4), "r2_score": round(r2, 4)}

    def predict_score(self, text: str) -> float:
        """
        Called live during the debate to score a single agent's argument.
        Returns a float clamped to the 1-10 range.
        """
        if self.model is None:
            raise Exception("Model is not trained yet! Call train_model() first.")

        features = self.extract_NLP_features(text)
        X = np.array([[features[col] for col in FEATURE_COLUMNS]])

        raw_score = float(self.model.predict(X)[0])
        clamped = max(1.0, min(10.0, raw_score))
        return round(clamped, 2)


def generate_mock_dataset(path: str = "historical_debates.csv", n_rows: int = 200, seed: int = 42):
    """
    Builds a synthetic historical_debates.csv so the pipeline is runnable
    out of the box without needing a real labeled dataset.

    The synthetic label is a plausible (not perfect) function of the
    features plus noise, so the trained model has a real pattern to find
    rather than pure randomness.
    """
    rng = np.random.default_rng(seed)

    word_count = rng.integers(15, 120, size=n_rows)
    complexity_score = rng.uniform(3.0, 7.5, size=n_rows)
    sentiment = rng.uniform(-1.0, 1.0, size=n_rows)

    # Persuasiveness rises with length (up to a point), rewards moderate
    # complexity, and rewards a somewhat positive/assertive tone.
    base = (
        3.0
        + 0.03 * word_count
        - 0.0002 * (word_count ** 1.5)
        + 0.4 * complexity_score
        + 1.5 * sentiment
    )
    noise = rng.normal(0, 0.6, size=n_rows)
    human_persuasiveness_score = np.clip(base + noise, 1, 10)

    df = pd.DataFrame({
        "word_count": word_count,
        "complexity_score": np.round(complexity_score, 2),
        "sentiment": np.round(sentiment, 2),
        "human_persuasiveness_score": np.round(human_persuasiveness_score, 2),
    })
    df.to_csv(path, index=False)
    print(f"Mock dataset written to {path} ({n_rows} rows).")
    return path


if __name__ == "__main__":
    # Quick standalone sanity check: generate data, train, predict.
    generate_mock_dataset()
    judge = DebateRegressionJudge()
    metrics = judge.train_model("historical_debates.csv")
    print(metrics)
    print(judge.predict_score("This is a strong, confident argument in favor of the motion!"))