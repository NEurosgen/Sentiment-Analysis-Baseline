from typing import Literal, Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

VectorizerName = Literal["bow", "tfidf"]
def build_pipeline(vectorizer: VectorizerName= "bow",ngram_range=(1, 2),max_iter= 1000,random_state= 42,class_weight= None):
    if vectorizer == "bow":
        vect = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=ngram_range,
            min_df=1
        )
    elif vectorizer == "tfidf":
        vect = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=ngram_range,
            min_df=1
        )
    else:
        raise ValueError("vectorizer must be 'bow' or 'tfidf'")
    clf = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight
    )
    pipe = Pipeline([
        ("vect", vect),
        ("clf", clf),
    ])
    return pipe
