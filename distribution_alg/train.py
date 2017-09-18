import os
import scipy
import cPickle
import xgboost as xgb
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss
from imblearn.combine import SMOTEENN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer

"""
- Predict the storage distribution:
  storage = f(X),
   X = [L, P, T, C],
   storage -> 5 classes
  
  -- Multi-class classification
  -- log-loss
  
  1. Encode categorical features -> L, P -> save encoder 
  2. text features (T, C) -> TfidfVectorizer -> save vectorizer
  3. decrease text vectorizer dimension with SVD
  4. LinearSVC + KFold -> meta-features, model
  5. meta-features + xgboost / RF
  
- Predict the popularity:
  popularity = f(X),
    X = [N, V, D, T, C, SS, L, P],
    popularity -> {0, 1} e.g. popular or not
"""

# Categorical vars
L = "Language"
P = "Privacy"
SS = "storage"

# Numerical vars
D = "duration_min"
V = "Views"
N = "lifetime"

# Text vars
T = "Video Title"
C = "Channel"

METRIC = "logloss"

encoders = {}


def train(X, y, args):
    models = {
        'xgboost': {
            'cls': xgb.XGBClassifier(),
            'params': {
                'eta': [0.01, 0.015, 0.025, 0.05, 0.1],  # learning rate
                'nthread': 6,
                'max_depth': [3, 5, 7, 9],
                'num_class': len(np.unique(y)),
                'objective': 'multi:softprob',
                'min_child_weight': [1, 3, 5, 7],
                'alpha': 0,  # l1 regularization
                'lambda': [0.01, 0.1, 0.5, 1],  # l2 regularization
                'silent': 1,
                'eval_metric': METRIC
            }
        },
        'svm': {
            'cls': LinearSVC(),
            'params': {
                'C': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1],
                'penalty': ['l1', 'l2']
            }
        }
    }

    print "[ ] Over- and under- sampling imbalanced samples per class.."
    print "[ ] Original counts per class - {}".format(sorted(Counter(y).items()))
    smoteenn = SMOTEENN()

    X_resampled, y_resampled = smoteenn.fit_sample(X, y)
    print "[+] Sampled counts per class - {}".format(sorted(Counter(y_resampled).items()))

    print "[ ] Stratified 3-fold split of samples.."
    sample_shuffler = StratifiedKFold(n_splits=3, shuffle=True)
    train_index, test_index = next(iter(sample_shuffler.split(X_resampled, y_resampled)))
    X_train, y_train = X_resampled[train_index], y_resampled[train_index]
    X_test, y_test = X_resampled[test_index], y_resampled[test_index]
    print "[+] Train size - {}, Test size - {}".format(X_train.shape, X_test.shape)

    for model_name, data in models.iteritems():
        try:
            print "[ ] Training %s" % model_name
            grid_search_cv = GridSearchCV(data['cls'],
                                          param_grid=data['params'],
                                          n_jobs=1, scoring=METRIC, refit=True, verbose=1)
            if model_name == "xgboost":
                print "[!] Building DMatrix for XGBoost.."
                xg_train = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
                xg_test = xgb.DMatrix(X_test, y_test, feature_names=X_test.columns.values)
                tqdm(grid_search_cv.fit(xg_train, xg_test))
            else:
                tqdm(grid_search_cv.fit(X_train, y_train))

            cPickle.dump(grid_search_cv, open(os.path.join(os.path.dirname(args.data), "grid-search-cv-%s.pkl" % model_name)),
                         cPickle.HIGHEST_PROTOCOL)
        except Exception, ex:
            print "[-] Error for {} -\n{}".format(model_name, ex)
            continue


def main(args):
    df = pd.read_csv(args.data, error_bad_lines=False)

    y_encoder = LabelEncoder()
    y_vals = list(df[SS].values)
    y_encoder.fit(y_vals)
    encoders.update({SS: y_encoder})
    y = y_encoder.transform(y_vals).astype(np.float64)

    # 1. Normalize categorical features
    X_label_encoded = []
    for cat_field in [L, P]:
        label_encoder = LabelEncoder()
        label_encoder.fit(df.loc[:, cat_field].values)
        X_labels = label_encoder.transform(df.loc[:, cat_field])

        ohe = OneHotEncoder()
        ohe.fit(X_labels)
        X_label_encoded.append(ohe.transform(X_labels))
        encoders.update({"%s_label_encoder" % cat_field: label_encoder})
        encoders.update({"%s_ohe" % cat_field: ohe})

    # 2. Vectorize text features
    text_series = pd.concat([df[C], df[T]]).reset_index(name='text')['text']
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    tfidf.fit(text_series)
    text_matrix = tfidf.transform(text_series)
    encoders['%s_text_vectorizer'] = tfidf
    if args.reduce_feat_dim:
        # 2.1. Reduce text features dimensionality
        tsvd = TruncatedSVD(n_components=120)
        tsvd.fit(text_matrix)
        text_matrix_svd = tsvd.transform(text_matrix)
        text_matrix = []
        text_matrix = text_matrix_svd
        encoders['%s_text_vect_svd'] = tsvd

    # Stack features
    # import ipdb;ipdb.set_trace()
    X = scipy.sparse.hstack([text_matrix, X_label_encoded[0], X_label_encoded[1]], format='csr')
    # X = df[[L, P, C, T]]

    if args.ready_to_dump:
        # Dump encoders
        [cPickle.dump(encoder, open(os.path.join(os.path.dirname(args.data), "%s.pkl" % name)),
                      cPickle.HIGHEST_PROTOCOL) for name, encoder in encoders.iteritems()]

    train(X, y, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Filepath to the dataset")
    parser.add_argument("--reduce_feat_dim", help="Boolean flag in order to reduce features dim", default=False)
    parser.add_argument("--ready_to_dump", help="Boolean flag in order to dump encoders / models", default=False)

    args = parser.parse_args()
    main(args)
