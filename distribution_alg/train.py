import os
import glob
import cPickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from tqdm import tqdm
from scipy import sparse
from collections import Counter
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import log_loss
from imblearn.combine import SMOTEENN
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
"""

# Categorical vars
L = "Language"
P = "Privacy"
SS = "storage"

# Text vars
T = "Video Title"
C = "Channel"

METRIC = "logloss"

encoders = {}


# temporal option to ignore DeprecationWarnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def train(X, y, args):
    # samples of classes in the given dataset is imbalanced
    print "[ ] Over- and under- sampling imbalanced samples per class.."
    print "[ ] Original counts per class - {}".format(sorted(Counter(y).items()))
    smoteenn = SMOTEENN()

    X_dense = X.toarray()  # imbalance requires dense matrix rather than sparse
    X_resampled, y_resampled = smoteenn.fit_sample(X_dense, y)
    print "[+] Sampled counts per class - {}".format(sorted(Counter(y_resampled).items()))

    print "[ ] Stratified 3-fold split of samples.."
    sample_shuffler = StratifiedKFold(n_splits=3, shuffle=True)
    train_index, test_index = next(iter(sample_shuffler.split(X_resampled, y_resampled)))
    X_train, y_train = X_resampled[train_index], y_resampled[train_index]
    X_test, y_test = X_resampled[test_index], y_resampled[test_index]
    print "[+] Train size - {}, Test size - {}".format(X_train.shape, X_test.shape)

    if args.param_opt:
        # greedy param tuning (long process)
        models = {
            'xgboost': {
                'cls': xgb.XGBClassifier(),
                'params': {
                    'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],  # learning rate
                    'nthread': [6],
                    'max_depth': [3, 5, 7, 9],
                    'num_class': [len(np.unique(y))],
                    'objective': ['multi:softprob'],
                    'n_estimators': [100],
                    'min_child_weight': [1, 3, 5, 7],
                    'reg_alpha': [0],  # l1 regularization
                    'reg_lambda': [0.01, 0.1, 0.5, 1]  # l2 regularization
                }
            },
            'svm': {
                'cls': SVC(kernel='linear', probability=True),
                'params': {
                    'C': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
                }
            }
        }
        for model_name, data in models.iteritems():
            try:
                print "[ ] Training %s" % model_name
                grid_search_cv = GridSearchCV(data['cls'], param_grid=data['params'],
                                              n_jobs=1, scoring="neg_log_loss", refit=True, verbose=1)
                grid_search_cv.fit(X_train, y_train)

                cPickle.dump(grid_search_cv,
                             open(os.path.join(os.path.dirname(args.data), "grid-search-cv-%s.pkl" % model_name)),
                             cPickle.HIGHEST_PROTOCOL)
            except Exception, ex:
                print "[-] Error for {} -\n{}".format(model_name, ex)
                continue
    else:
        for model in (xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100,
                                        objective='multi:softprob'),
                      xgb.XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100,
                                        objective='multi:softprob')):
                      # SVC(C=1.0, kernel='linear', probability=True),
                      # SVC(C=0.5, kernel='linear', probability=True)):
            print model
            model.fit(X_train, y_train)
            y_hypo = model.predict_proba(X_test)
            log_loss_score = log_loss(y_test, y_hypo)
            print "[+] {} model gave {} logloss".format(model.__class__.__name__, log_loss_score)


def main(args):
    df = pd.read_csv(args.data, error_bad_lines=False)

    # read dumped encoders if they exist
    pwd_dir = os.path.dirname(args.data)

    for path in glob.glob(os.path.join(pwd_dir, "encoders")):
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as f:
            filename = os.path.basename(path).split('.')[0]
            encoders[filename] = cPickle.load(f)
            f.close()

    # 0. y vector encoder
    y_vals = list(df[SS].values)
    if SS in encoders:
        y_encoder = encoders[SS]
    else:
        y_encoder = LabelEncoder()
        y_encoder.fit(y_vals)
        encoders.update({SS: y_encoder})
    y = y_encoder.transform(y_vals).astype(np.float64)

    # 1. Normalize categorical features
    X_label_encoded = []
    for cat_field in [L, P]:
        encoder_name = "%s_label_encoder" % cat_field
        oh_encoder_name = "%s_ohe" % cat_field

        if encoder_name in encoders:
            label_encoder = encoders[encoder_name]
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(df.loc[:, cat_field].values)
            encoders.update({encoder_name: label_encoder})

        X_labels = label_encoder.transform(df.loc[:, cat_field])

        if oh_encoder_name in encoders:
            ohe = encoders[oh_encoder_name]
        else:
            ohe = OneHotEncoder()
            ohe.fit(X_labels)
            encoders.update({oh_encoder_name: ohe})

        X_label_encoded.append(ohe.transform(X_labels))

    # 2. Vectorize text features
    X_text_encoded = []
    for text_field in [T, C]:
        text_vectorizer_name = '%s_text_vectorizer' % text_field
        text_vectorizer_svd_name = '%s_text_vect_svd' % text_field

        if text_vectorizer_name in encoders:
            tfidf = encoders[text_vectorizer_name]
        else:
            tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                    sublinear_tf=True, use_idf=False, smooth_idf=False, lowercase=True)
            tfidf.fit(df[text_field])
            encoders[text_vectorizer_name] = tfidf

        text_matrix = tfidf.transform(df[text_field])

        # 2.1. Reduce text features dimensionality
        if args.reduce_feat_dim:
            if text_vectorizer_svd_name in encoders:
                tsvd = encoders[text_vectorizer_svd_name]
            else:
                tsvd = TruncatedSVD(n_components=120)
                tsvd.fit(text_matrix)
                encoders[text_vectorizer_svd_name] = tsvd
            text_matrix_svd = tsvd.transform(text_matrix)
            X_text_encoded.append(text_matrix_svd)
        else:
            X_text_encoded.append(text_matrix)

    # 3. Stack features
    # Transposing label_encoders and hstack them with other features
    X = sparse.hstack([X_label_encoded[0].T, X_label_encoded[1].T,
                       X_text_encoded[0], X_text_encoded[1], ], format='csr')

    if args.ready_to_dump:
        dump_dir = os.path.join(os.path.dirname(args.data), "encoders")
        if not os.path.isdir(dump_dir):
            try:
                os.mkdir(dump_dir)
            except Exception, ex:
                print "[-] Could not create {0} dir, error: {1}".format(dump_dir, ex)

        # Dump encoders
        [cPickle.dump(encoder, open(os.path.join(dump_dir, "%s.pkl" % name), 'wb+'),
                      cPickle.HIGHEST_PROTOCOL) for name, encoder in encoders.iteritems()]
    train(X, y, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Filepath to the dataset")
    parser.add_argument("--reduce_feat_dim", help="Boolean flag in order to reduce features dim", default=False)
    parser.add_argument("--ready_to_dump", help="Boolean flag in order to dump encoders / models", default=False)
    parser.add_argument("--param_opt", help="Boolean flag in order to optimize parameters in models"
                                            " with greedy search (long process)", default=False)
    args = parser.parse_args()
    main(args)
