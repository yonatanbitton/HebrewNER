import io
import os
from copy import deepcopy

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imblearn_pipeline
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as sklearn_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from constants import LOGISTIC_REGRESSION_MODEL, PASSIVE_AGGRESSIVE_MODEL, DECISION_TREE_MODEL, SVM_MODEL, \
    RANDOM_FOREST_MODEL, NAIVE_BAYES_MODEL

number_of_found_word_vecs = 0
number_of_not_found_word_vecs = 0

class ClfModel:
    def __init__(self, model_type):
        self.model_type = model_type
        smt = SMOTE(random_state=42, k_neighbors=3)

        self.clf = self.init_normal_model(model_type, smt)

        self.classes_without_O = ['U-PERCENT', 'L-PERS', 'U-PERS', 'L-ORG', 'L-LOC', 'I-ORG', 'I-LOC', 'B-ORG', 'L-DATE', 'I-MONEY', 'B-MISC', 'L-MISC', 'L-MONEY', 'B-LOC', 'B-PERS', 'I-PERS', 'U-DATE', 'B-DATE', 'U-LOC', 'B-MONEY', 'U-MISC', 'I-MISC', 'I-DATE', 'L-PERCENT', 'I-TIME', 'U-ORG', 'L-TIME', 'B-PERCENT', 'B-TIME', 'U-TIME', 'I-PERCENT', 'U-MONEY' ]

    def init_normal_model(self, model_type, smt):
        if model_type in [LOGISTIC_REGRESSION_MODEL, PASSIVE_AGGRESSIVE_MODEL]:
            pipeline_type = imblearn_pipeline
        else:
            pipeline_type = sklearn_pipeline

        if model_type == LOGISTIC_REGRESSION_MODEL:
            classifier = LogisticRegression(C=55)
        elif model_type == RANDOM_FOREST_MODEL:
            classifier = RandomForestClassifier()
        elif model_type == DECISION_TREE_MODEL:
            classifier = DecisionTreeClassifier()
        elif model_type == SVM_MODEL:
            classifier = SVC(kernel='linear', C=1.2)
        elif model_type == NAIVE_BAYES_MODEL:
            # classifier = MultinomialNB() # Can't use with negative values
            classifier = GaussianNB()
        elif model_type == PASSIVE_AGGRESSIVE_MODEL:
            classifier = PassiveAggressiveClassifier(C=2, max_iter=1000, random_state=0, tol=1e-3)
        else:
            raise Exception("No such clf")

        if model_type in [LOGISTIC_REGRESSION_MODEL, PASSIVE_AGGRESSIVE_MODEL]:
            pipe = pipeline_type([('smt', smt), ('classifier', classifier)])
        else:
            pipe = pipeline_type([('classifier', classifier)])
        return pipe


    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        if self.model_type in [DECISION_TREE_MODEL, RANDOM_FOREST_MODEL]:
            tree.export_graphviz(self.clf.steps[0][1], out_file='resources' + os.sep + 'decision_tree.dot')

    def train_by_grid_search(self, X_train, y_train):
        if self.model_type in [RANDOM_FOREST_MODEL, DECISION_TREE_MODEL]:
            parameters = self.prepare_trees_grid_params()
        elif self.model_type == LOGISTIC_REGRESSION_MODEL:
            parameters = self.prepare_log_reg_grid_params()
        elif self.model_type == PASSIVE_AGGRESSIVE_MODEL:
            parameters = self.prepare_passive_aggressive_grid_params()
        elif self.model_type == SVM_MODEL:
            parameters = self.prepare_svm_grid_params()
        else:
            raise Exception("no such model")

        grid_search = GridSearchCV(self.clf, parameters, cv=5, n_jobs=-1, verbose=1)

        grid_search.fit(X_train, y_train)
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        print(best_parameters)
        self.clf = grid_search.best_estimator_

    def prepare_trees_grid_params(self):
        parameters = {
            'classifier__max_depth': [5, 10, 100],
            'classifier__max_features': [5, 10, 20],
            'classifier__min_samples_leaf': [3, 15, 50],
            'classifier__min_samples_split': [3, 15, 50],
            'classifier__n_estimators': [30, 100, 300, 1000]
        }
        return parameters


    def prepare_log_reg_grid_params(self):
        # params_list = [1, 10, 50, 100]
        params_list = [40, 45, 50, 55, 60]
        parameters = {
            'classifier__C': params_list
        }
        return parameters

    def prepare_svm_grid_params(self):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1, 2]
        parameters = {
            'classifier__C': Cs,
            'classifier__gamma': gammas
        }
        # Consider adding class weights
        return parameters


    def prepare_passive_aggressive_grid_params(self):
        parameters = {
            'classifier__C': [1, 2, 3, 4, 8, 16]
        }
        return parameters

    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        return y_pred

    def evaluate(self, y_true, y_pred):
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
        cross_tab = pd.crosstab(y_true, y_pred, rownames=['Real Label'], colnames=['Prediction'], margins=True)
        report = classification_report(y_true, y_pred, labels=self.classes_without_O, target_names=self.classes_without_O)
        report_with_O = classification_report(y_true, y_pred)
        return cross_tab, report, report_with_O


def get_data():
    data_path = 'resources' + os.sep + 'dataset_biluo.csv'
    df = pd.read_csv(data_path)
    y = df['BILUO']
    if str(y.iloc[len(y)-1]) == 'nan':
        y.iloc[len(y)-1] = 'O'
    df.drop(columns=['BILUO', 'Bio'], inplace=True)
    X = df
    return X, y

class FeatureExtractor(TransformerMixin):
    def __init__(self):
        self.gazzet_sets = self.load_gazzets()
        self.stop_words = self.load_stop_words()
        print("Loading Word Embeddings... Please wait...")
        model_path = 'resources' + os.sep + 'cc.he.300.vec'
        self.trained_model = self.load_word_embeddings(model_path)
        self.VECTOR_SIZE = 300
        print("FeatureExtractor initialized!")

    def load_word_embeddings(self, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            # data[tokens[0]] = map(float, tokens[1:])
            data[tokens[0]] = [float(x) for x in tokens[1:]]
        return data

    def load_stop_words(self):
        f = open("resources" + os.sep + "all_heb_stop_words.txt", 'r', encoding='utf-8')
        stop_words = [w.strip("\n") for w in f.readlines()]
        f.close()
        return stop_words

    def load_gazzets(self):
        f = open("resources" + os.sep + "naama_gazzets" + os.sep + "Dictionary.txt", 'r', encoding='utf-8')
        all_gazzets = f.readlines()
        f.close()
        gazzet_sets = {'LOC': set(), 'PERS': set(), 'ORG': set(), 'MONEY': set(), 'PERCENT': set()}
        for item in all_gazzets:
            for possibility in gazzet_sets.keys():
                if possibility in item:
                    item = item.replace(possibility, "").strip("\n").strip(" ")
                    if item != "":
                        gazzet_sets[possibility].add(item)
        print("Done loading gazzets")
        return gazzet_sets

    def transform(self, data):
        X = []
        all_features = {'Gender', 'Lemma', 'Number', 'Person', 'Pos', 'Status', 'Tense', 'Token', 'TokenOrder', 'Word', 'Prev_Prev_Pos', 'Prev_Pos', 'Next_Pos', 'Next_Next_Pos', 'Prev_Prev_Number', 'Prev_Number', 'Next_Number', 'Next_Next_Number', 'Prev_Prev_Gender', 'Prev_Gender', 'Next_Gender', 'Next_Next_Gender', 'Prev_Word', 'Next_Word', 'Prev_Token', 'Next_Token', 'In_LOC_Gazzet', 'In_PERS_Gazzet', 'In_ORG_Gazzet', 'In_MONEY_Gazzet',  'In_PERCENT_Gazzet', 'Suffix'}

        wanted_features = {'Prev_Pos', 'Next_Pos', 'Suffix', 'Prefix', 'TokenVector', 'NextTokenVector',
                           'PrevTokenVector', 'Person', 'Pos', 'In_MONEY_Gazzet', 'In_ORG_Gazzet', 'In_LOC_Gazzet',
                           'In_PERS_Gazzet', 'In_PERCENT_Gazzet', 'is_stop_word'} # Best subset - 77% f1-score

        for i in range(0, len(data)):
            prev_prev_row_data, prev_row_data, curr_row_data, next_row_data, next_next_row_data = \
                self.get_prev_curr_next_row_data(data, i)

            if 'TokenVector' in wanted_features:
                    self.add_word_vectors(curr_row_data)

            if 'NextTokenVector' in wanted_features and 'PrevTokenVector' in wanted_features:
                    self.add_word_vectors(prev_row_data)
                    self.add_word_vectors(next_row_data)

            self.add_context_features(curr_row_data, next_next_row_data, next_row_data, prev_prev_row_data,
                                      prev_row_data, wanted_features)
            self.add_gazzet_features(curr_row_data)

            curr_row_data['is_stop_word'] = curr_row_data['Token'] in self.stop_words

            for feat in all_features.difference(wanted_features):
                del curr_row_data[feat]

            if 'TokenVector' in wanted_features:
                if 'NextTokenVector' in wanted_features and 'PrevTokenVector' in wanted_features:
                    self.convert_vectors_to_features(prev_row_data, curr_row_data, next_row_data, include_contex=True)
                else:
                    self.convert_vectors_to_features(prev_row_data, curr_row_data, next_row_data, include_contex=False)

            X.append(curr_row_data)

        print("wanted_features")
        print(wanted_features)

        print(f"number_of_found_word_vecs: {number_of_found_word_vecs}")
        print(f"number_of_not_found_word_vecs: {number_of_not_found_word_vecs}")
        print(f"percent of words without vects: "
              f"{number_of_not_found_word_vecs / (number_of_found_word_vecs + number_of_not_found_word_vecs)}")

        df = pd.DataFrame(X)
        return df

    def add_word_vectors(self, curr_row_data):
        global number_of_found_word_vecs, number_of_not_found_word_vecs
        if curr_row_data['Token'] in self.trained_model:
            curr_row_data['TokenVector'] = self.trained_model[curr_row_data['Token']]
            number_of_found_word_vecs += 1
        else:
            curr_row_data['TokenVector'] = [float(0)] * 300
            number_of_not_found_word_vecs += 1

    def convert_vectors_to_features(self, prev_row_data, curr_row_data, next_row_data, include_contex):
            for i in range(self.VECTOR_SIZE):  # vector size
                curr_row_data['wordvec_' + str(i)] = curr_row_data['TokenVector'][i]
                if include_contex:
                    curr_row_data['next_wordvec_' + str(i)] = next_row_data['TokenVector'][i]
                    curr_row_data['prev_wordvec_' + str(i)] = prev_row_data['TokenVector'][i]

            del curr_row_data['TokenVector']
            if include_contex:
                del prev_row_data['TokenVector']
                del next_row_data['TokenVector']

    def add_gazzet_features(self, curr_row_data):
        for gazzet_key, gazzet_set in self.gazzet_sets.items():
            if curr_row_data['Word'] in gazzet_set or curr_row_data['Token'] in gazzet_set:
                curr_row_data['In_' + gazzet_key + '_Gazzet'] = True
                # print(curr_row_data['Word'], curr_row_data['Token'], " in gazzet: ", gazzet_key)
            else:
                curr_row_data['In_' + gazzet_key + '_Gazzet'] = False

    def add_context_features(self, curr_row_data, next_next_row_data, next_row_data, prev_prev_row_data, prev_row_data, wanted_features):
        curr_row_data['Prev_Prev_Pos'] = prev_prev_row_data['Pos']
        curr_row_data['Prev_Pos'] = prev_row_data['Pos']
        curr_row_data['Next_Pos'] = next_row_data['Pos']
        curr_row_data['Next_Next_Pos'] = next_next_row_data['Pos']

        curr_row_data['Prev_Prev_Number'] = prev_prev_row_data['Number']
        curr_row_data['Prev_Number'] = prev_row_data['Number']
        curr_row_data['Next_Number'] = next_row_data['Number']
        curr_row_data['Next_Next_Number'] = next_next_row_data['Number']

        curr_row_data['Prev_Prev_Gender'] = prev_prev_row_data['Gender']
        curr_row_data['Prev_Gender'] = prev_row_data['Gender']
        curr_row_data['Next_Gender'] = next_row_data['Gender']
        curr_row_data['Next_Next_Gender'] = next_next_row_data['Gender']

        curr_row_data['Prev_Word'] = prev_row_data['Word']
        curr_row_data['Next_Word'] = next_row_data['Word']

        curr_row_data['Prev_Token'] = prev_row_data['Token']
        curr_row_data['Next_Token'] = next_row_data['Token']

    def get_prev_curr_next_row_data(self, data, i):
        if i % 1000 == 0:
            print(i)
        if i <= 1:
            prev_prev_row = data.iloc[i]
            prev_row = data.iloc[i]
        else:
            prev_prev_row = data.iloc[i - 2]
            prev_row = data.iloc[i - 1]
        curr_row = data.iloc[i]
        if i >= len(data) - 2:
            next_row = data.iloc[i]
            next_next_row = data.iloc[i]
        else:
            next_row = data.iloc[i + 1]
            next_next_row = data.iloc[i + 2]
        prev_prev_row_data = dict(prev_prev_row)
        prev_row_data = dict(prev_row)
        curr_row_data = dict(curr_row)
        next_row_data = dict(next_row)
        next_next_row_data = dict(next_next_row)
        return prev_prev_row_data, prev_row_data, curr_row_data, next_row_data, next_next_row_data

    def fit(self, X, y=None):
        return self


def get_freqs(y_data):
    y_without_o = y_data[y_data.values != ['O']]
    y_freqs = y_without_o.value_counts().apply(lambda x: x / y_without_o.value_counts().sum())
    return y_freqs


def add_missing_columns(all_y_cols, y_data):
    diff_train = set(all_y_cols.keys()).difference(set(y_data.keys()))
    if len(diff_train) > 0:
        for col in diff_train:
            y_data[col] = 0
    return y_data


def check_frequencies_of_labels_in_data(y, y_train, y_test):
    y_freqs = get_freqs(y)
    y_train_freqs = get_freqs(y_train)
    y_test_freqs = get_freqs(y_test)
    y_train_freqs = add_missing_columns(y_freqs, y_train_freqs)
    y_test_freqs = add_missing_columns(y_freqs, y_test_freqs)
    print("We got frequencies of labels in y, y_train, y_test :-) ")
    y_freqs.sum(), y_train_freqs.sum(), y_test_freqs.sum()
    compare_df = pd.DataFrame(columns=y_freqs.keys())
    compare_df.keys = ['y', 'y_train', 'y_test']
    compare_df.loc['y'] = y_freqs
    compare_df.loc['y_train'] = y_train_freqs
    compare_df.loc['y_test'] = y_test_freqs
    return compare_df

def check_which_df_col_contains_null(df):
    print(df.columns[df.isna().any()].tolist())

def save_dataset_as_csv(X_train, X_test, y_train, y_test):
    print(f"Saving... X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
    X_train['label'] = list(y_train)
    X_test['label'] = list(y_test)
    X_train.to_csv("resources" + os.sep + "train_data.csv", index=False)
    X_test.to_csv("resources" + os.sep + "test_data.csv", index=False)
    print("Saved dataset at resources directory :-)")

def load_local_dataset():
    train_data = pd.read_csv("resources" + os.sep + "train_data.csv")
    test_data = pd.read_csv("resources" + os.sep + "test_data.csv")
    if 'Suffix' in train_data:
        train_data['Suffix'].fillna('"', inplace=True)
    y_train = train_data['label']
    y_test = test_data['label']
    train_data.drop(columns=['label'], inplace=True)
    test_data.drop(columns=['label'], inplace=True)

    print("Checking if X_train contains null")
    check_which_df_col_contains_null(train_data)
    print("Checking if X_test contains null")
    check_which_df_col_contains_null(test_data)
    return train_data, test_data, y_train, y_test


def main():
    LOAD_READY_DATASET = True
    model_type = SVM_MODEL
    print(f"model_type: {model_type}")
    print(f"LOAD_READY_DATASET: {LOAD_READY_DATASET}")

    if LOAD_READY_DATASET:
        X_train, X_test, y_train, y_test = load_local_dataset()
        print("Loaded dataset")

    else:
        X, y = get_data()
        dummies_cols = ['Person', 'Pos', 'Prev_Pos', 'Next_Pos', 'Suffix', 'Prefix']

        feature_extractor = FeatureExtractor()
        X_transformed = feature_extractor.transform(X)
        X_final = make_dataset_with_dummies(X_transformed, dummies_cols)
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.20, shuffle=True)
        compare_df = check_frequencies_of_labels_in_data(y, y_train, y_test)
        print(compare_df)
        print("Curr cols: ")
        print(X_train.columns)
        save_dataset_as_csv(deepcopy(X_train), deepcopy(X_test), deepcopy(y_train), deepcopy(y_test))

    print("Shapes:")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    print(f"training {model_type} Model")
    clf_model = ClfModel(model_type=model_type)
    clf_model.train(X_train, y_train)
    # clf_model.train_by_grid_search(X_train, y_train)

    y_pred = clf_model.predict(X_test=X_test)
    cross_tab, report, report_with_O = clf_model.evaluate(y_true=y_test, y_pred=y_pred)
    print(f"Results for {model_type} model")
    print(report)
    print(report_with_O)
    print(cross_tab)
    #
    # if clf_model.model_type in {DECISION_TREE_MODEL, RANDOM_FOREST_MODEL}:
    #     print_tree_feature_importance(X_train, clf_model)
    #
    # print("Trying again with exploiting previous tags")
    # X_train_with_tag, tree_model_with_tags = retrain_with_exploit_previous_tags(X_test, X_train, y_test, y_train, model_type)
    #
    # if clf_model.model_type in {'decision_tree', 'random_forest'}:
    #     print_tree_feature_importance(X_train_with_tag, tree_model_with_tags)


def make_dataset_with_dummies(X_transformed, dummies_cols):
    print(f"shape before dummies: {X_transformed.shape}")
    X_dummies = pd.get_dummies(X_transformed[dummies_cols])
    X_transformed = X_transformed.drop(columns=dummies_cols)
    X_final = pd.concat([X_transformed, X_dummies], axis=1)
    print(f"X_dummies.shape: {X_dummies.shape}, X_transformed.shape: {X_transformed.shape}, X_final.shape: {X_final.shape}")
    return X_final


def retrain_with_exploit_previous_tags(X_test, X_train, y_test, y_train, clf_type):
    X_train_with_tag, y_train = make_train_data_with_tags(X_train, clf_type, y_train)

    clf_model_with_tags = ClfModel(model_type=clf_type)
    clf_model_with_tags.train(X_train_with_tag, y_train)

    X_test_with_tag = init_prev_tag_dummy_variables_for_test_data_like_the_train(X_test, X_train_with_tag)

    new_y_pred = loop_of_predict_with_previous_tag(X_test_with_tag, clf_model_with_tags)

    cross_tab, report, report_with_O = clf_model_with_tags.evaluate(y_true=y_test, y_pred=new_y_pred)
    print(f"Results for model {clf_type}")
    print("reports without O")
    print(report)
    print("report_with_O")
    print(report_with_O)
    print(cross_tab)
    return X_train_with_tag, clf_model_with_tags


def loop_of_predict_with_previous_tag(X_test_with_tag, clf_model_with_tags):
    X_test_with_tag.loc[X_test_with_tag.index[0], 'prev_tag_O'] = 1
    new_y_pred = []
    for i in range(0, len(X_test_with_tag)):
        curr_df_to_predict = pd.DataFrame(X_test_with_tag.iloc[i]).T
        pred = clf_model_with_tags.predict(X_test=curr_df_to_predict)[0]
        if i + 1 < len(X_test_with_tag):
            X_test_with_tag.loc[X_test_with_tag.index[i + 1], 'prev_tag_' + pred] = 1
        new_y_pred.append(pred)
    return new_y_pred


def init_prev_tag_dummy_variables_for_test_data_like_the_train(X_test, X_train_with_tag):
    X_test_with_tag = deepcopy(X_test)
    all_prev_tag_train_dummies_cols = [col for col in X_train_with_tag.columns if col.startswith("prev_tag")]
    for col in all_prev_tag_train_dummies_cols:
        X_test_with_tag[col] = 0
    return X_test_with_tag


def make_train_data_with_tags(X_train, clf_type, y_train):
    X_train_with_tag = deepcopy(X_train)
    X_train_with_tag['prev_tag'] = ['O'] + list(y_train)[:-1]
    prev_tag_dummies = pd.get_dummies(X_train_with_tag[['prev_tag']])
    X_train_with_tag = pd.concat([X_train, prev_tag_dummies], axis=1, sort=False)
    return X_train_with_tag, y_train


def print_tree_feature_importance(X_train, tree_model):
    print("Tree feature importance:")
    feature_importance_d = {}
    for feat, importance in zip(list(X_train.columns), tree_model.clf.steps[0][1].feature_importances_):
        feature_importance_d[feat] = importance
        # print ('feature: {f}, importance: {i}'.format(f=feat, i=importance))
    for key, value in sorted(feature_importance_d.items(), key=lambda kv: kv[1], reverse=True)[:25]:
        print(f"feature: {key}, importance: {value}")

def produce_x_test_df_with_predictions(clf_model, X_test, X, y, y_pred):
    X_test['y_pred'] = y_pred
    original_x_test_only = X.loc[X_test.index]
    original_y_test_only = y.loc[X_test.index]
    original_x_test_only['y_pred'] = X_test['y_pred']
    original_x_test_only['y_test'] = original_y_test_only
    print("Sanity check, report scores should be the same")
    _, report, _= clf_model.evaluate(y_true=original_x_test_only['y_test'],
                                                              y_pred=original_x_test_only['y_pred'])
    print(report)
    original_x_test_only.to_csv("resources" + os.sep + "X_test_with_predictions.csv", index=False)


if __name__ == '__main__':
    main()
