"""This method implements blindness method in the paper of Counterfactual Fairness in Text Classification through
Robustness """

import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import utils


def replace_words(doc, replace):
    # replace numbers
    doc = re.sub('^[0-9]+', 'number', doc)
    # replace words
    doc = [word if word not in replace else 'identity' for word in doc.split()]

    return ' '.join(doc)


def build_lr_blind(task_dic):
    # load the replacement words
    with open('./resource/replace.txt') as dfile:
        replaces = set()
        for line in dfile:
            # only use unigram
            if len(line.split(' ')) > 1:
                continue

            replaces.add(line.strip())

    doc_idx = 2
    result_path = './results/lr_blind.' + task_dic['name']
    vect_path = './vect/lr_blind.' + task_dic['name']
    clf_path = './clf/lr_blind.' + task_dic['name']

    # build vectorizer
    print('Building vectorizer....', task_dic['name'])
    if os.path.exists(vect_path):
        vect = pickle.load(
            open(vect_path, 'rb')
        )
    else:
        docs = []
        with open(task_dic['datap']) as dfile:
            cols = dfile.readline().split('\t')
            idx = cols.index(task_dic['name'])

            for line in dfile:
                line = line.strip().split('\t')
                if line[idx] == 'x':
                    continue

                # replace the words and numbers into IDENTITY and NUMBER
                docs.append(replace_words(line[doc_idx], replaces))

        vect = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2, max_features=8000, stop_words=utils.stopwords()
        )
        vect.fit(docs)

        pickle.dump(vect, open(vect_path, 'wb'))

    # build classifier
    print('Building classifier....', task_dic['name'])
    if os.path.exists(clf_path):
        clf = pickle.load(
            open(clf_path, 'rb')
        )
    else:
        with open(task_dic['train']) as dfile:
            # cols = dfile.readline().split('\t')
            data = {'x': [], 'y': []}
            for line in dfile:
                line = line.strip().split('\t')

                # replace the words and numbers into IDENTITY and NUMBER
                data['x'].append(replace_words(line[doc_idx], replaces))
                data['y'].append(int(line[-1]))

        clf = LogisticRegression(class_weight='balanced', solver='liblinear')
        clf.fit(vect.transform(data['x']), data['y'])
        pickle.dump(clf, open(clf_path, 'wb'))

    # Testing
    print('Testing...', task_dic['name'])
    #    if not os.path.exists(result_path):
    docs = []
    with open(task_dic['test']) as dfile:
        dfile.readline()

        for line in dfile:
            line = line.strip().split('\t')
            docs.append(line[doc_idx])

    docs = vect.transform(docs)
    y_pred = clf.predict(docs)
    y_prob = clf.predict_proba(docs)

    with open(task_dic['test']) as dfile:
        with open(result_path, 'w') as wfile:
            wfile.write(dfile.readline().strip() + '\tpred\tpred_prob\n')

            for idx, line in enumerate(dfile):
                wfile.write(line.strip() + '\t' + str(y_pred[idx]) + '\t' + str(y_prob[idx][1]) + '\n')

    utils.fair_eval(result_path)


if __name__ == '__main__':
    task_list = [
        # {
        #    'name': 'gender',
        #    'datap': '../analysis/all_data_encoded.tsv',
        #    'train': '../split_data/gender.train',
        #    'valid': '../split_data/gender.valid',
        #    'test': '../split_data/gender.test',
        # },
        # {
        #    'name': 'ethnicity',
        #    'datap': '../analysis/all_data_encoded.tsv',
        #    'train': '../split_data/ethnicity.train',
        #    'valid': '../split_data/ethnicity.valid',
        #    'test': '../split_data/ethnicity.test',
        # },
        # {
        #    'name': 'age',
        #    'datap': '../analysis/all_data_encoded.tsv',
        #    'train': '../split_data/age.train',
        #    'valid': '../split_data/age.valid',
        #    'test': '../split_data/age.test',
        #    'info': None
        # },
        # {
        #    'name': 'country',
        #    'datap': '../analysis/all_data_encoded.tsv',
        #    'train': '../split_data/country.train',
        #    'valid': '../split_data/country.valid',
        #    'test': '../split_data/country.test',
        #    'info': None
        # },
        # {
        #    'name': 'region',
        #    'datap': '../analysis/all_data_encoded.tsv',
        #    'train': '../split_data/region.train',
        #    'valid': '../split_data/region.valid',
        #    'test': '../split_data/region.test',
        #    'info': {1: [2,3], 0: [0,1]}, # binary mapping attributes
        # },
        {
            'name': 'ethMulti',
            'datap': '../analysis/all_data_encoded.tsv',
            'train': '../split_data/ethMulti.train',
            'valid': '../split_data/ethMulti.valid',
            'test': '../split_data/ethMulti.test',
            # 'info': {1: [1,2,3], 0: [0]}, # multi-attributes
        },

    ]
    for task in task_list:
        build_lr_blind(task)
