"""measure performance variations across different groups"""

import os
import json
import pickle

# from gensim.corpora import Dictionary
# from gensim.models import LdaModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import utils


def self_tokenizer(text):
    return text.split()


def build_lr(task_dic):
    """Obtain Test Results of prediction
    """
    doc_idx = 2
    result_path = './results/lr.' + task_dic['name']
    clf_path = './clf/lr.' + task_dic['name']
    vect_path = './vect/lr.' + task_dic['name']

    # pos features
    docs_pos = {}
    with open('../analysis/overlaps/pos/data_tags.tsv') as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            doc = json.loads(line[doc_idx])
            docs_pos[line[0]] = ' '.join([item[1] for item in doc])

    # topic features
    # dictp = '../analysis/overlaps/topic/twitter.dict'
    # tmodelp = '../analysis/overlaps/topic/twitter.model'
    # lda_dict = Dictionary.load(dictp)
    # lda = LdaModel.load(tmodelp)

    # build vectorizer
    print('Building vectorizer....', task_dic['name'])
    if os.path.exists(vect_path + '1'):
        vect = pickle.load(
            open(vect_path, 'rb')
        )
    else:
        vect = []
        docs = []
        with open(task_dic['datap']) as dfile:
            cols = dfile.readline().split('\t')
            idx = cols.index(task_dic['name'])

            for line in dfile:
                line = line.strip().split('\t')
                if line[idx] == 'x':
                    continue
                docs.append(line[doc_idx])

        vect.append(TfidfVectorizer(
            ngram_range=(1, 2), max_df=0.9,
            min_df=2, max_features=8000,
            stop_words=utils.stopwords()
        ))
        vect[0].fit(docs)

        pickle.dump(vect, open(vect_path, 'wb'))

    # build classifier
    print('Building classifier....', task_dic['name'])
    if os.path.exists(clf_path + '1'):
        clf = pickle.load(
            open(clf_path, 'rb')
        )
    else:
        # lda_feas = []
        with open(task_dic['train']) as dfile:
            # cols = dfile.readline().split('\t')
            data = {'x': [], 'x_pos': [], 'y': []}
            for line in dfile:
                line = line.strip().split('\t')
                data['x'].append(line[doc_idx])
                data['y'].append(int(line[-1]))
                data['x_pos'].append(docs_pos[line[0]])

        clf = LogisticRegression(class_weight='balanced', solver='liblinear')
        clf.fit(vect[0].transform(data['x']), data['y'])
        pickle.dump(clf, open(clf_path, 'wb'))

    # Testing
    print('Testing...', task_dic['name'])
    #    if not os.path.exists(result_path):
    docs = []
    test_pos = []
    # lda_feas = []
    with open(task_dic['test']) as dfile:
        dfile.readline()

        for line in dfile:
            line = line.strip().split('\t')
            docs.append(line[doc_idx])
            test_pos.append(docs_pos[line[0]])

    docs = vect[0].transform(docs)
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
        #    'info': None,
        # },
        # {
        #    'name': 'ethnicity',
        #    'datap': '../analysis/all_data_encoded.tsv',
        #    'train': '../split_data/ethnicity.train',
        #    'valid': '../split_data/ethnicity.valid',
        #    'test': '../split_data/ethnicity.test',
        #    'info': {1: [1,2,3], 0: [0]}, # binary mapping attributes
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
        build_lr(task)
