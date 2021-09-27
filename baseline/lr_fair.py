"""This uses Generized glove vector,
    then average the word representations to obtain document representation,
    Test by logistic regression.
"""

import os
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

import utils


def build_lr_fair(task_dic):
    doc_idx = 2
    emb_size = 300
    clf_path = './clf/lr_fair.' + task_dic['name']
    result_path = './results/lr_fair.' + task_dic['name']
    embed_path = '../embeddings/fair/' + \
                 'GoogleNews-vectors-negative300-hard-debiased.txt'
    vect_path = './vect/lr_fair.vect'

    # load the embedding as vectorizer
    print('Building vectorizer...', task_dic['name'])
    if os.path.exists(vect_path):
        embeds = pickle.load(open(vect_path, 'rb'))
    else:
        # obtain the unique words, 
        # filter out unnecessary word embeddings
        uniq_words = set()
        with open(task_dic['datap']) as dfile:
            dfile.readline()
            for line in dfile:
                line = line.strip().split('\t')[doc_idx]
                uniq_words.update(line.split())

        embeds = dict()
        with open(embed_path) as dfile:
            dfile.readline()
            for line in dfile:
                line = line.split()
                if line[0] not in uniq_words:
                    continue

                embeds[line[0]] = [float(item) for item in line[1:]]
        pickle.dump(embeds, open(vect_path, 'wb'))

    # load and encode the data
    print('Training...', task_dic['name'])
    if os.path.exists(clf_path):
        clf = pickle.load(open(clf_path, 'rb'))
    else:
        data = {'x': [], 'y': []}
        with open(task_dic['train']) as dfile:
            dfile.readline()
            for line in dfile:
                line = line.strip().split('\t')
                tmp_vect = [0.0] * emb_size
                count = 0
                for word in line[doc_idx].split():
                    if word in embeds:
                        count += 1
                        tmp_vect = np.add(tmp_vect, embeds[word])
                if count != 0:
                    tmp_vect = np.divide(tmp_vect, count)

                data['x'].append(tmp_vect)
                data['y'].append(int(line[-1]))

        # start to train the classifier
        data['x'] = np.asarray(data['x'])
        clf = LogisticRegression(class_weight='balanced', solver='liblinear')
        clf.fit(data['x'], data['y'])
        pickle.dump(clf, open(clf_path, 'wb'))

    # Testing
    print('Testing...', task_dic['name'])
    #    if not os.path.exists(result_path):
    docs = []
    with open(task_dic['test']) as dfile:
        dfile.readline()

        for line in dfile:
            line = line.strip().split('\t')
            tmp_vect = [0.0] * emb_size
            count = 0

            for word in line[doc_idx].split():
                if word in embeds:
                    count += 1
                    tmp_vect = np.add(tmp_vect, embeds[word])

            if count != 0:
                tmp_vect = np.divide(tmp_vect, count)
            docs.append(tmp_vect)

    docs = np.asarray(docs)
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
        #        {
        #            'name': 'gender',
        #            'datap': '../analysis/all_data_encoded.tsv',
        #            'train': '../split_data/gender.train',
        #            'valid': '../split_data/gender.valid',
        #            'test': '../split_data/gender.test',
        #            'info': None,
        #        },
        #        {
        #            'name': 'ethnicity',
        #            'datap': '../analysis/all_data_encoded.tsv',
        #            'train': '../split_data/ethnicity.train',
        #            'valid': '../split_data/ethnicity.valid',
        #            'test': '../split_data/ethnicity.test',
        #            'info': {1: [1,2,3], 0: [0]}, # binary mapping attributes
        #        },
        #        {
        #            'name': 'age',
        #            'datap': '../analysis/all_data_encoded.tsv',
        #            'train': '../split_data/age.train',
        #            'valid': '../split_data/age.valid',
        #            'test': '../split_data/age.test',
        #            'info': None
        #        },
        #        {
        #            'name': 'country',
        #            'datap': '../analysis/all_data_encoded.tsv',
        #            'train': '../split_data/country.train',
        #            'valid': '../split_data/country.valid',
        #            'test': '../split_data/country.test',
        #            'info': None
        #        },
        #        {
        #            'name': 'region',
        #            'datap': '../analysis/all_data_encoded.tsv',
        #            'train': '../split_data/region.train',
        #            'valid': '../split_data/region.valid',
        #            'test': '../split_data/region.test',
        #            'info': {1: [2,3], 0: [0,1]}, # binary mapping attributes
        #        },
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
        build_lr_fair(task)
