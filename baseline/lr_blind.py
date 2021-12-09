"""This method implements blindness method in the paper of Counterfactual Fairness in Text Classification through
Robustness

Gender sensitive words come from https://github.com/conversationai/unintended-ml-bias-analysis
"""

import os
import pickle
import datetime
from tqdm import tqdm
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from nltk.corpus import stopwords
import nltk
from imblearn.over_sampling import RandomOverSampler
import numpy as np

import utils
nltk.download('stopwords')


def replace_words(doc, replace):
    # replace numbers
    doc = re.sub('^[0-9]+', 'number', doc)
    # replace words
    doc = [word if word not in replace else 'identity' for word in doc.split()]

    return ' '.join(doc)


def build_lr_blind(params):
    # load the replacement words
    with open('../resources/lexicons/replace_{}.txt'.format(params['lang'])) as dfile:
        replaces = set()
        for line in dfile:
            # only use unigram
            if len(line.split(' ')) > 1:
                continue

            replaces.add(line.strip())

    print('Loading Data...')
    data = utils.data_loader(dpath=params['dpath'], lang=params['lang'])
    print('Building Domain Vectorizer...')

    # build vectorizer
    vect_path = os.path.join(params['model_dir'], params['dname'] + '-lr_vect.pkl')
    if os.path.exists(vect_path):
        lr_vect = pickle.load(open(vect_path, 'rb'))
    else:
        try:
            spw_set = set(stopwords.words(params['lang']))
        except OSError:
            spw_set = None
        lr_vect = TfidfVectorizer(
            min_df=3, max_features=params['max_feature'],
            stop_words=spw_set, max_df=0.9, ngram_range=(1, 3),
        )
        lr_vect.fit(data)
        pickle.dump(lr_vect, open(vect_path, 'wb'))
    train_indices, val_indices, test_indices = utils.data_split(data)

    # train classifier
    input_data = {
        'docs': [replace_words(data['docs'][item], replaces) for item in train_indices],
        'labels': [data['labels'][item] for item in train_indices],
    }
    if params['over_sample']:
        ros = RandomOverSampler(random_state=33)
        sample_indices = list(range(len(input_data['docs'])))
        sample_indices, _ = ros.fit_resample(sample_indices, input_data['labels'])
        input_data = {
            'docs': [input_data['docs'][item] for item in sample_indices],
            'labels': [input_data['labels'][item] for item in sample_indices],
        }

    # too large data to fit memory, remove some
    # training data size: 200000
    if len(input_data['docs']) > 200000:
        np.random.seed(33)
        indices = list(range(len(input_data['docs'])))
        np.random.shuffle(indices)
        indices = indices[:200000]
        input_data = {
            'docs': [input_data['docs'][item] for item in indices],
            'labels': [input_data['labels'][item] for item in indices],
        }

    print('Training Classifier...')
    input_feats = lr_vect.transform(input_data['docs'])
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(input_feats, input_data['labels'])

    # load test
    print('Loading Test data')
    input_data = {
        'docs': [replace_words(data['docs'][item], replaces) for item in test_indices],
        'labels': [data['labels'][item] for item in test_indices],
        params['domain_name']: [input_data[params['domain_name']][item] for item in test_indices],
    }

    print('Testing.............................')
    input_feats = lr_vect.transform_test(input_data['docs'])
    pred_label = clf.predict(input_feats)
    fpr, tpr, _ = metrics.roc_curve(
        y_true=input_data['labels'], y_score=clf.predict_proba(input_feats)[:, 1],
    )

    with open(params['result_path'], 'a') as wfile:
        wfile.write('{}...............................\n'.format(datetime.datetime.now()))
        wfile.write('Performance Evaluation for the task: {}\n'.format(params['dname']))
        wfile.write('F1-weighted score: {}\n'.format(
            metrics.f1_score(y_true=input_data['labels'], y_pred=pred_label, average='weighted')
        ))
        wfile.write('AUC score: {}\n'.format(
            metrics.auc(fpr, tpr)
        ))
        wfile.write(metrics.classification_report(
            y_true=input_data['labels'], y_pred=pred_label, digits=3) + '\n')
        wfile.write('\n')

        wfile.write('Fairness Evaluation\n')
        wfile.write(
            utils.fair_eval(
                true_labels=input_data['labels'],
                pred_labels=pred_label,
                domain_labels=input_data[params['domain_name']]
            ) + '\n'
        )

        wfile.write('...............................\n\n')
        wfile.flush()


if __name__ == '__main__':
    review_dir = '../data/review/'
    hate_speech_dir = '../data/hatespeech/'
    model_dir = '../resources/model/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = model_dir + os.path.basename(__file__) + '/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    result_dir = '../resources/results/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    data_list = [
        # ['review_amazon_english', review_dir + 'amazon/amazon.tsv', 'english'],
        # ['review_yelp-hotel_english', review_dir + 'yelp_hotel/yelp_hotel.tsv', 'english'],
        # ['review_yelp-rest_english', review_dir + 'yelp_rest/yelp_rest.tsv', 'english'],
        # ['review_twitter_english', review_dir + 'twitter/twitter.tsv', 'english'],
        ['review_trustpilot_english', review_dir + 'trustpilot/united_states.tsv', 'english'],
        ['review_trustpilot_french', review_dir + 'trustpilot/france.tsv', 'french'],
        ['review_trustpilot_german', review_dir + 'trustpilot/german.tsv', 'german'],
        ['review_trustpilot_danish', review_dir + 'trustpilot/denmark.tsv', 'danish'],
        ['hatespeech_twitter_english', hate_speech_dir + 'English/corpus.tsv', 'english'],
        ['hatespeech_twitter_spanish', hate_speech_dir + 'Spanish/corpus.tsv', 'spanish'],
        ['hatespeech_twitter_italian', hate_speech_dir + 'Italian/corpus.tsv', 'italian'],
        ['hatespeech_twitter_portuguese', hate_speech_dir + 'Portuguese/corpus.tsv', 'portuguese'],
        ['hatespeech_twitter_polish', hate_speech_dir + 'Polish/corpus.tsv', 'polish'],
    ]

    for data_entry in tqdm(data_list):
        print('Working on: ', data_entry)

        parameters = {
            'result_path': os.path.join(result_dir, os.path.basename(__file__) + '.txt'),
            'model_dir': model_dir,
            'dname': data_entry[0],
            'dpath': data_entry[1],
            'lang': data_entry[2],
            'max_feature': 10000,
            'over_sample': False,
        }

        build_lr_blind(parameters)
