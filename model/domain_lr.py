# domain adaption + adversarial training
import os
import pickle
import datetime

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix, csc_matrix, hstack
import numpy as np
from imblearn.over_sampling import RandomOverSampler

import utils


def da_tokenizer(text):
    return text.split()


class DomainVectorizer(TransformerMixin):
    def __init__(self, params):
        self.params = params
        self.uniq_domains = None
        self.tfidf_vec_da = None
        self.use_large = self.params.get('use_large', False)

    def fit(self, dataset):
        """

        :param dataset: a dictionary with x, y, and domain labels
        :return:
        """
        # if len(dataset) > 15469:  # this number is length of "./yelp/yelp_Hotels_year_sample.tsv"
        #     self.use_large = True
        print('start to fit')
        spw_set = set(stopwords.words(self.params['lang']))
        self.uniq_domains = sorted(
            np.unique([item for item in dataset[self.params['domain_name']] if item != 'docs']))
        self.tfidf_vec_da = dict.fromkeys(self.uniq_domains)

        if not self.use_large:
            for key in self.uniq_domains:
                print('Domain:' + str(key))
                self.tfidf_vec_da[key] = TfidfVectorizer(
                    ngram_range=(1, 3), min_df=2, max_features=self.params['max_feature'],
                    stop_words=spw_set, max_df=0.9, tokenizer=da_tokenizer
                )
                new_docs = []
                for idx, item in enumerate(dataset[self.params['domain_name']]):
                    if item == key:
                        new_docs.append(dataset['docs'])
                self.tfidf_vec_da[key].fit(new_docs)
            self.tfidf_vec_da["general"] = TfidfVectorizer(
                ngram_range=(1, 3), min_df=2, max_features=self.params['max_feature'],
                stop_words=spw_set, max_df=0.9, tokenizer=da_tokenizer
            )
            self.tfidf_vec_da["general"].fit(dataset['docs'])
        else:
            for key in self.tfidf_vec_da:
                print('Domain:' + str(key))
                self.tfidf_vec_da[key] = os.path.join(self.params['model_dir'], str(key) + '.pkl')
                tmp_vect = TfidfVectorizer(
                    min_df=3, tokenizer=da_tokenizer, max_features=self.params['max_feature'],
                    stop_words=spw_set, max_df=0.9, ngram_range=(1, 3),
                )
                tmp_vect.fit(
                    [item for idx, item in enumerate(dataset['docs'])
                     if dataset[self.params['domain_name']][idx] == key]
                )
                pickle.dump(tmp_vect, open(self.tfidf_vec_da[key], 'wb'))

            tmp_vect = TfidfVectorizer(
                min_df=3, tokenizer=da_tokenizer, max_features=self.params['max_feature'],
                stop_words=spw_set, max_df=0.9, ngram_range=(1, 3),
            )
            self.tfidf_vec_da["general"] = os.path.join(self.params['model_dir'], 'general.pkl')
            tmp_vect.fit(dataset['docs'])
            pickle.dump(tmp_vect, open(self.tfidf_vec_da['general'], 'wb'))
        return self

    def transform(self, dataset):
        fvs = csc_matrix(np.zeros(shape=(len(dataset['docs']), 1)))

        if not self.use_large:
            for domain in self.uniq_domains:
                tmp_fvs = csc_matrix(self.tfidf_vec_da[domain].transform(
                    [item if dataset[self.params['domain_name']] == domain else ""
                     for idx, item in enumerate(dataset['docs'])]
                ))
                fvs = hstack([fvs, tmp_fvs])
            fvs = fvs[:, 1:]
            tmp_fvs = csc_matrix(self.tfidf_vec_da['general'].transform(dataset['docs']))
            fvs = hstack([fvs, tmp_fvs])
        else:
            for domain in self.uniq_domains:
                dm_vect = pickle.load(open(self.tfidf_vec_da[domain], 'rb'))
                tmp_fvs = csc_matrix(dm_vect.transform(dataset['docs']))

                fvs = hstack([fvs, tmp_fvs])
            fvs = fvs[:, 1:]

            dm_vect = pickle.load(open(self.tfidf_vec_da['general'], 'rb'))
            tmp_fvs = csc_matrix(dm_vect.transform(dataset['docs']))
            fvs = hstack([fvs, tmp_fvs])
        return fvs

    def transform_test(self, docs):  # test data only keeps the general features
        fvs = csc_matrix((
            len(docs),
            sum([len(self.tfidf_vec_da[domain].vocabulary_) for domain in self.uniq_domains]))
        )
        # only for the small data
        tmp_fvs = csc_matrix(self.tfidf_vec_da['general'].transform(docs))
        fvs = hstack([fvs, tmp_fvs])
        return fvs


def domain_lr(params):
    data = utils.data_loader(dpath=params['dpath'], lang=params['lang'])
    da_vect = DomainVectorizer(params)
    da_vect.fit(data)
    train_indices, val_indices, test_indices = utils.data_split(data)

    # train classifier
    input_data = {
        'docs': [data['docs'][item] for item in train_indices],
        'labels': [data['labels'][item] for item in train_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in train_indices],
    }
    if params['over_sample']:
        ros = RandomOverSampler(random_state=33)
        sample_indices = list(range(len(input_data['docs'])))
        sample_indices, _ = ros.fit_resample(sample_indices, input_data['labels'])
        input_data = {
            'docs': [input_data['docs'][item] for item in sample_indices],
            'labels': [input_data['labels'][item] for item in sample_indices],
            params['domain_name']: [input_data[params['domain_name']][item] for item in sample_indices],
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
            params['domain_name']: [input_data[params['domain_name']][item] for item in indices],
        }

    input_feats = da_vect.transform(input_data['docs'])
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(input_feats, input_data['labels'])

    # parameter tuning
    general_len = -1 * len(da_vect.tfidf_vec_da['general'].vocabulary_)
    best_lambda = 1
    best_valid = 0
    lambda_list = [0.3, 1, 10, 30, 100, 300]
    print('Loading Valid data')
    input_data = {
        'docs': [data['docs'][item] for item in val_indices],
        'labels': [data['labels'][item] for item in val_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in val_indices],
    }
    print('Transforming valid data....................')
    input_feats = da_vect.transform_test(input_data['docs'])
    # for using only general features
    input_feats = lil_matrix(input_feats)
    # because the general features were appended finally, previous features are all domain features.
    input_feats[:, :general_len] = 0

    for lambda_item in lambda_list:
        exp_data = input_feats * lambda_item
        pred_label = clf.predict(exp_data)
        report_da = metrics.f1_score(y_true=input_data['labels'], y_pred=pred_label, average='weighted')
        if report_da > best_valid:
            best_valid = report_da
            best_lambda = lambda_item

    # load test
    print('Loading Test data')
    input_data = {
        'docs': [data['docs'][item] for item in test_indices],
        'labels': [data['labels'][item] for item in test_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in test_indices],
    }

    print('Transforming test data....................')
    input_feats = da_vect.transform_test(input_data['docs'])
    # for using only general features
    input_feats = lil_matrix(input_feats)
    input_feats[:, :general_len] = 0
    input_feats = input_feats * best_lambda

    print('Testing.............................')
    pred_label = clf.predict(input_feats)
    fpr, tpr, _ = metrics.roc_curve(
        y_true=input_data['labels'], y_score=clf.predict_proba(input_feats),
    )

    with open(params['result_path'], 'w') as wfile:
        wfile.write('{}...............................\n'.format(datetime.datetime.now()))
        wfile.write('Performance Evaluation\n')
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
            )+'\n'
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
        ['hatespeech_twitter_english', hate_speech_dir + 'english/corpus.tsv', 'english'],
        ['hatespeech_twitter_spanish', hate_speech_dir + 'spanish/corpus.tsv', 'spanish'],
        ['hatespeech_twitter_italian', hate_speech_dir + 'italian/corpus.tsv', 'italian'],
        ['hatespeech_twitter_portuguese', hate_speech_dir + 'portuguese/corpus.tsv', 'portuguese'],
        ['hatespeech_twitter_polish', hate_speech_dir + 'polish/corpus.tsv', 'polish'],
    ]

    for data_entry in data_list:
        print('Working on: ', data_entry)

        parameters = {
            'result_path': os.path.join(result_dir, os.path.basename(__file__)+'.txt'),
            'model_dir': model_dir,
            'dname': data_entry[0],
            'dpath': data_entry[1],
            'lang': data_entry[2],
            'max_feature': 10000,
            'use_large': False,
            'domain_name': 'gender',
            'over_sample': False,
        }

        domain_lr(parameters)
