"""
Implement methods of:
Demographics Should Not Be the Reason of Toxicity:
Mitigating Discrimination in Text Classifications with Instance Weighting

Some codes adopted from https://github.com/ghzhang233/Non-Discrimination-Learning-for-Text-Classification
"""
import os
import argparse
import datetime

from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Embedding, GRU, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from keras_preprocessing.sequence import pad_sequences

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import class_weight

import numpy as np
from sklearn import metrics
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

import utils

# cpu would be faster than using the GPU with original implementation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def data_gen(docs, labels, weights=None, batch_size=64):
    """
        Batch generator
    """
    if weights is not None:
        data_indices = list(range(len(docs)))
        np.random.shuffle(data_indices)  # random shuffle the training documents
        docs = [docs[idx] for idx in data_indices]
        labels = [labels[idx] for idx in data_indices]
        weights = [weights[idx] for idx in data_indices]

    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for step in range(steps):
        batch_docs = []
        batch_labels = []
        batch_weights = []

        for idx in range(step * batch_size, (step + 1) * batch_size):
            if idx > len(docs) - 1:
                break
            batch_docs.append(np.asarray(docs[idx]))
            batch_labels.append(labels[idx])
            if weights is not None:
                batch_weights.append(weights[idx])

        # convert to array
        batch_docs = np.asarray(batch_docs)
        batch_labels = np.asarray(batch_labels)
        batch_weights = np.asarray(batch_weights)

        yield batch_docs, batch_labels, batch_weights


def get_model(embedding,
              num_lstm=1,
              num_hidden=1,
              dim_hidden=128,
              num_classes=2,
              max_seq_len=35,
              dropout_rate=0.5,
              lr=1e-3,
              clipping=5.0):
    model_in = Input(shape=(max_seq_len,), dtype='int32')
    embedding_layer = Embedding(embedding.shape[0],
                                embedding.shape[1],
                                mask_zero=False,
                                weights=[embedding],
                                trainable=False,
                                input_length=max_seq_len)
    hidden = embedding_layer(model_in)

    for j in range(num_lstm):
        rnn_cell = GRU
        lstm_layer = rnn_cell(dim_hidden, return_sequences=(j != num_lstm - 1))
        hidden = lstm_layer(hidden)

    for _ in range(num_hidden):
        hidden = Dense(dim_hidden, activation='relu')(hidden)
        hidden = BatchNormalization()(hidden)
    hidden = Dropout(dropout_rate)(hidden)

    if num_classes < 3:
        model_out = Dense(1, activation='softmax')(hidden)
    else:
        model_out = Dense(num_classes, activation='softmax')(hidden)

    ret_model = Model(model_in, model_out)
    ret_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr, clipnorm=clipping))

    return ret_model


def get_tfidf(docs, sensitive_words):
    tf = np.zeros([len(docs), len(sensitive_words)])
    idf = np.zeros(len(sensitive_words))
    tfidf = np.zeros([len(docs), len(sensitive_words)])
    idxs_sens = []
    for i in range(len(docs)):
        for j in range(len(sensitive_words)):
            tf[i, j] = sum([(1 if word == sensitive_words[j] else 0) for word in docs[i].split()])
            if tf[i, j] > 0:
                idf[j] += 1
        if tf[i].sum() > 0:
            idxs_sens.append(i)
    for i in range(len(sensitive_words)):
        idf[i] = np.log(len(idxs_sens) / (idf[i] + 1))
    for i in range(len(docs)):
        for j in range(len(sensitive_words)):
            tfidf[i, j] = tf[i, j] * idf[j]
    return tfidf, idxs_sens


def make_weights(params):
    sensitive_words = list()
    with open('../resources/lexicons/replace_{}.txt'.format(params['lang'])) as dfile:
        for line in dfile:
            sensitive_words.append(line.strip().lower())
    sensitive_words = list(set(sensitive_words))
    data = utils.data_loader(dpath=params['dpath'], lang=params['lang'])
    params['unique_domains'] = np.unique(data[params['domain_name']])

    train_indices, val_indices, test_indices = utils.data_split(data)
    train_data = {
        'docs': [data['docs'][item] for item in train_indices + val_indices],
        'labels': [data['labels'][item] for item in train_indices + val_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in train_indices + val_indices],
    }

    sensitive_z, idxs_sens = get_tfidf(train_data['docs'], sensitive_words)
    sensitive_labels = np.asarray(train_data['labels'])[idxs_sens]
    # obtaining the weights
    clf = RandomForestClassifier(n_estimators=1000, max_depth=27, random_state=233, n_jobs=-1, criterion='entropy')
    y_pred = cross_val_predict(
        clf, sensitive_z[idxs_sens], sensitive_labels,
        cv=5, n_jobs=-1, method='predict_proba'
    )
    # print('Refit log loss: %.5f' % (log_loss(sensitive_z[idxs_sens], y_pred[:, 1])))

    p1 = sum(sensitive_labels) / len(sensitive_labels)
    p0 = 1 - p1
    print(roc_auc_score(to_categorical(sensitive_labels), y_pred))
    print(accuracy_score(sensitive_labels, np.argmax(y_pred, 1)), max(p0, p1))

    p1 = sum(train_data['labels']) / len(train_data['labels'])
    p0 = 1 - p1
    propensity = np.array([
        (p1 if train_data['labels'][i] == 1 else p0) for i in range(len(train_data['labels']))])
    propensity[idxs_sens] = np.array([
        y_pred[i, train_data['labels'][idxs_sens[i]]] for i in range(len(idxs_sens))])
    # normalize the propensity
    propensity = np.asanyarray([item if item != 0 else 2. for item in propensity])
    np.save(params['model_dir'] + "propensity_%s.npy" % params['dname'], propensity)
    # propensity = np.load(dir_processed + "propensity_%s.npy" % name_dataset)

    weights = 1 / propensity
    a = np.mean(
        np.array([weights[i] for i in range(len(weights)) if train_data['labels'][i] == 0]))
    b = np.mean(
        np.array([weights[i] for i in range(len(weights)) if train_data['labels'][i] == 1]))
    weights = np.array([
        (weights[i] / a if train_data['labels'][i] == 0 else weights[i] / b) for i in range(len(weights))])
    weights /= weights.mean()

    ret = np.zeros(len(data['docs']))
    ret[train_indices + val_indices] = weights
    np.save(params['model_dir'] + "weights_{}.npy".format(params['dname']), ret)


def build_weight(params):
    print('Loading Data...')
    data = utils.data_loader(dpath=params['dpath'], lang=params['lang'])
    debias_weights = np.load(params['model_dir'] + "weights_{}.npy".format(params['dname']))

    # build tokenizer and weight
    tok_dir = os.path.dirname(params['dpath'])
    params['tok_dir'] = tok_dir
    params['word_emb_path'] = os.path.join(
        tok_dir, params['dname'] + '.npy'
    )
    tok = utils.build_tok(
        data['docs'], max_feature=params['max_feature'],
        opath=os.path.join(tok_dir, '{}-{}.tok'.format(params['dname'], params['lang']))
    )
    if not os.path.exists(params['word_emb_path']):
        emb = utils.build_wt(tok, params['emb_path'], params['word_emb_path'])
    else:
        emb = np.load(params['word_emb_path'], allow_pickle=True)

    # tokenization and fill
    data['docs'] = pad_sequences(tok.texts_to_sequences(data['docs']), maxlen=params['max_len'])

    # split
    train_indices, val_indices, test_indices = utils.data_split(data)
    train_data = {
        'docs': [data['docs'][item] for item in train_indices],
        'labels': [data['labels'][item] for item in train_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in train_indices],
    }
    debias_weights = debias_weights[train_indices]

    valid_data = {
        'docs': [data['docs'][item] for item in val_indices],
        'labels': [data['labels'][item] for item in val_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in val_indices],
    }
    test_data = {
        'docs': [data['docs'][item] for item in test_indices],
        'labels': [data['labels'][item] for item in test_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in test_indices],
    }
    if params['over_sample']:
        ros = RandomOverSampler(random_state=33)
        sample_indices = list(range(len(train_data['docs'])))
        sample_indices, _ = ros.fit_resample(sample_indices, train_data['labels'])
        train_data = {
            'docs': [train_data['docs'][item] for item in sample_indices],
            'labels': [train_data['labels'][item] for item in sample_indices],
            params['domain_name']: [train_data[params['domain_name']][item] for item in sample_indices],
        }
        debias_weights = debias_weights[sample_indices]

    # too large data to fit memory, remove some
    # training data size: 200000
    if len(train_data['docs']) > 200000:
        np.random.seed(33)
        indices = list(range(len(train_data['docs'])))
        np.random.shuffle(indices)
        indices = indices[:200000]
        train_data = {
            'docs': [train_data['docs'][item] for item in indices],
            'labels': [train_data['labels'][item] for item in indices],
            params['domain_name']: [train_data[params['domain_name']][item] for item in indices],
        }
        debias_weights = debias_weights[indices]

    # train
    model = get_model(
        emb, num_lstm=1, max_seq_len=params['max_len'],
        dim_hidden=params['emb_dim'], num_classes=params['num_label'],
        dropout_rate=params['dp_rate'], lr=params['lr'], clipping=1
    )
    best_valid = 0.0
    cl_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(train_data['labels']),
        y=train_data['labels']
    )
    cl_weights = [np.exp(item / sum(cl_weights)) for item in cl_weights]
    cl_weights = dict(enumerate(cl_weights))

    for _ in tqdm(range(params['epochs'])):
        train_iter = data_gen(train_data['docs'], train_data['labels'], debias_weights, params['batch_size'])

        for x_train, y_labels, x_weights in train_iter:
            model.train_on_batch(
                x=x_train, y=y_labels, sample_weight=x_weights,
                class_weight=cl_weights
            )

        # valid
        y_preds_dev = []
        y_devs = []
        dev_iter = data_gen(valid_data['docs'], valid_data['labels'], None, params['batch_size'])
        for x_dev, y_dev, _ in dev_iter:
            x_dev = np.asarray(x_dev)
            tmp_preds = model.predict(x_dev)
            for item_tmp in tmp_preds:
                y_preds_dev.append(np.round(item_tmp[0]))
            for item_tmp in y_dev:
                y_devs.append(int(item_tmp))

        # test
        eval_score = metrics.f1_score(y_pred=y_preds_dev, y_true=y_devs, average='weighted')
        if eval_score > best_valid:
            best_valid = eval_score
            test_iter = data_gen(test_data['docs'], test_data['labels'], None, params['batch_size'])

            y_preds = []
            y_probs = []
            y_trues = []
            # evaluate on the test set
            for x_test, y_test, _ in test_iter:
                x_test = np.asarray(x_test)
                tmp_preds = model.predict(x_test)
                for item_tmp in tmp_preds:
                    y_probs.append(item_tmp[0])
                    y_preds.append(np.round(item_tmp[0]))
                for item_tmp in y_test:
                    y_trues.append(int(item_tmp))

            with open(params['result_path'], 'a') as wfile:
                wfile.write('{}...............................\n'.format(datetime.datetime.now()))
                wfile.write('Performance Evaluation for the task: {}\n'.format(params['dname']))
                wfile.write('F1-weighted score: {}\n'.format(
                    metrics.f1_score(y_true=y_trues, y_pred=y_preds, average='weighted')
                ))
                fpr, tpr, _ = metrics.roc_curve(y_true=y_trues, y_score=y_probs)
                wfile.write('AUC score: {}\n'.format(
                    metrics.auc(fpr, tpr)
                ))
                report = metrics.classification_report(
                    y_true=y_trues, y_pred=y_preds, digits=3
                )
                print(report)
                wfile.write(report)
                wfile.write('\n')

                wfile.write('Fairness Evaluation\n')
                wfile.write(
                    utils.fair_eval(
                        true_labels=y_trues,
                        pred_labels=y_preds,
                        domain_labels=test_data[params['domain_name']]
                    ) + '\n'
                )

                wfile.write('...............................\n\n')
                wfile.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--lr', type=float, help='Learning rate', default=.0001)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--max_len', type=int, help='Max length', default=512)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

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
        # ['review_trustpilot_english', review_dir + 'trustpilot/united_states.tsv', 'english'],
        # ['review_trustpilot_french', review_dir + 'trustpilot/france.tsv', 'french'],
        # ['review_trustpilot_german', review_dir + 'trustpilot/german.tsv', 'german'],
        # ['review_trustpilot_danish', review_dir + 'trustpilot/denmark.tsv', 'danish'],
        ['hatespeech_twitter_english', hate_speech_dir + 'english/corpus.tsv', 'english'],
        ['hatespeech_twitter_spanish', hate_speech_dir + 'spanish/corpus.tsv', 'spanish'],
        ['hatespeech_twitter_italian', hate_speech_dir + 'italian/corpus.tsv', 'italian'],
        ['hatespeech_twitter_portuguese', hate_speech_dir + 'portuguese/corpus.tsv', 'portuguese'],
        ['hatespeech_twitter_polish', hate_speech_dir + 'polish/corpus.tsv', 'polish'],
    ]

    for data_entry in tqdm(data_list):
        print('Working on: ', data_entry)

        parameters = {
            'result_path': os.path.join(result_dir, os.path.basename(__file__) + '.txt'),
            'model_dir': model_dir,
            'dname': data_entry[0],
            'dpath': data_entry[1],
            'lang': data_entry[2],
            'over_sample': False,
            'domain_name': 'gender',
            'epochs': 20,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_len': args.max_len,
            'dp_rate': .2,
            'optimizer': 'rmsprop',
            'emb_path': '../resources/embeddings/{}.vec'.format(data_entry[2]),  # adjust for different languages
            'emb_dim': 200,
            'word_emb_path': os.path.join(model_dir, data_entry[0] + '.npy'),
            'unique_domains': [],
            'bidirectional': False,
            'device': args.device,
            'num_label': 2,
            'max_feature': 10000,
        }

        make_weights(parameters)
        build_weight(parameters)
