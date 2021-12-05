import pickle
import os
import json

import numpy as np
import gensim
from sklearn import metrics
from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer


def swap_gender(docs, labels):
    """Swap correference words by gender; for example
        he <-> she;
        him <-> her;
        his <-> her;
        mother <-> father;
        dad <-> mom;
        mr <-> mrs;

        This is only for training data.
        Because this method is to balance data during training.
    """
    m2f = {
        'he': 'she',
        'him': 'her',
        'his': 'her',
        'dad': 'mom',
        'daddy': 'mommy',
        'father': 'mother',
        'mr': 'mrs',
        'male': 'female',
        'white': 'black',
        # 'white': 'asian',
        # 'white': 'hispanic',
        'america': 'china',
        'american': 'chinese',
    }

    f2m = {
        'she': 'he',
        'her': 'him',
        'mom': 'dad',
        'mommy': 'daddy',
        'mother': 'father',
        'mrs': 'mr',
        'female': 'male',
        'asian': 'white',
        'black': 'white',
        'hispanic': 'white',
        'china': 'america',
        'chinese': 'american',
    }

    new_docs = []
    new_labels = []
    new_idx = []

    for idx in range(len(docs)):
        # for male 2 female
        new_doc = ' '.join([
            m2f.get(word, word) for word in docs[idx].split()
        ])
        if new_doc != docs[idx]:
            new_docs.append(new_doc)
            new_labels.append(labels[idx])
            new_idx.append(idx)

        # for female 2 male
        new_doc = ' '.join([
            f2m.get(word, word) for word in docs[idx].split()
        ])
        if new_doc != docs[idx]:
            new_docs.append(new_doc)
            new_labels.append(labels[idx])
            new_idx.append(idx)

    return new_docs, new_labels, new_idx


def stopwords():
    return [
        'user', 'url', '.', 'hashtag', 'the', ',', 'to',
        'a', 'i', '!', 'and', ':', 'you', 'of', 'is',
        'in', 'for', '…', 'rt', '"', '?', 'that', 'on',
        'it', 'this', '...', 'my', 'with', '-', 'are'
    ]


def data_loader(dpath, domain_name='gender', filter_null=True, lang='english'):
    """
    Default data format, tsv
    :param domain_name:
    :param lang: Language of the corpus, currently only supports languages defined by punkt
    :param filter_null: if filter out gender is empty or not.
    :param dpath:
    :return:
    """
    data = {
        'docs': [],
        'labels': [],
        'gender': [],
    }
    with open(dpath) as dfile:
        cols = dfile.readline().strip().split('\t')
        doc_idx = cols.index('text')
        domain_idx = cols.index(domain_name)
        label_idx = cols.index('label')

        for idx, line in enumerate(dfile):
            line = line.strip().lower().split('\t')
            if len(line) != len(cols):
                continue
            if len(line[doc_idx].strip().split()) < 10:
                continue

            # print(idx, line)
            if filter_null and line[domain_idx] == 'x':
                continue

            # binarize labels in the trustpilot dataset to keep the same format.
            try:
                label = int(line[label_idx])
            except ValueError:
                # encode hate speech data
                if line[label_idx] in ['0', 'no', 'neither', 'normal']:
                    label = 0
                else:
                    label = 1

            # label trustpilot review scores
            if 'trustpilot' in dpath:
                if label == 3:
                    continue
                elif label > 3:
                    label = 1
                else:
                    label = 0

            # encode gender.
            gender = line[domain_idx].strip()
            if gender != 'x':
                if gender not in ['1', '0']:
                    if 'f' in gender:
                        gender = 1
                    else:
                        gender = 0
                else:
                    gender = int(gender)

            data['docs'].append(' '.join(word_tokenize(line[doc_idx], language=lang)))
            data['labels'].append(label)
            data[domain_name].append(gender)
    return data


class TorchDataset(Dataset):
    def __init__(self, dataset, domain_name):
        self.dataset = dataset
        self.domain_name = domain_name

    def __len__(self):
        return len(self.dataset['docs'])

    def __getitem__(self, idx):
        if self.domain_name in self.dataset:
            return self.dataset['docs'][idx], self.dataset['labels'][idx], self.dataset[self.domain_name][idx]
        else:
            return self.dataset['docs'][idx], self.dataset['labels'][idx], -1


def data_split(data):
    """

    :param data:
    :return:
    """
    data_indices = list(range(len(data['docs'])))
    np.random.seed(33)  # for reproductive results
    np.random.shuffle(data_indices)

    train_indices = data_indices[:int(.8 * len(data_indices))]
    dev_indices = data_indices[int(.8 * len(data_indices)):int(.9 * len(data_indices))]
    test_indices = data_indices[int(.9 * len(data_indices)):]
    return train_indices, dev_indices, test_indices


def cal_fpr(fp, tn):
    """False positive rate"""
    return fp / (fp + tn)


def cal_fnr(fn, tp):
    """False negative rate"""
    return fn / (fn + tp)


def cal_tpr(tp, fn):
    """True positive rate"""
    return tp / (tp + fn)


def cal_tnr(tn, fp):
    """True negative rate"""
    return tn / (tn + fp)


def fair_eval(true_labels, pred_labels, domain_labels):
    scores = {
        'fned': 0.0,  # gap between fnr
        'fped': 0.0,  # gap between fpr
        'tped': 0.0,  # gap between tpr
        'tned': 0.0,  # gap between tnr
    }

    # get overall confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true=true_labels, y_pred=pred_labels
    ).ravel()

    # get the unique types of demographic groups
    uniq_types = np.unique(domain_labels)
    for group in uniq_types:
        # calculate group specific confusion matrix
        group_indices = [item for item in range(len(domain_labels)) if domain_labels[item] == group]
        group_labels = [true_labels[item] for item in group_indices]
        group_preds = [pred_labels[item] for item in group_indices]

        g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
            y_true=group_labels, y_pred=group_preds
        ).ravel()

        # calculate and accumulate the gaps
        scores['fned'] = scores['fned'] + abs(
            cal_fnr(fn, tp) - cal_fnr(g_fn, g_tp)
        )
        scores['fped'] = scores['fped'] + abs(
            cal_fpr(fp, tn) - cal_fpr(g_fp, g_tn)
        )
        scores['tped'] = scores['tped'] + abs(
            cal_tpr(tp, fn) - cal_tpr(g_tp, g_fn)
        )
        scores['tned'] = scores['tned'] + abs(
            cal_tnr(tn, fp) - cal_tnr(g_tn, g_fp)
        )
    return json.dumps(scores)


def build_wt(tkn, emb_path, opath):
    """Build weight using word embedding"""
    embed_len = len(tkn.word_index)
    if embed_len > tkn.num_words:
        embed_len = tkn.num_words

    if emb_path.endswith('.bin'):
        embeds = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True, unicode_errors='ignore'
        )
        emb_size = embeds.vector_size
        emb_matrix = list(np.zeros((embed_len + 1, emb_size)))
        for pair in zip(embeds.wv.index2word, embeds.wv.syn0):
            if pair[0] in tkn.word_index and \
                    tkn.word_index[pair[0]] < tkn.num_words:
                emb_matrix[tkn.word_index[pair[0]]] = np.asarray([
                    float(item) for item in pair[1]
                ], dtype=np.float32)
    else:
        dfile = open(emb_path)
        line = dfile.readline().strip().split()
        if len(line) < 5:
            line = dfile.readline().strip().split()
        emb_size = len(line[1:])
        emb_matrix = list(np.zeros((embed_len + 1, emb_size)))
        dfile.close()

        with open(emb_path) as dfile:
            for line in dfile:
                line = line.strip().split()
                if line[0] in tkn.word_index and \
                        tkn.word_index[line[0]] < tkn.num_words:
                    emb_matrix[tkn.word_index[line[0]]] = np.asarray([
                        float(item) for item in line[1:]
                    ], dtype=np.float32)
    # emb_matrix = np.array(emb_matrix, dtype=np.float32)
    np.save(opath, emb_matrix)
    return emb_matrix


def build_tok(docs, max_feature, opath):
    if os.path.exists(opath):
        return pickle.load(open(opath, 'rb'))
    else:
        # load corpus
        tkn = Tokenizer(num_words=max_feature)
        tkn.fit_on_texts(docs)

        with open(opath, 'wb') as wfile:
            pickle.dump(tkn, wfile)
        return tkn


class DataEncoder(object):
    def __init__(self, params, mtype='rnn'):
        """

        :param params:
        :param mtype: Model type, rnn or bert
        """
        self.params = params
        self.mtype = mtype
        if self.mtype == 'rnn':
            self.tok = pickle.load(open(
                os.path.join(params['model_dir'], params['dname'] + '.tok'), 'rb'))
        elif self.mtype == 'bert':
            self.tok = BertTokenizer.from_pretrained(params['bert_name'])
        else:
            raise ValueError('Only support BERT and RNN data encoders')

    def __call__(self, batch):
        docs = []
        labels = []
        domains = []
        for text, label, domain in batch:
            if self.mtype == 'bert':
                text = self.tok.encode_plus(
                    text, padding='max_length', max_length=self.params['max_len'],
                    return_tensors='pt', return_token_type_ids=False,
                    truncation=True,
                )
                docs.append(text['input_ids'][0])
            else:
                docs.append(text)
            labels.append(label)
            domains.append(domain)

        labels = torch.tensor(labels, dtype=torch.long)
        domains = torch.tensor(domains, dtype=torch.long)
        if self.mtype == 'rnn':
            # padding and tokenize
            docs = self.tok.texts_to_sequences(docs)
            docs = pad_sequences(docs)
            docs = torch.Tensor(docs).long()
        else:
            docs = torch.stack(docs).long()
        return docs, labels, domains
