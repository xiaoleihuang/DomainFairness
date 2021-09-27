import pickle
import os
import json

import numpy as np
import gensim
import pandas as pd
from sklearn import metrics

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def build_tkn(dpath='../analysis/all_data_encoded.tsv', opt='./vect/keras.tkn'):
    """Build keras tokenizer to create indices
    """
    doc_idx = 2  # column index of doc in the csv file
    if os.path.exists(opt):
        return pickle.load(open(opt, 'rb'))
    else:
        corpora = []
        tkn = Tokenizer(num_words=8000)

        print('Working on: ', dpath)
        with open(dpath) as dfile:
            dfile.readline()  # skip the column
            for line in dfile:
                line = line.strip().split('\t')[doc_idx]
                corpora.append(line)
        tkn.fit_on_texts(corpora)

        with open(opt, 'wb') as wfile:
            pickle.dump(tkn, wfile)
        return tkn


def build_indices(tname):
    """Convert document into indices
        tname: the name of task, either gender or ethinicity
    """
    doc_idx = 2
    max_len = 30  # maximum sentence length for padding

    # load the tokenizer
    tkn = build_tkn()

    for suffix in ['train', 'valid', 'test']:
        filep = '../split_data/' + tname + '.' + suffix
        opt = '../split_indices/' + tname + '.' + suffix

        if not os.path.exists(filep):
            print('File does not exist: %s' % filep)
            continue

        with open(opt, 'w') as wfile:
            with open(filep) as dfile:
                dfile.readline()

                corpora = []
                for line in dfile:
                    line = line.strip().split('\t')
                    corpora.append(line[doc_idx])

            corpora = pad_sequences(tkn.texts_to_sequences(corpora), maxlen=max_len)
            with open(filep) as dfile:
                wfile.write(dfile.readline())

                for idx, line in enumerate(dfile):
                    line = line.strip().split('\t')
                    line[doc_idx] = ' '.join(map(str, corpora[idx]))
                    line = '\t'.join(line) + '\n'
                    wfile.write(line)


def build_wt(filep='', opt='embd.npy'):
    """Build embedding weights by the tokenizer
        filep: the embedding file path
    """
    size = 300  # embedding size, in this study, we use 300

    if os.path.exists(opt):
        return np.load(opt)
    else:
        tkn = build_tkn()
        embed_len = len(tkn.word_index)
        if embed_len > tkn.num_words:
            embed_len = tkn.num_words

        # load embedding
        emb_matrix = np.zeros((embed_len + 1, size))

        if filep.endswith('.bin'):
            embeds = gensim.models.KeyedVectors.load_word2vec_format(
                filep, binary=True
            )

            for pair in zip(embeds.wv.index2word, embeds.wv.syn0):
                if pair[0] in tkn.word_index and \
                        tkn.word_index[pair[0]] < tkn.num_words:
                    emb_matrix[tkn.word_index[pair[0]]] = [
                        float(item) for item in pair[1]
                    ]
        else:
            with open(filep) as dfile:
                for line in dfile:
                    line = line.strip().split()

                    if line[0] in tkn.word_index and \
                            tkn.word_index[line[0]] < tkn.num_words:
                        emb_matrix[tkn.word_index[line[0]]] = [
                            float(item) for item in line[1:]
                        ]
        np.save(opt, emb_matrix)
        return np.asarray(emb_matrix)


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


def data_iter(tname, suffix='train', batch_size=64):
    """Data iterator to load train, valid, test data
        tname: the name of the task, gender or ethnicity
        suffix: three types, train, valid or test
    """
    doc_idx = 2  # idx of the doc column in the tsv data
    # data = []
    with open('../split_indices/' + tname + '.' + suffix) as dfile:
        dfile.readline()
        data = [item.strip().split('\t') for item in dfile]

    # shuffle every epoch
    np.random.shuffle(data)

    steps = len(data) // batch_size
    if len(data) % batch_size != 0:
        steps += 1

    for step in range(steps):
        docs = []
        labels = []

        for idx in range(step * batch_size, (step + 1) * batch_size):
            if idx > len(data) - 1:
                break
            docs.append([
                int(item) for item in data[idx][doc_idx].split()
            ])
            labels.append(int(data[idx][-1]))

        yield data[step * batch_size: (step + 1) * batch_size], docs, labels


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


def fair_eval(dpath):
    """Fairness Evaluation"""
    df = pd.read_csv(dpath, sep='\t')
    # get the task name from the file, gender or ethnicity
    task = dpath.split('.')[-1]

    scores = {
        'accuracy': metrics.accuracy_score(
            y_true=df.label, y_pred=df.pred
        ),
        'f1': metrics.f1_score(
            y_true=df.label, y_pred=df.pred, average='weighted'
        ),
        'auc': 0.0, 'fned': 0.0, 'fped': 0.0, 'tped': 0.0, 'tned': 0.0
    }

    # auc
    fpr, tpr, _ = metrics.roc_curve(
        y_true=df.label, y_score=df.pred_prob,
    )
    scores['auc'] = metrics.auc(fpr, tpr)

    '''fairness gaps'''
    # get overall confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true=df.label, y_pred=df.pred
    ).ravel()

    # get the unique types of demographic groups
    uniq_types = df[task].unique()
    for group in uniq_types:
        print(group)
        # calculate group specific confusion matrix
        group_df = df[df[task] == group]

        g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
            y_true=group_df.label, y_pred=group_df.pred
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
    with open(dpath + '.score', 'w') as wfile:
        wfile.write(json.dumps(scores))


def stopwords():
    return [
        'user', 'url', '.', 'hashtag', 'the', ',', 'to',
        'a', 'i', '!', 'and', ':', 'you', 'of', 'is',
        'in', 'for', '…', 'rt', '"', '?', 'that', 'on',
        'it', 'this', '...', 'my', 'with', '-', 'are'
    ]


if __name__ == '__main__':
    # build_tkn()
    # build_indices('gender')
    # build_indices('ethnicity')
    # build_indices('age')
    # build_indices('country')
    # build_indices('region')
    # build_indices('ethMulti')
    #
    # build_wt(
    #    '../embeddings/fair/GoogleNews-vectors-negative300-hard-debiased.txt'
    # )
    # build_wt(
    #    '../embeddings/'+\
    #    'GoogleNews-vectors-negative300.bin',
    #    opt='embd_unbias.npy'
    # )
    #
    # # swap the current documents in the split_indices folder
    # for task in ['age', 'country', 'region', 'ethnicity', 'gender', 'ethMulti']: #
    #    with open('../split_data/'+task+'_swap.train', 'w') as wfile:
    #        with open('../split_data/'+task+'.train') as dfile:
    #            wfile.write(dfile.readline())
    #
    #            data = []
    #            docs = []
    #            labels = []
    #            for line in dfile:
    #                wfile.write(line)
    #                line = line.strip().split('\t')
    #                data.append(line)
    #                docs.append(line[2])
    #                labels.append(line[-1])
    #
    #        new_docs, new_labels, new_idx = swap_gender(docs, labels)
    #        for idx, doc_idx in enumerate(new_idx):
    #            data[doc_idx][2] = new_docs[idx]
    #            wfile.write('\t'.join(data[doc_idx])+'\n')

    '''Do not forget to build swap indices manually'''
    # build indices for the swap file
    #    build_indices('gender_swap')
    #    build_indices('ethnicity_swap')
    #    build_indices('age_swap')
    #    build_indices('country_swap')
    #    build_indices('region_swap')
    #    build_indices('ethMulti_swap')

    pass
