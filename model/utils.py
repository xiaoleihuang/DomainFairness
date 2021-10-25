from nltk.tokenize import word_tokenize
import numpy as np
from sklearn import metrics
import json


def data_loader(dpath, filter_null=True, lang='english'):
    """
    Default data format, tsv
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
        gender_idx = cols.index('gender')
        label_idx = cols.index('label')

        for line in dfile:
            line = line.strip().lower().split('\t')
            if filter_null and line[gender_idx] == 'x':
                continue

            # binarize labels in the trustpilot dataset to keep the same format.
            label = int(line[label_idx])
            if 'trustpilot' in dpath:
                if label == 3:
                    continue
                elif label > 3:
                    label = 1
                else:
                    label = 0

            # encode gender.
            gender = line[gender_idx].strip()
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
            data['gender'].append(gender)
    return data


def data_split(data):
    """

    :param data:
    :return:
    """
    data_indices = list(range(len(data['docs'])))
    np.random.seed(33)  # for reproductive results
    np.random.shuffle(data_indices)

    train_indices = data_indices[:int(.8*len(data_indices))]
    dev_indices = data_indices[int(.8*len(data_indices)):int(.9*len(data_indices))]
    test_indices = data_indices[int(.9 * len(data_indices)):]
    return train_indices, dev_indices, test_indices


def cal_fpr(fp, tn):
    """False positive rate"""
    return fp/(fp+tn)


def cal_fnr(fn, tp):
    """False negative rate"""
    return fn/(fn+tp)


def cal_tpr(tp, fn):
    """True positive rate"""
    return tp/(tp+fn)


def cal_tnr(tn, fp):
    """True negative rate"""
    return tn/(tn+fp)


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
