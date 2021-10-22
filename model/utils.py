from nltk.tokenize import word_tokenize
import numpy as np


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

            data['docs'].append(' '.join(word_tokenize(line[doc_idx], language=lang)))
            data['labels'].append(label)
            data['gender'].append(line[gender_idx])
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
