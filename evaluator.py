"""Scripts for evaluation,
    metrics: F1, AUC, FNED, FPED
"""
import pandas as pd
from sklearn import metrics
import json


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


def fair_eval(dpath, opt):
    """Fairness Evaluation
        dpath: input eval file path
        opt: output results path
    """
    df = pd.read_csv(dpath, sep='\t', na_values='x')
    # get the task name from the file, gender or ethnicity
    tasks = ['gender', 'age', 'country', 'ethnicity']

    scores = {
        'accuracy': metrics.accuracy_score(
            y_true=df.label, y_pred=df.pred
        ),
        'f1-macro': metrics.f1_score(
            y_true=df.label, y_pred=df.pred, average='macro'
        ),
        'f1-weight': metrics.f1_score(
            y_true=df.label, y_pred=df.pred, average='weighted'
        ),
        'auc': 0.0
    }

    # accuracy, f1, auc
    fpr, tpr, _ = metrics.roc_curve(
        y_true=df.label, y_score=df.pred_prob,
    )
    scores['auc'] = metrics.auc(fpr, tpr)

    '''fairness gaps'''
    for task in tasks:

        '''Filter out some tasks'''
        if ('Polish' in dpath or 'Italian' in dpath) and \
                task in ['country', 'ethnicity']:
            continue

        scores[task] = {
            'fned': 0.0,  # gap between fnr
            'fped': 0.0,  # gap between fpr
            'tped': 0.0,  # gap between tpr
            'tned': 0.0,  # gap between tnr
        }
        # filter out the one does not have attributes
        task_df = df[df[task].notnull()]

        # get overall confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true=task_df.label, y_pred=task_df.pred
        ).ravel()

        # get the unique types of demographic groups
        uniq_types = task_df[task].unique()
        for group in uniq_types:
            # calculate group specific confusion matrix
            group_df = task_df[task_df[task] == group]

            g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
                y_true=group_df.label, y_pred=group_df.pred
            ).ravel()

            # calculate and accumulate the gaps
            scores[task]['fned'] = scores[task]['fned'] + abs(
                cal_fnr(fn, tp) - cal_fnr(g_fn, g_tp)
            )
            scores[task]['fped'] = scores[task]['fped'] + abs(
                cal_fpr(fp, tn) - cal_fpr(g_fp, g_tn)
            )
            scores[task]['tped'] = scores[task]['tped'] + abs(
                cal_tpr(tp, fn) - cal_tpr(g_tp, g_fn)
            )
            scores[task]['tned'] = scores[task]['tned'] + abs(
                cal_tnr(tn, fp) - cal_tnr(g_tn, g_fp)
            )
    with open(opt, 'w') as wfile:
        wfile.write(json.dumps(scores))
    print(scores)
