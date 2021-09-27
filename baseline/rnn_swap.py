"""GRU
"""
import pickle
import os
import numpy as np
from sklearn.metrics import f1_score

from keras.layers import Input, Embedding
from keras.layers import Bidirectional, GRU
from keras.layers import Dense
from keras.models import Model

import utils


def build_model(params):
    clf_path = './clf/gru_swap.'+params['tname']
    result_path = './results/gru_swap.'+params['tname']

    if os.path.exists(clf_path):
        best_model = pickle.load(open(clf_path, 'rb'))
    else:
        if params['num_cl'] > 2:
            pred_func = 'softmax'
            loss_func = 'categorical_crossentropy'
        else:
            pred_func = 'sigmoid'
            loss_func = 'binary_crossentropy'

        # load embedding matrix
        wt_matrix = utils.build_wt(params['emb_file'])

        # define the GRU model
        inputs = Input(
            shape=(params['seq_max_len'],), dtype='int32', name='input'
        )
        embeds = Embedding(
            wt_matrix.shape[0], wt_matrix.shape[1],
            weights=[wt_matrix], input_length=params['seq_max_len'],
            trainable=True, name='embedding'
        )(inputs)
        bigru = Bidirectional(GRU(
            params['rnn_size'], kernel_initializer="glorot_uniform"
        ))(embeds)
        predicts = Dense(
            params['num_cl'], activation=pred_func, name='predict'
        )(bigru)

        model = Model(inputs=inputs, outputs=predicts)
        model.compile(
            loss=loss_func, optimizer=params['opt'],
            metrics=['accuracy']
        )
        print(model.summary())

        # create indices of documents, 
        # because other models will overwrite the current indices
        utils.build_indices(params['tname'])

        best_valid_f1 = 0.0
        best_model = None
        
        for e in range(params['epochs']):
            accuracy = 0.0
            loss = 0.0
            step = 1

            print('--------------Epoch: {}--------------'.format(e))

            # load the datasets (swap)
            train_iter = utils.data_iter(
                params['tname']+'_swap', suffix='train',
                batch_size=params['batch_size']
            )

            # train the model
            for _, x_train, y_train in train_iter:
                if len(np.unique(y_train)) == 1:
                    continue

                tmp = model.train_on_batch(
                    [x_train], y_train,
                    class_weight=params['class_wt']
                )

                loss += tmp[0]
                loss_avg = loss / step
                accuracy += tmp[1]
                accuracy_avg = accuracy / step
                if step % 30 == 0:
                    print('Step: {}'.format(step))
                    print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
                    print('-------------------------------------------------')
                step += 1

            # valid the model
            print('---------------------------Validation------------------------------')
            valid_iter = utils.data_iter(
                params['tname'], suffix='valid',
                batch_size=params['batch_size']
            )

            y_preds = []
            y_valids = []

            for _, x_valid, y_valid in valid_iter:
                tmp_preds = model.predict([x_valid])
                
                for item_tmp in tmp_preds:
                    y_preds.append(round(item_tmp[0]))
                y_valids.extend(y_valid)

            valid_f1 = f1_score(y_true=y_valids, y_pred=y_preds, average='weighted')
            print('Validating f1-weighted score: ' + str(valid_f1))

            if best_valid_f1 < valid_f1:
                best_valid_f1 = valid_f1
                best_model = model

                pickle.dump(best_model, open(clf_path, 'wb'))

    print('------------------------------Test---------------------------------')
    if not os.path.exists(result_path):
        y_preds_test = []
        y_preds_prob = []

        test_iter = utils.data_iter(
            params['tname'], suffix='test',
            batch_size=params['batch_size']
        )

        data = []
        for data_batch, x_test, y_test in test_iter:
            tmp_preds = best_model.predict([x_test])
            data.extend(data_batch)
            for item_tmp in tmp_preds:
                y_preds_prob.append(item_tmp[0])
                y_preds_test.append(int(round(item_tmp[0])))

        assert len(y_preds_test) == len(data)

        with open(result_path, 'w') as wfile:
            wfile.write(
                '\t'.join([
                    'tid', 'uid', 'text', 'date', 'gender',
                    'age', 'region', 'country', 'ethnicity', 'ethMulti', 'label', 'pred', 'pred_prob'
                ]) + '\n'
            )
            for idx, label in enumerate(y_preds_test):
                data[idx].append(str(y_preds_test[idx]))
                data[idx].append(str(y_preds_prob[idx]))
                
                wfile.write('\t'.join(data[idx])+'\n')
    utils.fair_eval(result_path)


if __name__ == '__main__':
    # gender
    parameters = {
        'tname': 'ethMulti', 'seq_max_len': 30, 'dp_rate': 0.3, 'opt': 'rmsprop', 'epochs': 10,
        'class_wt': 'auto', 'batch_size': 64, 'lr': 0.001, 'num_cl': 1, 'dense_ac': 'relu', 'rnn_size': 256,
        'emb_file': '../embeddings/fair/GoogleNews-vectors-negative300-hard-debiased.txt'
    }

    # # gender
    # build_model(params)
    #
    # # ethnicity
    # params['tname'] = 'ethnicity'
    # build_model(params)
    #
    # # age
    # params['tname'] = 'age'
    # build_model(params)
    #
    # # country
    # params['tname'] = 'country'
    # build_model(params)
    #
    # # region
    # params['tname'] = 'region'
    # build_model(params)

    # ethnicity-multi
    build_model(parameters)
