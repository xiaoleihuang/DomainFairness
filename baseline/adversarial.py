"""Partial Codes adopt from Decoupling adversarial training for fair NLP
https://aclanthology.org/2021.findings-acl.41.pdf

model also follow the same idea as The Authors Matter:
Understanding and Mitigating Implicit Bias in Deep Text Classification
https://arxiv.org/pdf/2105.02778.pdf

"""
import os
import argparse
import datetime

import numpy as np
from sklearn import metrics
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils


class DeepMojiModel(nn.Module):
    def __init__(self, params):
        super(DeepMojiModel, self).__init__()
        self.params = params
        self.emb_size = self.params['emb_size']
        self.hidden_size = self.params['hidden_size']
        self.num_classes = self.params['num_label']
        self.adv_level = self.params['adv_level']
        self.n_hidden = self.params['n_hidden']
        self.device = self.params['device']
        self.dropout = nn.Dropout(p=self.params['dp_rate'])

        self.tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        self.AF = nn.Tanh()
        if self.params.get('AF', "") == "relu":
            self.AF = self.ReLU
        self.dense1 = nn.Linear(self.emb_size, self.hidden_size)
        self.dense2 = [nn.Linear(self.hidden_size, self.hidden_size).to(self.device) for _ in range(self.n_hidden)]
        self.dense3 = nn.Linear(self.hidden_size, self.num_label)

    def forward(self, inputs):
        out = self.dense1(inputs)
        out = self.AF(out)
        for hl in self.dense2:
            out = self.dropout(out)
            out = hl(out)
            out = self.tanh(out)
        out = self.dense3(out)
        return out

    def hidden(self, inputs):
        assert self.adv_level in {0, -1, -2}
        out = self.dense1(inputs)
        out = self.AF(out)
        if self.adv_level == -2:
            return out
        else:
            for hl in self.dense2:
                out = self.dropout(out)
                out = hl(out)
                out = self.tanh(out)
            if self.adv_level == -1:
                return out
            else:
                out = self.dense3(out)
                return out


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.GR = False
        self.grad_rev = GradientReversal(params['LAMBDA'])
        self.fc1 = nn.Linear(params['input_size'], params['adv_units'])
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(params['adv_units'], params['adv_units'])
        self.fc3 = nn.Linear(params['adv_units'], len(params['unique_domains']))

    def forward(self, inputs):
        if self.GR:
            inputs = self.grad_rev(inputs)
        out = self.fc1(inputs)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def hidden_representation(self, inputs):
        if self.GR:
            inputs = self.grad_rev(inputs)
        out = self.fc1(inputs)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        # Return the hidden representation from the second last layer
        return out

    def change_gradient_eversal(self, State=True):
        self.GR = State

    def get_weights(self):
        # return coef as numpy array
        dense_parameter = {name: param for name, param in self.fc3.named_parameters()}
        # get coef and covert to numpy
        # return dense_parameter["weight"].cpu().numpy()
        return dense_parameter["weight"].detach().cpu().numpy()


def build_base(params):
    if torch.cuda.is_available() and params['device'] != 'cpu':
        device = torch.device(params['device'])
    else:
        device = torch.device('cpu')
    params['device'] = device

    # load data
    data_encoder = utils.DataEncoder(params, mtype='bert')
    data = utils.data_loader(dpath=params['dpath'], lang=params['lang'])
    params['unique_domains'] = np.unique(data[params['domain_name']])

    train_indices, val_indices, test_indices = utils.data_split(data)
    train_data = {
        'docs': [data['docs'][item] for item in train_indices],
        'labels': [data['labels'][item] for item in train_indices],
        params['domain_name']: [data[params['domain_name']][item] for item in train_indices],
    }
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

    train_data = utils.TorchDataset(train_data, params['domain_name'])
    train_data_loader = DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True,
        collate_fn=data_encoder
    )
    valid_data = utils.TorchDataset(valid_data, params['domain_name'])
    valid_data_loader = DataLoader(
        valid_data, batch_size=params['batch_size'], shuffle=False,
        collate_fn=data_encoder
    )
    test_data = utils.TorchDataset(test_data, params['domain_name'])
    test_data_loader = DataLoader(
        test_data, batch_size=params['batch_size'], shuffle=False,
        collate_fn=data_encoder
    )

    # build model
    base_model = DeepMojiModel(params)
    base_model = base_model.to(device)
    discriminator = Discriminator(params)
    discriminator = discriminator.to(device)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=params['lr'])
    adv_optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, discriminator.parameters()), lr=1e-1 * params['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss().to(device)

    # train the networks
    print('Start to train...')
    print(params)
    best_score = 0.
    for epoch in tqdm(range(params['epochs'])):
        train_loss = 0
        base_model.train()
        discriminator.train()
        discriminator.GR = True

        for step, train_batch in enumerate(train_data_loader):
            train_batch = tuple(t.to(device) for t in train_batch)
            input_docs, input_labels, input_domains = train_batch
            optimizer.zero_grad()
            adv_optimizer.zero_grad()

            predictions = base_model(input_docs)
            loss = criterion(predictions, input_labels)
            train_loss += loss.item()

            hs = base_model.hidden(input_docs)
            adv_predictions = discriminator(hs)
            loss += criterion(adv_predictions, input_domains)

            loss_avg = train_loss / (step + 1)
            if (step + 1) % 101 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 0.5)
            optimizer.step()

            torch.nn.utils.clip_grad_norm(discriminator.parameters(), params['clipping_value'])
            adv_optimizer.step()

        # evaluate on the valid set
        y_preds = []
        y_trues = []
        discriminator.GR = False
        base_model.eval()
        discriminator.eval()
        valid_loss = 0.0
        for valid_batch in valid_data_loader:
            valid_batch = tuple(t.to(device) for t in valid_batch)
            input_docs, input_labels, input_domains = valid_batch
            with torch.no_grad():
                predictions = base_model(input_docs)
            valid_loss_batch = criterion(predictions, input_labels)
            valid_loss += valid_loss_batch.item()

            logits = torch.sigmoid(predictions.detach().cpu()).numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            y_preds.extend(pred_flat)
            y_trues.extend(input_labels.to('cpu').numpy())
        scheduler.step(valid_loss / len(valid_data_loader))

        eval_score = metrics.f1_score(y_pred=y_preds, y_true=y_trues, average='weighted')
        if eval_score > best_score:
            best_score = eval_score
            torch.save(base_model, params['model_dir'] + '{}.pth'.format(os.path.basename(__file__)))

            y_preds = []
            y_probs = []
            y_trues = []
            y_domains = []
            # evaluate on the test set
            for test_batch in test_data_loader:
                test_batch = tuple(t.to(device) for t in test_batch)
                input_docs, input_labels, input_domains = test_batch

                with torch.no_grad():
                    predictions = base_model(input_docs)
                logits = torch.sigmoid(predictions.detach().cpu()).numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                y_preds.extend(pred_flat)
                y_trues.extend(input_labels.to('cpu').numpy())
                y_probs.extend([item[1] for item in logits])
                y_domains.extend(input_domains.detach().cpu().numpy())

            with open(params['result_path'], 'a') as wfile:
                wfile.write('{}...............................\n'.format(datetime.datetime.now()))
                wfile.write('Performance Evaluation {}\n'.format(params['dname']))
                wfile.write('F1-weighted score: {}\n'.format(
                    metrics.f1_score(y_true=y_trues, y_pred=y_preds, average='weighted')
                ))
                fpr, tpr, _ = metrics.roc_curve(y_true=y_trues, y_score=y_probs)
                wfile.write('AUC score: {}\n'.format(
                    metrics.auc(fpr, tpr)
                ))
                wfile.write(metrics.classification_report(
                    y_true=y_trues, y_pred=y_preds, digits=3) + '\n')
                wfile.write('\n')

                wfile.write('Fairness Evaluation\n')
                wfile.write(
                    utils.fair_eval(
                        true_labels=y_trues,
                        pred_labels=y_preds,
                        domain_labels=y_domains
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
            'result_path': os.path.join(
                result_dir, '{}-{}.txt'.format(data_entry[0], os.path.basename(__file__))
            ),
            'model_dir': model_dir,
            'dname': data_entry[0],
            'dpath': data_entry[1],
            'lang': data_entry[2],
            'max_feature': 15000,
            'use_large': False,
            'domain_name': 'gender',
            'over_sample': False,
            'epochs': 10,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_len': args.max_len,
            'dp_rate': .2,
            'optimizer': 'adamw',
            'emb_dim': 200,
            'unique_domains': [],
            'bidirectional': False,
            'device': args.device,
            'num_label': 2,
            'kl_score': 0.01,
            'n_hidden': 1,  # default as original paper
            'adv_level': -1,
            'adv_units': 256,
            'clipping_value': 1,
        }

        # adjust parameters for other languages
        if parameters['lang'] != 'english':
            parameters['bert_name'] = 'bert-base-multilingual-uncased'

        build_base(parameters)
