# domain adaption + distance constraints
import argparse
import os
import datetime

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from imblearn.over_sampling import RandomOverSampler

import utils


class DomainRNN(nn.Module):
    def __init__(self, params):
        super(DomainRNN, self).__init__()
        self.params = params

        if 'word_emb_path' in self.params and os.path.exists(self.params['word_emb_path']):
            self.wemb = nn.Embedding.from_pretrained(
                torch.FloatTensor(np.load(self.params['word_emb_path']))
            )
        else:
            self.wemb = nn.Embedding(
                self.params['user_size'], self.params['emb_dim']
            )
            self.wemb.reset_parameters()
            nn.init.kaiming_uniform_(self.wemb.weight, a=np.sqrt(5))

        if self.params['bidirectional']:
            self.word_hidden_size = self.params['emb_dim'] // 2
        else:
            self.word_hidden_size = self.params['emb_dim']

        # domain adaptation
        self.doc_net_general = nn.GRU(
            self.params['emb_dim'], self.word_hidden_size,
            bidirectional=self.params['bidirectional'], dropout=self.params['dp_rate'],
            batch_first=True
        )
        self.doc_net_domain = nn.ModuleDict()
        for domain in self.params['unique_domains']:
            self.doc_net_domain['domain_{}'.format(domain)] = nn.GRU(
                self.params['emb_dim'], self.word_hidden_size,
                bidirectional=self.params['bidirectional'], dropout=self.params['dp_rate'],
                batch_first=True
            )

        # prediction
        self.predictor = nn.Linear(self.word_hidden_size * 3, self.params['num_label'])

        # mode setting
        self.mode = 'train'
        self.lambda_v = 1.0

    def change_mode(self, mode):
        self.mode = mode

    def change_lambda(self, lambda_v):
        self.lambda_v = lambda_v

    def forward(self, input_docs, input_domains=None):
        # encode the document from different perspectives
        doc_embs = self.wemb(input_docs)
        doc_general = self.doc_net_general(doc_embs)

        # mask out unnecessary features
        if self.mode == 'train':
            for domain in self.doc_net_domain:
                domain_mask = torch.Tensor(
                    [1 if int(domain.split('_')[1]) == item else 0 for item in input_domains],
                    device=self.params['device']
                )
                doc_domain = self.doc_net_domain[domain](doc_embs)
                # mask out features if domains do not match
                doc_domain = torch.mul(doc_domain, domain_mask[:, None])
                doc_general = torch.hstack((doc_general, doc_domain))
        else:
            # because domain encoder share the same shape with the general domain
            tensor_shape = doc_general.shape
            for _ in self.doc_net_domain:
                doc_general = torch.hstack((doc_general, torch.zeros(tensor_shape[0], tensor_shape[1])))
            doc_general *= self.lambda_v

        # prediction
        doc_preds = self.predictor(doc_general)
        return doc_preds


def domain_rnn(params):
    if torch.cuda.is_available() and params['device'] != 'cpu':
        device = torch.device(params['device'])
    else:
        device = torch.device('cpu')
    params['device'] = device

    # load data
    data_encoder = utils.DataEncoder(params)
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

    # build tokenizer and weight
    tok = utils.build_tok(
        data['docs'], max_feature=params['max_feature'],
        opath=os.path.join(params['model_dir'], params['dname'] + '.tok')
    )
    if not os.path.exists(params['word_emb_path']):
        utils.build_wt(tok, params['emb_path'], params['emb_dim'], params['word_emb_path'])

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
    rnn_model = DomainRNN(params)
    rnn_model = rnn_model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    kl_loss = nn.KLDivLoss().to(device)
    optimizer = torch.optim.RMSprop(rnn_model.parameters(), lr=params['lr'])

    # train the networks
    print('Start to train...')
    print(params)
    best_score = 0.
    for epoch in tqdm(range(params['epochs'])):
        train_loss = 0
        rnn_model.train()
        rnn_model.change_mode('train')

        for step, train_batch in enumerate(train_data_loader):
            train_batch = tuple(t.to(device) for t in train_batch)
            input_docs, input_labels, input_domains = train_batch
            optimizer.zero_grad()
            predictions = rnn_model(**{
                'input_docs': input_docs,
                'input_domains': input_domains
            })
            loss = criterion(predictions.view(-1, params['num_label']), input_labels.view(-1))
            for domain in params['unique_domains']:
                domain_kl = kl_loss(
                    rnn_model.doc_net_general.weight,
                    rnn_model.doc_net_domain['domain_{}'.format(domain)].weight
                )
                domain_kl = params['kl_score'] * domain_kl
                loss += domain_kl
            train_loss += loss.item()

            loss_avg = train_loss / (step + 1)
            if (step + 1) % 101 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 0.5)
            optimizer.step()

        # evaluate on the valid set
        y_preds = []
        y_trues = []
        rnn_model.change_mode('valid')
        for valid_batch in valid_data_loader:
            valid_batch = tuple(t.to(device) for t in valid_batch)
            input_docs, input_labels, input_domains = valid_batch
            with torch.no_grad():
                predictions = rnn_model(**{
                    'input_docs': input_docs,
                    'input_domains': input_domains
                })
            logits = predictions[0].detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            y_preds.extend(pred_flat)
            y_trues.extend(input_labels.to('cpu').numpy())

        eval_score = metrics.f1_score(y_pred=y_preds, y_true=y_trues, average='weighted')
        if eval_score > best_score:
            best_score = eval_score
            torch.save(rnn_model, params['model_dir'] + '{}.pth'.format(os.path.basename(__file__)))

            y_preds = []
            y_probs = []
            y_trues = []
            y_domains = []
            # evaluate on the test set
            for test_batch in test_data_loader:
                test_batch = tuple(t.to(device) for t in test_batch)
                input_docs, input_labels, input_domains = test_batch

                with torch.no_grad():
                    predictions = rnn_model(**{
                        'input_docs': input_docs,
                        'input_domains': input_domains
                    })
                logits = predictions[0].detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                y_preds.extend(pred_flat)
                y_trues.extend(input_labels.to('cpu').numpy())
                y_probs.extend([item[1] for item in logits])
                y_domains.extend(input_domains.detach().cpu().numpy())

            with open(params['result_path'], 'w') as wfile:
                wfile.write('{}...............................\n'.format(datetime.datetime.now()))
                wfile.write('Performance Evaluation\n')
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
        ['review_amazon_english', review_dir + 'amazon/amazon.tsv', 'english'],
        ['review_yelp-hotel_english', review_dir + 'yelp_hotel/yelp_hotel.tsv', 'english'],
        ['review_yelp-rest_english', review_dir + 'yelp_rest/yelp_rest.tsv', 'english'],
        ['review_twitter_english', review_dir + 'twitter/twitter.tsv', 'english'],
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
        model_dir = model_dir + data_entry[0] + '/'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

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
            'optimizer': 'rmsprop',
            'emb_path': '/data/models/embeddings/glove_twitter.vec',
            'emb_dim': 200,
            'word_emb_path': os.path.join(model_dir, data_entry[0] + '.npy'),
            'unique_domains': [],
            'bidirectional': False,
            'device': args.device,
            'num_label': 2,
            'kl_score': 0.001
        }

        domain_rnn(parameters)
