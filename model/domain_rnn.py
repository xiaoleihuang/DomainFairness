# domain adaption + adversarial training
import os

import numpy as np
import torch.nn as nn
import torch


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
        self.doc_net_male = nn.GRU(
            self.params['emb_dim'], self.word_hidden_size,
            bidirectional=self.params['bidirectional'], dropout=self.params['dp_rate'],
            batch_first=True
        )
        self.doc_net_female = nn.GRU(
            self.params['emb_dim'], self.word_hidden_size,
            bidirectional=self.params['bidirectional'], dropout=self.params['dp_rate'],
            batch_first=True
        )

        # prediction
        input_size = self.word_hidden_size * 3
        if self.params['bidirectional']:
            input_size = input_size * 2
        self.predictor = nn.Linear(input_size, self.params['num_label'])

    def forward(self, input_doc_ids, masks=None):
        # encode the document from different perspectives
        doc_embs = self.wemb(input_doc_ids)
        doc_general = self.doc_net_general(doc_embs)

        # mask out unnecessary features
        if masks:

            pass
        pass
    pass