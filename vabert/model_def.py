"""
model_def.py — BertForVAFactorized, the model architecture.

A BioBERT backbone (dmis-lab/biobert-base-cased-v1.2) with four
token-classification heads on the shared hidden states:

  span_head      3-way  O / B / I                 (every token)
  lat_head       4-way  Right Left Both Unk       (span tokens only)
  method_head    6-way  Aided Unaided AidedPH UnaidedPH UnkPH Unk
  notation_head  8-way  7 notations + Unk

Attribute heads are trained only on tokens inside a gold span; every other
token, and every span whose field was disputed during labelling, carries
IGNORE (-100) for that head. At inference a span's attribute is the mean
of its tokens' softmax distributions, so each field has its own
confidence and can be committed or withheld separately.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
IGNORE = -100

SPAN_LABELS = ['O', 'B', 'I']
LAT_LABELS = ['Right', 'Left', 'Both', 'Unk']
METHOD_LABELS = ['Aided', 'Unaided', 'AidedPH', 'UnaidedPH', 'UnkPH', 'Unk']
NOTATION_LABELS = ['snellen_m', 'snellen_ft', 'qualitative', 'etdrs', 'logmar', 'decimal', 'thomson', 'Unk']
HEADS = {'span': SPAN_LABELS, 'lat': LAT_LABELS, 'method': METHOD_LABELS, 'notation': NOTATION_LABELS}


class BertForVAFactorized(nn.Module):
    def __init__(self, model_name=MODEL_NAME, config=None):
        """config: a BertConfig to build the backbone without downloading
        pretrained weights (inference, where the checkpoint overwrites them)."""
        super().__init__()
        self.bert = AutoModel.from_config(config) if config is not None else AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, len(v)) for k, v in HEADS.items()})

    def forward(self, input_ids, attention_mask, labels=None):
        """labels: optional dict head -> LongTensor [batch, seq] (IGNORE where unlabelled)."""
        seq = self.dropout(self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        logits = {k: h(seq) for k, h in self.heads.items()}
        out = {f'{k}_logits': v for k, v in logits.items()}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE)
            active = attention_mask.view(-1) == 1
            total = 0.0
            for k, lg in logits.items():
                lab = labels[k].view(-1).clone()
                lab[~active] = IGNORE
                if (lab != IGNORE).any():
                    l_k = loss_fct(lg.view(-1, lg.shape[-1]), lab)
                else:
                    l_k = lg.sum() * 0.0
                out[f'{k}_loss'] = l_k
                total = total + l_k
            out['loss'] = total
        return out
