# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Original Copyright: https://github.com/1024er/cbert_aug/blob/develop/aug_dataset.py
# /root/yanan/env_cbert/bin/python

import csv,time,os,logging,argparse,random
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForMaskedLM, AdamW

from utils.data_processors import get_task_processor
import transformers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#assert torch.__version__=='1.6.0' and transformers.__version__ == '2.9.0'

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels):
        self.init_ids = init_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, seed=12345):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    # ----
    dupe_factor = 5
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20
    rng = random.Random(seed)

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        segment_id = label_map[example.label]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        # Due to we use conditional bert, we need to place label information in segment_ids
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(segment_id)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(segment_id)
        tokens.append("[SEP]")
        segment_ids.append(segment_id)
        masked_lm_labels = [-100] * max_seq_length

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)
        len_cand = len(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms_pos = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms_pos) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = tokens[cand_indexes[rng.randint(0, len_cand - 1)]]

            masked_lm_labels[index] = tokenizer.convert_tokens_to_ids([tokens[index]])[0]
            output_tokens[index] = masked_token
            masked_lms_pos.append(index)

        init_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            init_ids.append(0)
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)  # ?segment_id

        assert len(init_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if ex_index < 2:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     print("init_ids: %s" % " ".join([str(x) for x in init_ids]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     print("masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))

        features.append(
            InputFeatures(init_ids=init_ids,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          masked_lm_labels=masked_lm_labels))
    return features


def prepare_data(features):
    all_init_ids = torch.tensor([f.init_ids for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in features],
                                        dtype=torch.long)
    tensor_data = TensorDataset(all_init_ids, all_input_ids, all_input_mask, all_segment_ids,
                               all_masked_lm_labels)
    return tensor_data


def rev_wordpiece(str):
    #print(str)
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str[1:-1])

max_seq_length = 64
warmup_proportion = 0.1 # Proportion of training to perform linear learning rate warmup for. "
                         # E.g., 0.1 = 10%% of training.

def compute_dev_loss(model, dev_dataloader):
    model.eval()
    sum_loss = 0.
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        _, input_ids, input_mask, segment_ids, masked_ids = batch
        inputs = {'input_ids': batch[1],
                  'attention_mask': batch[2],
                  'token_type_ids': batch[3],
                  'masked_lm_labels': batch[4]
                  }

        outputs = model(**inputs)
        loss = outputs[0]
        sum_loss += loss.item()
    return sum_loss

# Prepare optimizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                                cache_dir='./cache_cbert', local_files_only=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                  do_lower_case=True,
                                                  cache_dir='./cache_cbert')


#t_total = num_train_steps
no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=4e-5, eps=1e-8)







