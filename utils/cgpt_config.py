import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel

from transformers import AdamW
from utils.data_processors import get_task_processor
import transformers
assert torch.__version__=='1.6.0' and transformers.__version__ == '2.9.0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GPT2_MODEL = 'gpt2'
EOS_TOKEN = '<|endoftext|>'
SEP_TOKEN = '<SEP>'

STOP_TOKENS = [EOS_TOKEN, '<']

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, examples):
        self.examples = examples




def convert_examples_to_features(examples, block_size, tokenizer, seed=12345):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    text = ""
    for (ex_index, example) in enumerate(examples):
        if ex_index:
            text += " " + example.label + SEP_TOKEN + example.text_a + EOS_TOKEN
        else:
            text += example.label + SEP_TOKEN + example.text_a + EOS_TOKEN

    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    for i in range(0, len(tokenized_text) - block_size + 1,
                   block_size):  # Truncate in block of block_size
        features.append(InputFeatures(
            examples=tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size])))

    return features


def prepare_data(features):
    all_input_ids = torch.tensor([f.examples for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.examples for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_labels)
    return tensor_data


max_seq_length = 64
warmup_proportion = 0.1 # Proportion of training to perform linear learning rate warmup for. "
                         # E.g., 0.1 = 10%% of training.
prefix = 3

def compute_dev_loss(model, dev_dataloader):
    model.eval()
    sum_loss = 0.
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'labels': batch[1]}

        outputs = model(**inputs)
        loss = outputs[0]
        sum_loss += loss.item()
    return sum_loss

tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL,
                                          do_lower_case=True,
                                          cache_dir='./CACHE')

model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL, local_files_only=True, 
                                        cache_dir='./CACHE')

#gpt2_nlp  = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, return_full_text=False)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=4e-5, eps=1e-8)

model.to(device)


block_size = min(256, tokenizer.max_len_single_sentence)
        




