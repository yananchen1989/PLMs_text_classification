# finetune T5
# !pip install transformers==2.9.0
# !pip install pytorch_lightning==0.7.5

import argparse
import glob
import os
import json
import time
import logging
import random
import re,gc
from itertools import chain
from string import punctuation
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from utils.load_data import * 
# def set_seed(seed):
#   random.seed(seed)
#   np.random.seed(seed)
#   torch.manual_seed(seed)
#   if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# set_seed(42)
parser = argparse.ArgumentParser()
#parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','nyt','stsa','cc'])
#parser.add_argument("--samplecnt", default=64, type=int)
parser.add_argument("--ftepochs", default=7, type=int)
parser.add_argument("--ft_pattern", default='pp', type=str, choices=['pp', 'tc', 'ep'])
parser.add_argument("--maxlen", default=512, type=int)
parser.add_argument("--ccsample", default=0.1, type=float)
parser.add_argument("--gpus", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=8, type=int)

argsin = parser.parse_args()

output_dir = 'ft_model_t5_{}'.format( argsin.ft_pattern) 
os.makedirs(output_dir, exist_ok=True)

class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams = hparams
    #for key in hparams.keys():
    #     self.hparams[key]=hparams[key]
    self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
  
  def is_logger(self):
    return self.trainer.proc_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
    )

  def _step(self, batch):
    lm_labels = batch["target_ids"]
    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        lm_labels=lm_labels,
        decoder_attention_mask=batch['target_mask']
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    if self.trainer.use_tpu:
      xm.optimizer_step(optimizer)
    else:
      optimizer.step()
    optimizer.zero_grad()
    self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path='train', args=self.hparams)
    print("train_dataset completed")
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=argsin.num_workers)
    print("train dataloader completed")
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
    print("val_dataset completed")
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=argsin.num_workers)



logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))
        if key == 'avg_val_loss':
          output_dir_ck = output_dir+'/'+'checkpoint_loss_{}'.format(metrics[key].cpu().numpy())
          os.makedirs(output_dir_ck, exist_ok=True)
          model.model.save_pretrained(output_dir_ck)
          logger.info("model save_pretrained")

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))


'''
Let's define the hyperparameters and other arguments. You can overide this dict for specific task as needed. 
While in most of cases you'll only need to change the data_dir and output_dir.

Here the batch size is 8 and gradient_accumulation_steps are 16 so the effective batch size is 128
'''

args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=argsin.maxlen,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=argsin.batch_size,
    eval_batch_size=argsin.batch_size,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=argsin.gpus,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)





df_cc_text2text = get_cc_text_double(argsin.ft_pattern, argsin.ccsample)

row = df_cc_text2text.sample(1)
print(row['text1'].tolist()[0])
print('\n')
print(row['text2'].tolist()[0])


df_train, df_valid = train_test_split(df_cc_text2text, test_size=0.05)

print('data loaded', df_train.shape[0], df_valid.shape[0])


#tokenizer = T5Tokenizer.from_pretrained('t5-base')
# ids_neg = tokenizer.encode('negative </s>')

# ids_neg = tokenizer.encode('Business </s>')

# label_name_ids_lens = []
# for label_name in ds.df_test['label_name'].unique():
#     print(label_name, tokenizer.encode('{} </s>'.format(label_name)))
#     label_name_ids_lens.append(len(tokenizer.encode('{} </s>'.format(label_name))))

# ids_pos = tokenizer.encode('positive </s>')
#len(ids_neg), len(ids_pos)




'''

The dataset below takes care of reading the review files and processing the examples in text-2-text format.

It cleans the review text by removing the html tags. 
It also appends the eos token </s> at the end of input and target as required by the T5 model

For T5 max input length is 512 and we can choose the max length for target sequence depending upon our dataset. 
The T5Tokenizer encodes both 'postive' and 'negative' as a single ids so I chose the max target length 2, 
extra 1 for the </s> token
'''


class EmotionDataset(Dataset):
  def __init__(self, tokenizer, type_path,  max_len=512):

    if type_path == 'train':
        self.df = df_train
    elif type_path == 'val':
        self.df = df_valid
    # elif type_path == 'test':
    #     self.df = ds.df_test

    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"]#.squeeze()
    target_ids = self.targets[index]["input_ids"]#.squeeze()

    src_mask    = self.inputs[index]["attention_mask"]#.squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"]#.squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    # for text1, text2 in zip(self.df['text1'].tolist(), self.df['text2'].tolist()):
    #    # tokenize inputs
    #   tokenized_inputs = self.tokenizer.batch_encode_plus(
    #       [text1+' {}'.format(self.tokenizer.eos_token)], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
    #   )
    #    # tokenize targets
    #   tokenized_targets = self.tokenizer.batch_encode_plus(
    #       [text2+' {}'.format(self.tokenizer.eos_token)], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
    #   )

    #   self.inputs.append(tokenized_inputs)
    #   self.targets.append(tokenized_targets)

    tokenized_inputs = self.tokenizer.batch_encode_plus(
        (self.df['text1']+' {}'.format(self.tokenizer.eos_token)).tolist(), max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
    tokenized_targets = self.tokenizer.batch_encode_plus(
        (self.df['text2']+' {}'.format(self.tokenizer.eos_token)).tolist(), max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )


    for i in range(self.df.shape[0]):
        self.inputs.append({'input_ids':tokenized_inputs['input_ids'][i] , \
                        'attention_mask':tokenized_inputs['attention_mask'][i] })
    for i in range(self.df.shape[0]):
        self.targets.append({'input_ids':tokenized_targets['input_ids'][i] , \
                        'attention_mask':tokenized_targets['attention_mask'][i] })



# dataset = ImdbDataset(tokenizer, 'aclImdb', 'train',  max_len=256)
# len(dataset)

# data = dataset[38]
# print(tokenizer.decode(data['source_ids']))
# print(tokenizer.decode(data['target_ids']))

# !mkdir -p t5_imdb_sentiment

args_dict.update({'output_dir': output_dir, 'num_train_epochs': argsin.ftepochs})
args = argparse.Namespace(**args_dict)



# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=3
# )

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=argsin.gpus,
    max_epochs=args.num_train_epochs,
    early_stop_callback=True,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=False,
    callbacks=[LoggingCallback(), EarlyStopping(monitor="val_loss")],
)



def get_dataset(tokenizer, type_path, args):
  return EmotionDataset(tokenizer=tokenizer, type_path=type_path,  max_len=args.max_seq_length)

#model = T5FineTuner(vars(args))
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)


trainer.fit(model)


#For inference we will use the generate method with greedy decoding with max length 2.
# import textwrap
# from tqdm.auto import tqdm
# from sklearn import metrics

# for se in ['train', 'test']:
#     dataset = EmotionDataset(tokenizer,  se,  max_len=256)

#     loader = DataLoader(dataset, batch_size=32, num_workers=8)
#     model.model.eval()
#     outputs = []
#     targets = []
#     for batch in tqdm(loader):
#         outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
#                                   attention_mask=batch['source_mask'].cuda(), 
#                                   max_length=max(label_name_ids_lens))

#         pred = [tokenizer.decode(ids,clean_up_tokenization_spaces=True, skip_special_tokens=True) for ids in outs]
#         target = [tokenizer.decode(ids,clean_up_tokenization_spaces=True, skip_special_tokens=True) for ids in batch["target_ids"]]

#         outputs.extend(pred)
#         targets.extend(target)


#     for i, pred in enumerate(outputs):
#       if pred not in ds.df_test['label_name'].unique():
#         print(pred, 'detected invalid prediction')

#     acc = metrics.accuracy_score(targets, outputs)
#     print(se, 'acc==>', acc)


#model.model.save_pretrained(output_dir)














