from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-base")

model = AutoModelWithLMHead.from_pretrained("t5-base")

def get_sentiment(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=256)

  dec = [tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True) for ids in output]
  label = dec[0]
  return label

get_sentiment(sent)

# Output: 'negative'



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
import re
from itertools import chain
from string import punctuation
from sklearn.model_selection import train_test_split
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize

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
parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','nyt','stsa'])
parser.add_argument("--samplecnt", default=64, type=int)
parser.add_argument("--ftepochs", default=30, type=int)
argsin = parser.parse_args()


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
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=16)
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
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=16)



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
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

# !wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xvf aclImdb_v1.tar.gz


#We will use 2000 samples from the train set for validation. 
#Let's choose 1000 postive reviews and 1000 negative reviews for validation and save them in the val directory
train_pos_files = glob.glob('aclImdb/train/pos/*.txt')
train_neg_files = glob.glob('aclImdb/train/neg/*.txt')

# !mkdir aclImdb/val aclImdb/val/pos aclImdb/val/neg
random.shuffle(train_pos_files)
random.shuffle(train_neg_files)

val_pos_files = train_pos_files[:1000]
val_neg_files = train_neg_files[:1000]

#mkdir aclImdb/train_sub aclImdb/train_sub/pos aclImdb/train_sub/neg

import shutil
# for f in val_pos_files:
#   shutil.move(f,  'aclImdb/val/pos')
# for f in val_neg_files:
#   shutil.move(f,  'aclImdb/val/neg')

# for f in train_pos_files[:128]:
#   shutil.copy(f,  'aclImdb/train_sub/pos')
# for f in train_neg_files[:128]:
#   shutil.copy(f,  'aclImdb/train_sub/neg')
ds = load_data(dataset=argsin.dsn, samplecnt= argsin.samplecnt)
ds.df_train['label_name'] = ds.df_train['label_name'].map(lambda x: x.lower())
ds.df_test['label_name'] = ds.df_test['label_name'].map(lambda x: x.lower())
df_train, df_valid = train_test_split(ds.df_train, test_size=0.15)


tokenizer = T5Tokenizer.from_pretrained('t5-base')
ids_neg = tokenizer.encode('negative </s>')

ids_neg = tokenizer.encode('Business </s>')

label_name_ids_lens = []
for label_name in ds.df_test['label_name'].unique():
    print(label_name, tokenizer.encode('{} </s>'.format(label_name)))
    label_name_ids_lens.append(len(tokenizer.encode('{} </s>'.format(label_name))))

ids_pos = tokenizer.encode('positive </s>')
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
    #self.path = os.path.join(data_dir, type_path + '.txt')

    self.data_column = "content"
    self.class_column = "label_name"
    # self.df = pd.read_csv(self.path, sep=";", header=None, names=[self.data_column, self.class_column],
    #                         engine="python")

    if type_path == 'train':
        self.df = df_train
    elif type_path == 'val':
        self.df = df_valid
    elif type_path == 'test':
        self.df = ds.df_test

    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    for content, label_name in zip(self.df['content'].tolist(), self.df['label_name'].tolist()):

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [content+' </s>'], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [label_name+' </s>'], max_length=max(label_name_ids_lens), pad_to_max_length=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)

class ImdbDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path,  max_len=512):
    self.pos_file_path = os.path.join(data_dir, type_path, 'pos')
    self.neg_file_path = os.path.join(data_dir, type_path, 'neg')
    
    self.pos_files = glob.glob("%s/*.txt" % self.pos_file_path)
    self.neg_files = glob.glob("%s/*.txt" % self.neg_file_path)
    
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    self._buil_examples_from_files(self.pos_files, 'positive')
    self._buil_examples_from_files(self.neg_files, 'negative')
  
  def _buil_examples_from_files(self, files, sentiment):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for path in files:
      with open(path, 'r') as f:
        text = f.read()
      
      line = text.strip()
      line = REPLACE_NO_SPACE.sub("", line) 
      line = REPLACE_WITH_SPACE.sub("", line)
      line = line + ' </s>'

      target = sentiment + " </s>"

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=max(label_name_ids_lens), pad_to_max_length=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)



# dataset = ImdbDataset(tokenizer, 'aclImdb', 'train',  max_len=256)
# len(dataset)

# data = dataset[38]
# print(tokenizer.decode(data['source_ids']))
# print(tokenizer.decode(data['target_ids']))

# !mkdir -p t5_imdb_sentiment


args_dict.update({'data_dir': 'aclImdb', 'output_dir': 't5_imdb_sentiment', 'num_train_epochs': argsin.ftepochs})
args = argparse.Namespace(**args_dict)



checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=1,
    max_epochs=args.num_train_epochs,
    early_stop_callback=True,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=False,
    callbacks=[LoggingCallback(), EarlyStopping(monitor="val_loss")],
)


'''
Define the get_dataset function to return the dataset. 
The model calls this function to get the train and val datasets.
 We are defining a dataset function so that we won't need to modify the model code at all. 
 Redefine the function to return different dataset according to the problem. 
 While this is not the best solution for now this works
'''
def get_dataset(tokenizer, type_path, args):
  return EmotionDataset(tokenizer=tokenizer, type_path=type_path,  max_len=args.max_seq_length)

#model = T5FineTuner(vars(args))
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

# !mkdir t5_base_imdb_sentiment
## save the model this way so next time you can load it using T5ForConditionalGeneration.from_pretrained



#For inference we will use the generate method with greedy decoding with max length 2.
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

for se in ['train', 'test']:
    dataset = EmotionDataset(tokenizer,  se,  max_len=256)

    loader = DataLoader(dataset, batch_size=32, num_workers=8)
    model.model.eval()
    outputs = []
    targets = []
    for batch in tqdm(loader):
        outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                                  attention_mask=batch['source_mask'].cuda(), 
                                  max_length=max(label_name_ids_lens))

        pred = [tokenizer.decode(ids,clean_up_tokenization_spaces=True, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids,clean_up_tokenization_spaces=True, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(pred)
        targets.extend(target)


    for i, pred in enumerate(outputs):
      if pred not in ds.df_test['label_name'].unique():
        print(pred, 'detected invalid prediction')

    acc = metrics.accuracy_score(targets, outputs)
    print(se, 'acc==>', acc)


# os.makedirs('t5_base_imdb_sentiment_ds__{}'.format(train_path), exist_ok=True)
# model.model.save_pretrained('t5_base_imdb_sentiment_ds__{}'.format(train_path))










# emotion










