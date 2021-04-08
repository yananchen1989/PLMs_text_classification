

import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


class generation():
    def __init__(self, model_name='gpt2',num_return_sequences=1):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_CLASSES = {
                    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
                    "finetuned_gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
                    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
                    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
                    "xlnet-base-cased": (XLNetLMHeadModel, XLNetTokenizer),
                    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
                    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
                }
        self.length = 100
        self.k = 0
        self.p = 0.9
        self.num_return_sequences = num_return_sequences
        self.model_class, self.tokenizer_class = self.MODEL_CLASSES[self.model_name]
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name)
        if 'gpt2' in self.model_name :
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # no eos_token for ctrl tokenizer

        self.model = self.model_class.from_pretrained(self.model_name)
        self.model.to(self.device)
        if self.model_name == "xlnet-base-cased":
            self.p=0.95
            self.k=60

        self.length = self.adjust_length_to_model(self.length, max_sequence_length=self.model.config.max_position_embeddings)

        if self.model_name == 'ctrl':
            self.temperature = 0.3
            self.repetition_penalty = 1.2
        else:
            self.temperature = 1.0
            self.repetition_penalty = 1.0

    def adjust_length_to_model(self, length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = 1000  # avoid infinite loop
        return length


    def ctrl_label2prefix(self, label):
        # https://github.com/salesforce/ctrl/blob/master/control_codes.py
        '''
        'Pregnancy Christianity Explain Fitness Saving Ask Ass Joke Questions Thoughts Retail 
        Feminism Writing Atheism Netflix Computing Opinion Alone Funny Gaming Human India Joker Diet 
        Legal Norman Tip Weight Movies Running Science Horror Confession Finance Politics Scary Support 
        Technologies Teenage Event Learned Notion Wikipedia Books Extract Confessions Conspiracy Links 
        Narcissus Relationship Relationships Reviews News Translation multilingual'
        '''
        #return 'News'

        if label in ('Sci/Tech', 'tech'):
            return 'News Science and Technologies'
        elif label in ('politics'):
            return 'News Politics'
        elif label in ('Sports', 'sport'):
            return 'News Sports'
        else:
            return 'News World'

    def augment(self, prompt_text):
        if self.model_name == 'ctrl':
            prefix = 'News '
        else:
            prefix = ''
        encoded_prompt = self.tokenizer.encode(prefix  + prompt_text, \
                        truncation=True, max_length=250,  \
                        add_special_tokens=False, return_tensors="pt")
        # if 'ctrl' in self.model_name:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     encoded_prompt = self.tokenizer.encode(prefix  + prompt_text, \
        #                 truncation=True, max_length=250, padding=True, \
        #                 add_special_tokens=False, return_tensors="pt")
                    
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length= self.length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.k,
            top_p=self.p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.num_return_sequences,
        )

        # Decode text
        text_generated = self.tokenizer.decode(output_sequences[0][len(encoded_prompt[0]):], clean_up_tokenization_spaces=True)
        return text_generated


# unit test
'''
augmentor = generation('gpt2')

prompt_text = "Microsoft has said it will replace more than 14 million power cables for its Xbox consoles due to safety concerns."
prompt_text = "Versace art portfolio up for sale The art collection of murdered fashion designer Gianni Versace could fetch \
up to £9m ($17m) when it is auctioned in New York and \
London later this year. <eod> </s> <eos>"

augmentor.augment('science & technology')
'''


'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error
model = GPT2LMHeadModel.from_pretrained('gpt2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

texts = ["this is a first prompt", "this is a second prompt"]
encoding = tokenizer(texts, return_tensors='pt').to(device)
with torch.no_grad():
    generated_ids = model.generate(**encoding)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
'''
