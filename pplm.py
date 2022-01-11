#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse,os,gensim,joblib
import json,operator
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

# from pplm_classification_head import ClassificationHead

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}
BASE_NLI ={
    'politics':['Politics','War', 'Election','Constitution','Democracy','Conflict','Military',\
                'Terrorism', 'Government', 'Ideology', 'fascism', 'Socialism', 'Totalitarian', 'Religion'],
    'law':      ['Law', 'Legitimacy','Court','Crime','Murder','Jurisdiction'],
    'science':  ['Science','Aerospace','Physics','Chemistry','Biology','Scientist','Astronomy','Universe','Big Bang'],
    'technology':['Technology','Biotech', 'IT','Computers','Internet','Algorithm','Space','Bitcoin','artificial Intelligence','Robot'],
    'health': ['Health','Healthcare','Medicine','Clinics','Vaccine','Wellness','Nutrition','Dental','HIV','Disease'],
    'business': ['Business','Finance','Oil price','Supply','Inflation','Dollars','Bank','Wall Street','Bitcoin',
                        'Federal Reserve','Accrual','Accountancy','Sluggishness','Consumerism','Trade','Quarterly earnings',\
                         'Deposit','Revenue','Stocks','Recapitalization','Marketing','Futures'],
    'sports': ['Sports','Athletics','Championships','Football','Olympic','Tournament','Chelsea','League','Golf',
                            'NFL','Super bowl','World Cup']
}

prefix_cond = ["In summary", "This essay discusses", "Views on", "The connection", "Foundational to this is",\
     "To review,", "In brief,", "An illustration of", "Furthermore,", "The central theme", \
     "To conclude,", "The key aspect", "Prior to this", "Emphasised are", "To summarise", \
     "The relationship", "More importantly,", "It has been shown", "The issue focused on", "In this essay"]

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument(
    "--pretrained_model",
    "-M",
    type=str,
    default="gpt2-medium",
    help="pretrained model name or path to local checkpoint",
)
parser.add_argument(
    "--cond_text", type=str, default="To conclude",
    help="Prefix texts to condition on"
)
parser.add_argument(
    "--uncond", action="store_true",
    help="Generate from end-of-text as prefix"
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=1,
    help="Number of samples to generate from the modified latents",
)
# parser.add_argument(
#     "--bag_of_words",
#     "-B",
#     type=str,
#     default=None,
#     help="Bags of words used for PPLM-BoW. "
#          "Either a BOW id (see list in code) or a filepath. "
#          "Multiple BoWs separated by ;",
# )
parser.add_argument(
    "--discrim",
    "-D",
    type=str,
    default=None,
    choices=("clickbait", "sentiment", "toxicity", "generic"),
    help="Discriminator to use",
)
parser.add_argument('--discrim_weights', type=str, default=None,
                    help='Weights for the generic discriminator')
parser.add_argument('--discrim_meta', type=str, default=None,
                    help='Meta information for the generic discriminator')
parser.add_argument(
    "--class_label",
    type=int,
    default=-1,
    help="Class label used for the discriminator",
)
parser.add_argument("--length", type=int, default=100)
parser.add_argument("--stepsize", type=float, default=0.02)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument(
    "--sample", action="store_true",
    help="Generate from end-of-text as prefix"
)
parser.add_argument("--num_iterations", type=int, default=3)
parser.add_argument("--grad_length", type=int, default=10000)
parser.add_argument(
    "--window_length",
    type=int,
    default=0,
    help="Length of past which is being optimized; "
         "0 corresponds to infinite window length",
)
parser.add_argument(
    "--horizon_length",
    type=int,
    default=1,
    help="Length of future to optimize over",
)
parser.add_argument("--decay", action="store_true",
                    help="whether to decay or not")
parser.add_argument("--gamma", type=float, default=1.5)
parser.add_argument("--gm_scale", type=float, default=0.9)
parser.add_argument("--kl_scale", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", default=7, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")



def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    # if torch.cuda.is_available() and device == 'cuda':
    #     x = x.cuda()
    # elif device != 'cuda':
    #     x = x.to(device)
    return Variable(x.to(device), requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01):
    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        #print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            #print(" pplm_bow_loss:", loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past,
                    inputs_embeds=inputs_embeds
                )
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)

            prediction = classifier(new_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))

            label = torch.tensor(prediction.shape[0] * [class_label],
                                 device=device,
                                 dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            # print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            # print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        # print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter



def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices

def get_s3_words(ll):
    filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[ll])
    with open(filepath, "r") as f:
        words = f.read().strip().split("\n")
    return words


def get_seed_words():
    gram_diff = joblib.load("gram_diff___{}".format(args.dsn))
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin',binary=True)
    vocab_w2v = set(list(model_w2v.index_to_key))
    label_expands_auto = {}
    for l, gram_scores in gram_diff.items():
        gram_scores_sum = {g:round(np.array(scores).sum(),4) for g, scores in gram_scores.items() }
        gram_scores_sum_sort = sorted(gram_scores_sum.items(), key=operator.itemgetter(1), reverse=True) 
        gram_scores_mean = {g:round(np.array(scores).mean(),4) for g, scores in gram_scores.items() }
        gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
        gram_scores_sort = gram_scores_sum_sort + gram_scores_mean_sort
        label_expands_auto[l] = set()
        for j in gram_scores_sort:
            if j[0] not in vocab_w2v or j[0] in ['news']:
                #print(j[0])
                continue
            if ' and ' in l:
                w0 = l.split('and')[0].strip().lower()
                w1 = l.split('and')[1].strip().lower()
                simi = max(model_w2v.similarity(w0, j[0]), model_w2v.similarity(w1, j[0]) )
            else:
                simi = model_w2v.similarity(l.lower(), j[0])
            if simi >= 0.1:
                label_expands_auto[l].add(j[0])
            if len(label_expands_auto[l])-1 == 64:
                break 
        if ' and ' in l:
            label_expands_auto[l].add(l.split('and')[0].strip())
            label_expands_auto[l].add(l.split('and')[1].strip())
        else:
            label_expands_auto[l].add(l)
        for ll in BAG_OF_WORDS_ARCHIVE_MAP:
            if (ll in l.lower()) or ('world' in l.lower() and ll == 'politics') or \
                (('science' in l.lower() or 'technology' in l.lower()) and ll == 'space'):
                words_s3 = get_s3_words(ll)
                label_expands_auto[l].update(words_s3)
        for ll, expands in BASE_NLI.items():
            if ll in l.lower():
                label_expands_auto[l].update( [w.lower() for w in expands ] )
        print(l, label_expands_auto[l], '\n')
    return label_expands_auto


from utils.load_data import * 
ds = load_data(dataset=args.dsn, samplecnt= 8)
labels_candidates = ds.df_train['label_name'].unique().tolist()
print(labels_candidates)

label_expands_auto = get_seed_words()
print(label_expands_auto)
def get_bag_of_words_indices_dsn_label(words, tokenizer):
    bow_indices = []
    bow_indices.append(
        [tokenizer.encode(word.strip(),
                          add_prefix_space=True,
                          add_special_tokens=False)
         for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
        model,
        tokenizer,
        cond_text=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        **kwargs
):
    classifier, class_id = None, None #get_classifier(
    #     discrim,
    #     class_label,
    #     device
    # )

    #bow_indices = []
    if bag_of_words:
        #bow_indices = get_bag_of_words_indices(bag_of_words.split("_"), tokenizer)
        bow_indices = get_bag_of_words_indices_dsn_label(bag_of_words, tokenizer)
    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    context = tokenizer.encode(
                tokenizer.bos_token + cond_text,
                add_special_tokens=False
            )

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False
    )
    torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    for i in range(length):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            # print(
            #     "unperturbed discrim loss",
            #     unpert_discrim_loss.data.cpu().numpy()
            # )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        #print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time



# set Random seed
# torch.manual_seed(seed)
# np.random.seed(seed)


# set the device

# load pretrained model
model = GPT2LMHeadModel.from_pretrained(args.pretrained_model, output_hidden_states=True, cache_dir='./cache')
model.to(device)
model.eval()
# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model, cache_dir='./cache')
# Freeze GPT-2 weights
for param in model.parameters():
    param.requires_grad = False






ite = 0
infos = []

while True:
    for label_name in labels_candidates:
        bag_of_words = label_expands_auto[label_name]
        print("ite:{}".format(ite), "bow:{}".format(bag_of_words))

        for cond_text in prefix_cond:
            
            unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
                model=model,
                tokenizer=tokenizer,
                cond_text=cond_text,
                device=device,
                num_samples=args.num_samples,
                bag_of_words= bag_of_words,
                discrim= args.discrim,
                class_label= args.class_label,
                length= args.length,
                stepsize= args.stepsize,
                temperature= args.temperature,
                top_k= args.top_k,
                sample= args.sample,
                num_iterations= args.num_iterations,
                grad_length= args.grad_length,
                horizon_length= args.horizon_length,
                window_length= args.window_length,
                decay= args.decay,
                gamma= args.gamma,
                gm_scale= args.gm_scale,
                kl_scale= args.kl_scale
            )

            # untokenize unperturbed text
            unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0], skip_special_tokens=True)
            print("ori===>", unpert_gen_text.replace('\n', ' ').strip())
            print()

            for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
                pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0], skip_special_tokens=True)
                print("{}-Perturbed===>".format(i + 1), pert_gen_text.replace('\n', ' ').strip() )
                print()
                infos.append((label_name, bag_of_words, unpert_gen_text.replace('\n', ' ').strip(), pert_gen_text.replace('\n', ' ').strip() ))

    df = pd.DataFrame(infos, columns=['label_name', 'bag_of_words', 'content_ori', 'content_pplm_syn'])    
    df.to_csv("{}_pplm_gen.csv".format(args.dsn), index=False)
    ite += 1




