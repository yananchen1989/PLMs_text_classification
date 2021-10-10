
#. ppo
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt


from transformers import GPT2Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./cache', local_files_only=True)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 

#def get_ppo_trainer(ft_path, device, params):
    # if ft_pattern in ['tc', 'pp']:
    #     gpt2_tokenizer.sep_token = '<|sep|>'
    #     gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained('gpt2_{}_ft_on_ccnews'.format(ft_pattern))
    #     gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained('gpt2_{}_ft_on_ccnews'.format(ft_pattern))
        
    #elif ft_pattern in ['no']:
params = vars(args)
gpt2_tokenizer.sep_token = ''
gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained(ft_path)
gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained(ft_path)
config = {
    # "lm_name": "lvwerra/gpt2-imdb",
    # "ref_lm_name": "lvwerra/gpt2-imdb",
     "cls_model_name": "lvwerra/bert-imdb",
    #"tk_name": "gpt2",
    #"steps": 25600,
    "forward_batch_size": params['forward_batch_size'],
    "ppo_epochs": 4,   
    #"txt_in_len": 5,
    #"txt_out_len": 15,
    "batch_size": params['ppo_batchsize'] ,
    "lr": 1.41e-5,
    "init_kl_coef":params['init_kl_coef'],
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": params['cliprange'],
    "cliprange_value": params['cliprange_value'],
    "vf_coef":.1, 
}
gpt2_model_ref_trl.to(device)
gpt2_model_trl.to(device)
ppo_trainer = PPOTrainer(gpt2_model_trl, gpt2_model_ref_trl, **config)
#    return ppo_trainer, gpt2_model_trl, gpt2_model_ref_trl


def reponse(df_batch, gpt2_model_trl, maxlen, gpt2_tokenizer, device, params):

    query_tensors = gpt2_tokenizer(df_batch['query'].tolist(), return_tensors="pt", pad_to_max_length =True, \
                                truncation=True, padding=True, max_length=maxlen_query)['input_ids'].to(device)

    response_tensors = respond_to_batch(gpt2_model_trl, query_tensors,
                                      txt_len = maxlen, top_p=0.9,  \
                                      temperature=params['temperature'], \
                                      min_tokens_to_keep = params['min_tokens_to_keep'] )

    df_batch['response'] = [gpt2_tokenizer.decode(response_tensor, clean_up_tokenization_spaces=True, skip_special_tokens=True).strip() \
                                for response_tensor in response_tensors]
    return df_batch, query_tensors, response_tensors


def reponse_single(df_batch, gpt2_model_trl, maxlen, gpt2_tokenizer, device, params):
    responses = []
    query_tensors_ll, response_tensors_ll = [], []
    query_token_lens = []
    for query in df_batch['query'].tolist():
        # query_tensor = gpt2_tokenizer.encode([query+ ' {}'.format(gpt2_tokenizer.sep_token) ], return_tensors="pt", \
        #                         pad_to_max_length =True, \
        #                         truncation=True, padding=True, max_length=maxlen)['input_ids'].to(device)
        query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt").to(device)
        # response_tensor = respond_to_batch(gpt2_model_trl, query_tensor,
        #                               txt_len = maxlen, top_p=0.9,  \
        #                               temperature = params['temperature'], \
        #                               min_tokens_to_keep= params['min_tokens_to_keep'])
        response_tensor  = respond_to_batch(gpt2_model_trl, query_tensor)
        response = gpt2_tokenizer.decode(response_tensor[0], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()


        # get reward from another component
        train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
        # print(query_tensor.shape[1], response_tensor.shape[1] )
        # print("query==>", query)
        # print("response==>", response, '\n')
        responses.append(response)
        #query_tensors_ll.append(query_tensor)
        response_tensors_ll.append(response_tensor)
        
        query_token_lens.append(query_tensor.shape[1])

    #query_tensors = torch.cat(query_tensors_ll)
    response_tensors = torch.cat(response_tensors_ll)
    query_tensors = gpt2_tokenizer((df_batch['query']+' {}'.format(gpt2_tokenizer.sep_token)).tolist(), return_tensors="pt", pad_to_max_length =True, \
                                truncation=True, padding=True, max_length=max(query_token_lens))['input_ids'].to(device)
    df_batch['response'] = responses
    return df_batch, query_tensors, response_tensors