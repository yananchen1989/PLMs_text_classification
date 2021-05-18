
from load_data import *
from transformers import pipeline
import random,torch
print(torch.__version__)
ds = load_data(dataset='ag', samplecnt=-1)


nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )






from transformers import pipeline

from transformers import AutoModelWithLMHead, AutoTokenizer
lang = 'zh'
model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang))
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang))
nlp = pipeline("translation_en_to_{}".format(lang), model=model, tokenizer=tokenizer, device=0)

results = nlp(ds.df.sample(1000)['content'].tolist(), max_length=128, do_sample=False)
print(results[0]['translation_text'])



nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)




contents = ds.df_train['content'].tolist()[:50]

nlp  = pipeline("text-generation", model='gpt2', device=0, return_full_text=False)
results = nlp(['safe conduct , however ambitious and well-intentioned , fails to hit the entertainment bull s - eye .'], max_length=100, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=1, num_return_sequences=16)

for ii in results:
    print('==>', ii['generated_text'])






import nltk; nltk.download('wordnet')
from eda import *



parser.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
parser.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
parser.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
parser.add_argument("--alpha_sr", required=False, type=float, default=0.1, help="percent of words in each sentence to be replaced by synonyms")
parser.add_argument("--alpha_ri", required=False, type=float, default=0.1, help="percent of words in each sentence to be inserted")
parser.add_argument("--alpha_rs", required=False, type=float, default=0.1, help="percent of words in each sentence to be swapped")
parser.add_argument("--alpha_rd", required=False, type=float, default=0.1, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = parts[0]
        sentence = parts[1]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "\t" + aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))

#main function
if __name__ == "__main__":

    #generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)








