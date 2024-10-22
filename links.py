### MURDER ON THE LINKS

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
from data_processing import preprocess, save_metadata
import os


tokenizer = MWETokenizer([
        ('hercule', 'poirot'),
        ('mr.', 'poirot'),
        ('marthe', 'daubreuil'),
        ('marthe', 'beroldy'),
        ('jack', 'renauld'),
        ('paul', 'renauld')
    ], separator='')

word_dict = {
    'marthedaubreuil': 'marthedaubreuil', #technically not necessary if mapping to itself but here for clarity
    'martheberoldy': 'marthedaubreuil',
    'marthe': 'marthedaubreuil',
    'herculepoirot': 'poirot',
    'poirot': 'poirot',
    'mr.poirot': 'poirot',
    'jackrenauld': 'jack',
    'jack': 'jack',
    'paulrenauld': 'paul',
    'paul': 'paul'
}

links_tokens = preprocess("./Data/books/The-Murder-on-the-Links.txt", tokenizer=tokenizer, word_dict=word_dict)

model = Word2Vec(links_tokens, vector_size=100, window=5, min_count=1, sg=0) #CBOW model


# Train model on novels as specified
model.build_vocab(links_tokens)
model.train(links_tokens, total_examples=model.corpus_count, epochs=5)


words = list(model.wv.index_to_key) # Gets words from models vocabularly                             
w_vec = np.array([model.wv[word] for word in words]) # Word Vectors

print(words[:10])


# Then separate our Styles word vector from the rest 
# styles_words = set([word for sentence in links_tokens for word in sentence])
# filtered_words = [word for word in styles_words if word in model.wv]

np.savetxt('./projector_files/links_mwe.tsv', w_vec, delimiter='\t')

with open('./projector_files/mwe_meta_links.tsv', 'w', encoding='utf-8') as f:
    for word in words:
        f.write(f"{word}\n")

print('end')