### MURDER OF ROGER ACKROYD

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
from data_processing import preprocess, save_metadata, pull_from_book_metadata, calculate_embeddings
import os

BOOK = "The-Murder-of-Roger-Ackroyd"

tokenizer = MWETokenizer([
        ('hercule', 'poirot'),
        ('mr.', 'poirot'),
        ('dr.', 'shepherd'),
        ('dr.', 'james', 'shepherd'),
        ('james', 'shepherd'),
        ('roger', 'ackroyd')
    ], separator='')

word_dict = {
    'jamesshepherd': 'jamesshepherd', #technically not necessary if mapping to itself but here for clarity
    'dr.jamesshepherd': 'jamesshepherd',
    'shepherd': 'jamesshepherd',
    'james': 'jamesshepherd',
    'herculepoirot': 'poirot',
    'poirot': 'poirot',
    'mr.poirot': 'poirot',
    'roger': 'roger',
    'rogerackroyd': 'roger',
}

for lem in [True, False]:
    tokens = preprocess(f"./Data/books/{BOOK}.txt", tokenizer, word_dict, lem=lem)

    for algorithm in ["CBOW", "Skip-Gram"]:
        #model
        if algorithm == "CBOW": 
            model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, sg=0) #CBOW model
        else:
            model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, sg=1)

        # Train model on novels as specified
        model.build_vocab(tokens)
        model.train(tokens, total_examples=model.corpus_count, epochs=5)

        # Gets words from models vocabularly   
        words = list(model.wv.index_to_key)

        # Word Vectors
        w_vec = np.array([model.wv[word] for word in words])

        #Save files
        lemmatization = "with" if lem else "without"
        np.savetxt(f'./projector_files/{BOOK}-{algorithm}-{lemmatization}lem_mwe.tsv', w_vec, delimiter='\t')
        save_metadata(words, f'./projector_files/mwe_meta_{BOOK}-{algorithm}-{lemmatization}lem.tsv')

        #Add to projector_config.pbtxt (if it doesn't exist, create it)
        #If the embedding path exists, don't add it again

        write_config = True

        if not os.path.exists('./projector_files'):
            os.makedirs('./projector_files')
        
        if not os.path.exists('./projector_files/projector_config.pbtxt'):
            with open('./projector_files/projector_config.pbtxt', 'w') as f:
                pass
        with open(f'./projector_files/projector_config.pbtxt', 'r') as f:
            if f'    tensor_path: "{BOOK}-{algorithm}-{lemmatization}lem_mwe.tsv"\n' in f.read():
                write_config = False
        if write_config:
            with open(f'./projector_files/projector_config.pbtxt', 'a') as f:
                f.write('embeddings: {\n')
                f.write(f'    tensor_name: "{BOOK} {algorithm} {"with" if lem else "without"} lemmatization"\n')
                f.write(f'    tensor_path: "{BOOK}-{algorithm}-{lemmatization}lem_mwe.tsv"\n')
                f.write(f'    metadata_path: "mwe_meta_{BOOK}-{algorithm}-{lemmatization}lem.tsv"\n')
                f.write('}\n')

        #Find distances between characters by grabbing from the metadata file

        book_dict = pull_from_book_metadata(f"Data/book_metadata/{BOOK}.txt")

        print("\n", algorithm, f"({lemmatization} lemmatization)", end="\n\n")
        print("Distance between protagionist and antagionist:", calculate_embeddings(model, book_dict["protagonist"], book_dict["antagonist"]))
        print("Distance between protagionist and murder weapon:", calculate_embeddings(model, book_dict["protagonist"], book_dict["murder weapon"]))
        print("Distance between antagionist and murder weapon:", calculate_embeddings(model, book_dict["antagonist"], book_dict["murder weapon"]))
        print("Distance between victim and murder weapon", calculate_embeddings(model, book_dict["victim"], book_dict['murder weapon']))
        print("Distance between victim and antagionist", calculate_embeddings(model, book_dict["victim"], book_dict["antagonist"]))

print('end')