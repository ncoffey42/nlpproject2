### MYSTERIOUS AFFAIR AT STYLES

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
from data_processing import preprocess, save_metadata


styles_tokens = preprocess("./Data/books/The-Mysterious-Affair-at-Styles.txt")


for sentence in styles_tokens:
    if 'alfred_inglethorp' in sentence:
        print("Albert Inglethorp found!")
    if 'emily_inglethorp' in sentence:
        print("Emily Inglethorp found!")

#print(styles_tokens[:50])  # Print first 50 tokens to check



# novel_tokens = []
# Loop through all novels in the folder
# for filename in os.listdir("./Data/books"):
#     if filename.endswith(".txt"):  # Process only .txt files
#         file_path = "./Data/books/" + filename
#         tokens = preprocess(file_path)  
#         novel_tokens += tokens  


# All novel tokens combines
#novel_tokens = styles_tokens + links_tokens + ackroyd_tokens
# print(len(novel_tokens))
# print(len(styles_tokens))

model = Word2Vec(styles_tokens, vector_size=100, window=5, min_count=1, sg=0) #CBOW model



# All novel tokens combines
#novel_tokens = styles_tokens + links_tokens + ackroyd_tokens
# print(len(novel_tokens))
print(len(styles_tokens))

#model = Word2Vec(styles_tokens, vector_size=100, window=5, min_count=1, sg=0) #CBOW model


# Train model on novels as specified
model.build_vocab(styles_tokens)
model.train(styles_tokens, total_examples=model.corpus_count, epochs=5)


words = list(model.wv.index_to_key) # Gets words from models vocabularly                             
w_vec = np.array([model.wv[word] for word in words]) # Word Vectors

print(words[:10])

#save_metadata(vocab, "./projector_files/mwe_metadata_styles.tsv")

# Then separate our Styles word vector from the rest 
# styles_words = set([word for sentence in styles_tokens for word in sentence])
# filtered_words = [word for word in styles_words if word in model.wv]

np.savetxt('./projector_files/styles_mwe.tsv', w_vec, delimiter='\t')

with open('./projector_files/mwe_meta_styles.tsv', 'w', encoding='utf-8') as f:
    for word in words:
        f.write(f"{word}\n")

print('end')