from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer
import sys
import csv

# Function for preprocessing: Normalization & Tokenization, Stopword removal, OPTIONALLY investigate Lemmatization 
def preprocess(file_path, tokenizer, word_dict):
    with open(file_path, encoding='utf-8') as f:
        text = f.read().lower() # Lowercase


    # Mutli-word expression to separate the Inglethorps
    mwe_tokenizer = tokenizer


    # Tokenization
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]

    mwe_tokens = [mwe_tokenizer.tokenize(sentence) for sentence in tokens]

    print(mwe_tokens[:50], file=sys.stderr)


    # if 'mr._inglethorp' in mwe_tokens:
    #     print("MR INGLETHORP!")
    #flattened_tokens = [word for sentence in tokens for word in sentence]
    #print(flattened_tokens)

    #if 'mrs._inglethorp' in flattened_tokens:
    #    print("mrsInglethorp!!!")



    print("after", file=sys.stderr)
    # Substitute references to the same characters
    sub_tokens = []
    for sentence in tokens:
        sub_sentence = []
        for word in sentence:
            if word in word_dict.keys():
                #print to stderr
                print("word found:", word, word_dict[word], file = sys.stderr)
                sub_sentence.append(word_dict[word])
            else:
                sub_sentence.append(word)
        sub_tokens.append(sub_sentence)  # Keep sentences as lists of tokens

    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    clean_tokens = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in sub_tokens]

    #lemmatizer = WordNetLemmatizer()
    # Part of Speech = Verb
    #tokens = [[lemmatizer.lemmatize(word, pos='v') for word in sentence] for sentence in tokens]

    #return sub_tokens
    return clean_tokens


def save_metadata(vocab, metadata_file_path):
    with open(metadata_file_path, 'w') as f:
        for token in vocab:
            f.write(f"{token}\n")


# def coref(file_path):
#     # Step 1: Read the text
#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read().lower()  # Lowercase for normalization

#     # Step 2: Apply coreference resolution
#     doc = nlp(text)
#     resolved_text = doc._.coref_resolved

#     # Step 3: Apply NLTK's sent_tokenize to split text into sentences
#     sentences = sent_tokenize(resolved_text)

#     # Step 4: Apply MWE Tokenizer for character names
#     mwe_tokenizer = MWETokenizer([
#         ('alfred', 'inglethorp'),
#         ('mr.', 'inglethorp'),
#         ('emily', 'inglethorp'),
#         ('mrs.', 'inglethorp')
#     ], separator='_')

#     # Tokenize the resolved sentences into words and apply MWE tokenizer
#     mwe_tokens = [mwe_tokenizer.tokenize(word_tokenize(sentence)) for sentence in sentences]

#     # Step 5: Flatten the tokens into a single list
#     tokens = [word for sentence in mwe_tokens for word in sentence]

#     # Step 6: Remove stop words and lemmatize
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     cleaned_tokens = [
#         lemmatizer.lemmatize(word) for word in tokens
#         if word.isalnum() and word not in stop_words
#     ]

#     return cleaned_tokens