from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer
import csv

# Function for preprocessing: Normalization & Tokenization, Stopword removal, OPTIONALLY investigate Lemmatization 
def preprocess(file_path):
    with open(file_path, encoding='utf-8') as f:
        text = f.read().lower() # Lowercase


    # Mutli-word expression to separate the Inglethorps
    mwe_tokenizer = MWETokenizer([
        ('alfred', 'inglethorp'),
        ('mr.', 'inglethorp'),
        ('emily', 'inglethorp'),
        ('mrs.', 'inglethorp')
    ], separator='_')


    # Tokenization
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]

    mwe_tokens = [mwe_tokenizer.tokenize(sentence) for sentence in tokens]

    print(mwe_tokens[:50])


    if 'mr._inglethorp' in mwe_tokens:
        print("MR INGLETHORP!")
    #flattened_tokens = [word for sentence in tokens for word in sentence]
    #print(flattened_tokens)

    #if 'mrs._inglethorp' in flattened_tokens:
    #    print("mrsInglethorp!!!")



    print("after")
    # Substitute references to the same characters
    sub_tokens = []
    for sentence in tokens:
        sub_sentence = []
        for word in sentence:
            if word in ['alfred', 'alfred_inglethorp', 'mr._inglethorp', 'inglethorp']:
                sub_sentence.append('alfredinglethorp')
            elif word in ['emily', 'emily_inglethorp', 'mrs._inglethorp']:
                sub_sentence.append('emilyinglethorp')
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