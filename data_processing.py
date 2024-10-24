from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer
import sys
import csv

# Function for preprocessing: Normalization & Tokenization, Stopword removal, OPTIONALLY investigate Lemmatization 
def preprocess(file_path, tokenizer, word_dict):
    '''
    args:

    file_path: str -- path to the book file
    tokenizer: MWETokenizer -- multi-word expression tokenizer (used for character names that should be treated as a single token, like ryan peruski should be ryanperuski)
    word_dict: dict -- dictionary mapping words to their corresponding character names (when multiple tokens should be treated as the same token, like ryanperuski and ryan both should map to Ryan)

    returns:
    cleaned tokens: as a 2D list of strings -- list of sentences, each sentence is a list of tokens

    console output:
    prints some testing to stderr. If you want to avoid, run with 2> /dev/null
    
    '''

    #Open file
    with open(file_path, encoding='utf-8') as f:
        text = f.read().lower() # Lowercase


    # Mutli-word expression to combine character names into one token
    mwe_tokenizer = tokenizer

    # Tokenization
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]

    mwe_tokens = [mwe_tokenizer.tokenize(sentence) for sentence in tokens]

    print(mwe_tokens[:50], file=sys.stderr)


    print("after", file=sys.stderr)

    # Substitute references to the same characters (using word_dict)
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

    #Lemmatize
    lemmatizer = WordNetLemmatizer()
    lem_tokens = [[lemmatizer.lemmatize(word, pos='v') for word in sentence] for sentence in clean_tokens]

    return lem_tokens


def save_metadata(vocab, metadata_file_path):
    '''
    args:
    vocab: list of strings -- list of tokens
    metadata_file_path: str -- path to the metadata file

    returns: None

    Writes the tokens to the metadata file
    
    '''
    with open(metadata_file_path, 'w') as f:
        for token in vocab:
            f.write(f"{token}\n")