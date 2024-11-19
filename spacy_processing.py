import spacy
from spacy.matcher import PhraseMatcher
from spacy.symbols import ORTH
import re
import coreferee
from gensim.models import Word2Vec
import numpy as np

nlp = spacy.load("en_core_web_lg")

def preprocess(file_path, word_dict, character_ner, lem=0):

    '''
    args:
    file_path: str -- path to the book file
    word_dict: dict -- dictionary mapping words to their corresponding character names
    character_ner -- list of dictonaries, used for custom Named Entity Recognition labelling
        Example:
            character_ner = [
            {"label": "PERSON", "pattern": "alfredInglethorp"},
            {"label": "PERSON", "pattern": "emilyInglethorp"}
            ]
    lem: bool -- whether to lemmatize the tokens or not

    returns:
    cleaned tokens: as a 2D list of strings -- list of sentences, each sentence is a list of tokens
    '''


    with open(file_path, encoding='utf-8') as f:
        text = f.read().lower()
     

    # Regular expression used to find references to a character within the novel
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in word_dict.keys()) + r')\b')

    # Substitutes varying character aliases to one name
    text = pattern.sub(lambda match : word_dict[match.group(0)], text)

    #print(text)

    # Adds custom NER for the novel characters - 'alfredInglethorp' == PERSON
    ruler = nlp.add_pipe('entity_ruler', before='ner')
    ruler.add_patterns(character_ner)


    # Coreference resolution
    # John went home, then he went to bed. --> John went home, then John went to bed.
    nlp.add_pipe('coreferee')

    # Run spacy pipeline
    doc = nlp(text)

    # Use Coreference resolution to get the resolved text
    # Code from https://github.com/explosion/spaCy/discussions/12142
    resolved_text = ""
    for token in doc:
    
        repres = doc._.coref_chains.resolve(token)
        #print(repres)
        if repres:
            resolved_text += " " + " and ".join([t.text for t in repres])
        else:
            resolved_text += " " + token.text
        
    #print(resolved_text)

    # Run the spacy pipeline again to then clean and return our tokens
    doc = nlp(resolved_text)

    # Token processing - removal of stopwords and non-alphanumeric characters. Return clean 2d list of [sentences][tokens]
    sentences = []
    for sent in doc.sents:
        tokens = []
        for token in sent:
            if token.is_alpha and not token.is_stop:
                tokens.append(token.text)
        sentences.append(tokens)
    
    #print(sentences[:6])
    return sentences


def save_metadata(vocab, metadata_file_path):
    '''
    args:
    vocab: list of strings -- list of tokens
    metadata_file_path: str -- path to the metadata file

    returns: None

    Writes the tokens to the metadata file
    
    '''
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f"{token}\n")


## Testing using The Mysterious Affair at Styles
test_dict = {
    'mr. inglethorp': 'alfredInglethorp',
    'mrs. inglethorp': 'emilyInglethorp',
    'alfred inglethorp' : 'alfredInglethorp',
    'inglethorp' : 'alfredInglethorp',
    'emily inglethorp' : 'emilyInglethorp',

    'hercule poirot' : 'Poirot',
    'poirot' : 'Poirot',
    'detective poirot' : 'Poirot',

    'mr. hastings' : 'Hastings',
    'hastings' : 'Hastings',

    'james japp' : 'Japp',
    'inspector' : 'Japp',
    'japp' : 'Japp',

    'john cavendish' : 'johnCavendish',
    'john' : 'johnCavendish',
    'mr. john cavendish' : 'johnCavendish',
    'mr. john' : 'johnCavendish',
    'mr. cavendish' : 'johnCavendish',


    'lawrence cavendish' : 'lawrenceCavendish',
    'mr. lawrence cavendish' : 'lawrenceCavendish',
    'lawrence' : 'lawrenceCavendish',
    'mr. lawrence' : 'lawrenceCavendish',
    
    'mrs. cavendish' : 'maryCavendish',
    'mary cavendish' : 'maryCavendish',
    'mary' : 'maryCavendish',

    'evelyn' : 'evelynHoward',
    'evelyn howard' : 'evelynHoward',
    'howard' : 'evelynHoward',
    'miss howard' : 'evelynHoward',

    'cynthia murdoch' : 'cynthiaMurdoch',
    'miss murdoch' : 'cynthiaMurdoch',
    'cynthia' : 'cynthiaMurdoch',
    'murdoch' : 'cynthiaMurdoch',

    'dr. bauerstein' : 'Bauerstein',
    'bauerstein' : 'Bauerstein',

    'dorcas' : 'Dorcas',
}


character_ner = [
        {"label": "PERSON", "pattern": "alfredInglethorp"},
        {"label": "PERSON", "pattern": "emilyInglethorp"},
        {"label": "PERSON", "pattern": "Poirot"},
        {"label": "PERSON", "pattern": "Hastings"},
        {"label": "PERSON", "pattern": "Japp"},
        {"label": "PERSON", "pattern": "johnCavendish"},
        {"label": "PERSON", "pattern": "lawrenceCavendish"},
        {"label": "PERSON", "pattern": "maryCavendish"},
        {"label": "PERSON", "pattern": "evelynHoward"},
        {"label": "PERSON", "pattern": "cynthiaMurdoch"},
        {"label": "PERSON", "pattern": "Bauerstein"},
        {"label": "PERSON", "pattern": "Dorcas"},

    ]


#book = preprocess("./Data/books/test-book.txt", test_dict, character_ner)
#print(book)

tokens = preprocess("./Data/books/The-Mysterious-Affair-at-Styles.txt", test_dict, character_ner)

model = Word2Vec(tokens, vector_size=100, window=5, min_count=2, sg=0) #CBOW model

model.build_vocab(tokens)
model.train(tokens, total_examples=model.corpus_count, epochs=10)

words = list(model.wv.index_to_key)

# Word Vectors
w_vec = np.array([model.wv[word] for word in words])
np.savetxt(f'./proj2_projector_files/styles3.tsv', w_vec, delimiter='\t')
save_metadata(words, "./proj2_projector_files/meta_styles3.tsv")
