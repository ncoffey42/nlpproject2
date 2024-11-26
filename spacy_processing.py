import spacy
from spacy.matcher import PhraseMatcher
from spacy.symbols import ORTH
import re
import coreferee
from gensim.models import Word2Vec
import numpy as np

nlp = spacy.load("en_core_web_lg")

def preprocess(file_path, word_dict=None, character_ner=None, coref=1):

    '''
    args:
    file_path: str -- path to the book file
    OPTIONAL word_dict: dict -- dictionary mapping words to their corresponding character names
    OPTIONAL character_ner -- list of dictonaries, used for custom Named Entity Recognition labelling
        Example:
            character_ner = [
            {"label": "PERSON", "pattern": "alfredInglethorp"},
            {"label": "PERSON", "pattern": "emilyInglethorp"}
            ]
    OPTIONAL lem: bool -- whether to lemmatize the tokens or not

    returns:
    cleaned tokens: as a 2D list of strings -- list of sentences, each sentence is a list of tokens
    '''


    with open(file_path, encoding='utf-8') as f:
        text = f.read().lower()
     

    # Regular expression used to find references to a character within the novel

    if word_dict:

        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in word_dict.keys()) + r')\b')

        # Substitutes varying character aliases to one name
        text = pattern.sub(lambda match : word_dict[match.group(0)], text)

    #print(text)

    # Adds custom NER for the novel characters - 'alfredInglethorp' == PERSON
    if character_ner:
        if 'entity_ruler' not in nlp.pipe_names:
            ruler = nlp.add_pipe('entity_ruler', before='ner')
        ruler.add_patterns(character_ner)


    # Coreference resolution
    # John went home, then he went to bed. --> John went home, then John went to bed.

    if coref:

        if 'coreferee' not in nlp.pipe_names:
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
    else:
        doc = nlp(text)

    # Token processing - removal of stopwords and non-alphanumeric characters. Return clean 2d list of [sentences][tokens]
    sentences = []
    for sent in doc.sents:
        tokens = []

        for token in sent:
            if token.is_alpha and not token.is_stop:
                tokens.append(token.text)
        if tokens:
            # sentences.append(tokens)
            sentences.append(" ".join(tokens))
    
    

    # print(sentences[:6])
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




#This is a lot of code, but basically this is taking the characters and their aliases and creating a dictionary to make sure they are treated as the same character. This is used with preprocessing the texts
styles_dict = {
    'mr. inglethorp': 'alfredinglethorp',
    'mrs. inglethorp': 'emilyinglethorp',
    'alfred inglethorp' : 'alfredinglethorp',
    'inglethorp' : 'alfredinglethorp',
    'emily inglethorp' : 'emilyinglethorp',
    'hercule poirot' : 'poirot',
    'poirot' : 'poirot',
    'detective poirot' : 'poirot',
    'mr. hastings' : 'hastings',
    'hastings' : 'hastings',
    'james japp' : 'japp',
    'inspector' : 'japp',
    'japp' : 'japp',
    'john cavendish' : 'johncavendish',
    'john' : 'johncavendish',
    'mr. john cavendish' : 'johncavendish',
    'mr. john' : 'johncavendish',
    'mr. cavendish' : 'johncavendish',
    'lawrence cavendish' : 'lawrencecavendish',
    'mr. lawrence cavendish' : 'lawrencecavendish',
    'lawrence' : 'lawrencecavendish',
    'mr. lawrence' : 'lawrencecavendish',
    'mrs. cavendish' : 'marycavendish',
    'mary cavendish' : 'marycavendish',
    'mary' : 'marycavendish',
    'evelyn' : 'evelynhoward',
    'evelyn howard' : 'evelynhoward',
    'howard' : 'evelynHoward',
    'miss howard' : 'evelynhoward',
    'cynthia murdoch' : 'cynthiamurdoch',
    'miss murdoch' : 'cynthiamurdoch',
    'cynthia' : 'cynthiamurdoch',
    'murdoch' : 'cynthiamurdoch',
    'dr. bauerstein' : 'bauerstein',
    'bauerstein' : 'bauerstein',
    'dorcas' : 'dorcas',
}

styles_character_ner = [
        {"label": "PERSON", "pattern": "alfredinglethorp"},
        {"label": "PERSON", "pattern": "emilyinglethorp"},
        {"label": "PERSON", "pattern": "poirot"},
        {"label": "PERSON", "pattern": "hastings"},
        {"label": "PERSON", "pattern": "japp"},
        {"label": "PERSON", "pattern": "johncavendish"},
        {"label": "PERSON", "pattern": "lawrencecavendish"},
        {"label": "PERSON", "pattern": "marycavendish"},
        {"label": "PERSON", "pattern": "evelynhoward"},
        {"label": "PERSON", "pattern": "cynthiamurdoch"},
        {"label": "PERSON", "pattern": "bauerstein"},
        {"label": "PERSON", "pattern": "dorcas"},

    ]

ackroyd_dict = {
    'roger ackroyd': 'rogerackroyd',
    'mr. ackroyd': 'rogerackroyd',
    'ackroyd': 'rogerackroyd',
    'ralph paton': 'ralphpaton',
    'ralph': 'ralphpaton',
    'mr. paton': 'ralphpaton',
    'flora ackroyd': 'floraackroyd',
    'flora': 'floraackroyd',
    'mrs. ackroyd': 'floraackroyd',
    'geoffrey raymond': 'geoffreyraymond',
    'raymond': 'geoffreyraymond',
    'geoffrey': 'geoffreyraymond',
    'major blunt': 'majorblunt',
    'blunt': 'majorblunt',
    'major': 'majorblunt',
    'caroline sheppard': 'carolinesheppard',
    'caroline': 'carolinesheppard',
    'mrs. sheppard': 'carolinesheppard',
    'dr. sheppard': 'jamessheppard',
    'sheppard': 'jamessheppard',
    'james sheppard': 'jamessheppard',
    'james': 'jamessheppard',
    'ursula bourne': 'ursulabourne',
    'ursula': 'ursulabourne',
    'miss bourne': 'ursulabourne',
    'parker': 'parker',
    'mrs. ferrars': 'mrsferrars',
    'ferrars': 'mrsferrars',
    'hercule poirot': 'poirot',
    'poirot': 'poirot',
    'detective poirot': 'poirot',
}

ackroyd_character_ner = [
        {"label": "PERSON", "pattern": "rogerackroyd"},
        {"label": "PERSON", "pattern": "ralphpaton"},
        {"label": "PERSON", "pattern": "floraackroyd"},
        {"label": "PERSON", "pattern": "geoffreyraymond"},
        {"label": "PERSON", "pattern": "majorblunt"},
        {"label": "PERSON", "pattern": "carolinesheppard"},
        {"label": "PERSON", "pattern": "jamessheppard"},
        {"label": "PERSON", "pattern": "ursulabourne"},
        {"label": "PERSON", "pattern": "parker"},
        {"label": "PERSON", "pattern": "mrsferrars"},
        {"label": "PERSON", "pattern": "poirot"},


    ]

links_dict = {
    'hercule poirot': 'poirot',
    'poirot': 'poirot',
    'detective poirot': 'poirot',
    'arthur hastings': 'hastings',
    'hastings': 'hastings',
    'captain hastings': 'hastings',
    'paul renault': 'paulrenault',
    'mr. renault': 'paulrenault',
    'renault': 'paulrenault',
    'madame renault': 'madamerenault',
    'mrs. renault': 'madamerenault',
    'eloise renault': 'madamerenault',
    'eloise': 'madamerenault',
    'jack renault': 'jackrenault',
    'jacques renault': 'jackrenault',
    'jack': 'jackrenault',
    'jacques': 'jackrenault',
    'giraud': 'giraud',
    'detective giraud': 'giraud',
    'm. giraud': 'giraud',
    'bella duveen': 'belladuveen',
    'bella': 'belladuveen',
    'miss duveen': 'belladuveen',
    'marta duveen': 'martaduveen',
    'marta': 'martaduveen',
    'francois': 'francois',
    'mrs. daubreuil': 'mrsdaubreuil',
    'madame daubreuil': 'mrsdaubreuil',
    'daubreuil': 'mrsdaubreuil',
    'cecile daubreuil': 'ceciledaubreuil',
    'cecile': 'ceciledaubreuil',
    'auguste': 'auguste',
    'auguste daubreuil': 'auguste',
    'gabrielle stoner': 'gabriellestoner',
    'gabrielle': 'gabriellestoner',
    'george connor': 'georgeconnor',
    'connor': 'georgeconnor',
    'george': 'georgeconnor',
}

links_character_ner = [
        {"label": "PERSON", "pattern": "poirot"},
        {"label": "PERSON", "pattern": "hastings"},
        {"label": "PERSON", "pattern": "paulrenault"},
        {"label": "PERSON", "pattern": "madamerenault"},
        {"label": "PERSON", "pattern": "jackrenault"},
        {"label": "PERSON", "pattern": "giraud"},
        {"label": "PERSON", "pattern": "belladuveen"},
        {"label": "PERSON", "pattern": "martaduveen"},
        {"label": "PERSON", "pattern": "francois"},
        {"label": "PERSON", "pattern": "mrsdaubreuil"},
        {"label": "PERSON", "pattern": "ceciledaubreuil"},
        {"label": "PERSON", "pattern": "auguste"},
        {"label": "PERSON", "pattern": "gabriellestoner"},
        {"label": "PERSON", "pattern": "georgeconnor"},
    ]


# book = preprocess("./Data/books/test-book.txt", styles_dict, character_ner, 0)
# print(book)

tokens = preprocess("./Data/books/Links_trimmed.txt", links_dict, links_character_ner, 0)

with open("./Data/book_tokens/Links_unresolved.txt", 'w', encoding='utf-8') as f:
    for sent in tokens:
        f.write(sent + "\n")
