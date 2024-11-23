import getpass
import os

from dotenv import load_dotenv
# import streamlit as st

load_dotenv()  # take environment variables from .env.

from langchain_community.document_loaders import TextLoader
#from spacy_processing import preprocess
file_path = "Data/books/The-Mysterious-Affair-at-Styles.txt"

# with open(file_path) as f:
#     text = f.read()

#preprocess the text

# from data_processing import preprocess

# text = preprocess(text)

#save the text


#load only chapter XIII

# text = text.split("CHAPTER XIII")[1]

# text = text.split("CHAPTER XIV")[0]

# with open("Data/books/The-Mysterious-Affair-at-Styles-Chapter-XIII.txt", "w") as f:
#     f.write(text)
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
    'dr. sheppard': 'jamesSheppard',
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



# text = preprocess(file_path, styles_dict, styles_character_ner)
#loader = TextLoader(file_path)
for path in ["./Data/book_tokens/Styles.txt", "./Data/book_tokens/Ackroyd.txt", "./Data/book_tokens/Links.txt"]:
    loader = TextLoader(path)

    docs = loader.load()

    # print(len(docs))

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")

    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever()

    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = (
        "You are an assistant for answering questions about Agatha Christie Mystery novels."
        "Use the following retrieved context to help answer the question"
        # "Use three sentences maximum and keep the answer concise"
        "Answer with one word in this format, 'The Answer is: '"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print(path)

    results = rag_chain.invoke({"input": "Who is the victim?"})
    print("Victim:", end=" ")
    print(results["answer"])

    results = rag_chain.invoke({"input": "Who is the protagonist?"})
    print("Protagonist:", end=" ")
    print(results["answer"])

    results = rag_chain.invoke({"input": "Who is the murderer?"})
    print("Murderer:", end=" ")
    print(results["answer"])

    results = rag_chain.invoke({"input": "Which chapter does the climax occur?"})
    print("Climax:", end=" ")
    print(results["answer"])