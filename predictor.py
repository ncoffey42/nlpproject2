import getpass
import os

from dotenv import load_dotenv
# import streamlit as st

load_dotenv()  # take environment variables from .env.

from langchain_community.document_loaders import TextLoader
from data_processing import pull_from_book_metadata


iterations = 1 #Number of times to run the program
books = ["Styles", "Ackroyd", "Links"] #Books to check (REMEMBER TO HAVE METADATA FOR GROUND TRUTH FOR THIS BOOK)


#from spacy_processing import preprocess

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

#This is a lot of code, but basically this is taking the characters and their aliases and creating a dictionary to make sure they are treated as the same character. This is also used with preprocessing the texts
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

#Initializing stat dicts for later
times_correct = {
    "Styles": 0,
    "Ackroyd": 0,
    "Links": 0
}

times_correct_parts = {
    "Protagonist": 0,
    "Victim": 0,
    "Antagonist": 0,
    "Climax Chapter": 0,
    "Murder Weapon": 0
}

#The actual loop -> runs on each of the books a certain number of times and keeps track of what it got correct
# You can run this a number of times because LLM's may give different results, given their random nature
# Or (by default) just run it once and see what happens

for i in range(iterations):
    for book in books:

        #Load in the book -> preprocessed and cleaned up first
        loader = TextLoader("./Data/book_tokens/" + book + ".txt")

        docs = loader.load()

        #Ground truthed answers
        answers = pull_from_book_metadata(f"./Data/book_metadata/{book}.txt")

        # langchain RAG stuff here -> straight from documentation
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

        #System prommpt that basically got us to only have one word answers
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

        #Start the test
        print(book)

        correct_answers = 0

        # Each of these "Invoke" calls is a question that the LLM is asked to answer -> this calls from the OpenAI API. We are using 4o mini for cost reasons
        results = rag_chain.invoke({"input": "Who is the victim?"})
        print("Victim:", end=" ")
        print(results["answer"])

        # If the one-word answer is in the ground truth, then it is correct. Lowercasing the LLM answer because it can be finicky
        if answers["victim"] in results["answer"].lower():
            print("Correct")
            correct_answers += 1
            times_correct[book] += 1  
            times_correct_parts["Victim"] += 1  

        results = rag_chain.invoke({"input": "Who is the protagonist?"})
        print("Protagonist:", end=" ")
        print(results["answer"])

        if answers["protagonist"] in results["answer"].lower():
            print("Correct")
            correct_answers += 1 
            times_correct[book] += 1   
            times_correct_parts["Protagonist"] += 1

        results = rag_chain.invoke({"input": "Who is the murderer?"})
        print("Murderer:", end=" ")
        print(results["answer"])

        if answers["antagonist"] in results["answer"].lower():
            print("Correct")
            correct_answers += 1 
            times_correct[book] += 1   
            times_correct_parts["Antagonist"] += 1

        results = rag_chain.invoke({"input": "Which chapter does the climax occur?"})
        print("Climax:", end=" ")
        print(results["answer"])
        
        if answers["climax chapter"] in results["answer"].lower():
            print("Correct")
            correct_answers += 1 
            times_correct[book] += 1   
            times_correct_parts["Climax Chapter"] += 1

        results = rag_chain.invoke({"input": "What was the murder weapon?"})
        print("Murder Weapon:", end=" ")
        print(results["answer"])
        
        if answers["murder weapon"] in results["answer"].lower():
            print("Correct")
            correct_answers += 1 
            times_correct[book] += 1   
            times_correct_parts["Murder Weapon"] += 1

        print(f"{correct_answers} out of 5 correct")

#Print final stats at the end
print("Times correct:", times_correct)
print(f"(This is out of {iterations*5} times)")

print("Parts correct:", times_correct_parts)
print(f"(This is out of {iterations*len(books)} times)")