# nlpproject2

## Deliverables

- Report: deliverables/NLP_Project_2.pdf
- Presentation slides: deliverables/NLP_Project_2.pptx

## Setup

### Project Setup

1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt`
3. Look at and run the ipynb files (if you want)
4. You can't run spaCy in the same environment as LangChain so they have to be in seperate environments

### Data Setup (to reproduce what we did)

1. Download the data from [Project Gutenberg](https://www.gutenberg.org/) (Sample code to do this is in `test.ipynb`).
2. Run `python3 nltk_downloads.py` to download the required nltk data.
3. Create a file with the contents of a Data/books and add it to the book folder. Create another file to store the books metadata following the format of `The-Murder-on-the-Links.txt` in the Data/book_metadata. Ensure that both of these files have the exact same name.
4. Create a python file that looks similar to `links.py` to store the links to the books you want to analyze. We ran `ackroyd.py`, `links.py` and `styles.py` for our analysis. Run the file when you are done to create the proper tensorboard files.

### Code Execution
1. All of the books are available in Data/books as well as the pre-processed version of each and their metadata
2. The code for our accuracy testing for short prompts is in predictor.py
3. The results for the long prompts is in long_answer_results.txt

Agatha Christie -- Implement a LangChain-based reasoning system that can interpret your author's novels and the plots.

## Docs

[Info Doc](https://docs.google.com/document/d/1SzlvAJDJ_J6TmqEKMKzm24yskDTdLdnPAg5V2sZGQQU/edit#heading=h.drgugs2suv61)

[LaTeX Code for Project 1](https://docs.google.com/document/d/1CAOgCz53kaDHoT7jyzDVTkqXNURDrYvL9NK5dGl2nt0/edit?tab=t.0)

[Slides for Project 1](https://docs.google.com/presentation/d/1ZUDemqwNvS-08xp7VWl1Ov0Y36tAkW7FoDArsWdbfEw/edit#slide=id.p)
