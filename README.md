# nlpproject

## Setup

### Project Setup

1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt`
3. Look at and run the ipynb files (if you want)

### Data Setup (to reproduce what we did)

1. Download the data from [Project Gutenberg](https://www.gutenberg.org/) (Sample code to do this is is `test.ipynb`).
2. Run `python3 nltk_downloads.py` to download the required nltk data.
3. Create a python file that looks similar to `links.py` to store the links to the books you want to analyze. We ran `ackroyd.py`, `links.py` and `styles.py` for our analysis. Run the file when you are done to create the proper tensorboard files.
4. For tensorboard visualization, edit `projector_files/projector_config.pbtxt` and create a new configuration. Look at that file for syntax.

### Visualization Setup

1. Run Tensorboard using `tensorboard --logdir=./projector_files/`
2. Go to localhost:6006 in your browser.
3. In the top-righthand corner select `Projector`

## Milestone 1 Assignment

- Use word embeddings analysis to analyze select crime novels (up to three per author) available on project Gutenberg; by author Agatha Christie
- Do word embeddings-based analysis -- produce work embeddings for your author's works and use those to do similar analysis like with plot or author analysis but by using embeddings metrics to assess the plot;
implement a custom, ultra-simplified prompt interface to run your analysis

Agatha Christie -- analyze the novels' plot using word embeddings  

## Docs

[Info Doc](https://docs.google.com/document/d/1SzlvAJDJ_J6TmqEKMKzm24yskDTdLdnPAg5V2sZGQQU/edit#heading=h.drgugs2suv61)

[LaTeX Code for Project 1](https://docs.google.com/document/d/1CAOgCz53kaDHoT7jyzDVTkqXNURDrYvL9NK5dGl2nt0/edit?tab=t.0)

[Slides for Project 1](https://docs.google.com/presentation/d/1ZUDemqwNvS-08xp7VWl1Ov0Y36tAkW7FoDArsWdbfEw/edit#slide=id.p)
