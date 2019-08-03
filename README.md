# NLP Project BGU - Named Entity Recognition 

This project was made by Yonatan Bitton, Assaf Peleg, and Benjamin Berend. Students at Natural Language Processing Course taught by Prof. Michael Elhadad.
Our mission was NER - Named Entity Recognition. 

NER defenition from Wikipedia: 

```sh
Named-entity recognition (NER) (also known as entity identification, entity chunking and entity extraction) is a subtask of information extraction that seeks to locate and classify named entity mentions in unstructured text into pre-defined categories such as the person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.
Most research on NER systems has been structured as taking an unannotated block of text, such as this one:

Jim bought 300 shares of Acme Corp. in 2006.

And producing an annotated block of text that highlights the names of entities:

[Jim]Person bought 300 shares of [Acme Corp.]Organization in [2006]Time.
```
We present to you four IPython notebooks: 

1. Phase1_DataPreperationAndExploration - Making the **dataset** (Tokenization, obtaining POS & Morphological attributes), exploring the dataset's attributes, etc.
2. Phase2_ML - **Machine learning** method solving our task with performance - **f1-score of 82%** without 'O' tag and 96% with the 'O' tag.
3. Phase3_DL - **Deep learning** method solving our task with performance - **f1-score of 90%** without 'O' tag and 98% with the 'O' tag.
4. Phase4_ErrorAnalysis - Understand the main causes for **errors** and what is possible to do in order to overcome them. 

### Prerequisites

Make virtual environment for the project. 
Necesarry packages:
```sh
pip install sklearn
pip install pandas
pip install nltk
```
run with ```jupyter notebook``` or ```jupyter lab```.

## Authors

Yonatan Bitton  
Assaf Peleg  
Benjamin Berend  

## Acknowledgments

* Dataset - https://www.cs.bgu.ac.il/~elhadad/nlpproj/naama/
* POS & Morphological attributes - https://www.cs.bgu.ac.il/~elhadad/nlpproj/LDAforHebrew.html
* All project was done under the guidence of Prof. Michael Elhadad, Ben Gurion University, Israel. 
