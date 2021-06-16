# UMUTeam at EXIST 2021
## Sexist Language Identification based on Linguistic Features and Transformers in Spanish and English
Sexism is harmful behaviour that can make women feel worthless promoting self-censorship and gender inequality. In the digital era, misogynists have found in social networks a place in which they can spread their oppressive discourse towards women. Although this particular form of oppressive speech is banned and punished on most social networks, its identification is quite challenging due to the large number of messages posted everyday. Moreover, sexist comments can be unnoticed as condescends or friendly statements which hinders its identification even for humans. With the aim of improving automatic sexist identification on social networks, we participate in EXIST-2021. This shared task involves the identification and categorisation of sexism language on Spanish and English documents compiled from micro-blogging platforms. Specifically, two tasks were proposed, one concerning a binary classification of sexism utterances and another regarding multi-class identification of sexist traits. Our proposal for solving both tasks is grounded on the combination of linguistic features and state-of-the-art transformers by means of ensembles and multi-input neural networks. To address the multi-language problem, we tackle the problem independently by language to put the results together at the end. Our best result was achieved in task 1 with an accuracy of 75.14\% and an accuracy of 61.70\% for task 2.


## Details
The source code is stored in the ```code``` folder. For training, the ```embeddings```folders there are symbolyc links to the pretrained word embeddings used. Due to size, however, you should download the ```glove.6b.300d.txt``` (https://nlp.stanford.edu/projects/glove/). The dataset is not submitted and you should download from codalab. If you need the trained model and feature sets you can request them by email <joseantonio.garcia8@um.es>.


## Install
1. Create a virtual environment in Python 3
2. Install the dependencies that are stored at requirements.txt
3. Copy the datasets at ```assets/exist/2021-es/dataset``` folder and ```assets/exist/2021-en/dataset```
4. Generate the dataset: 
    ```python -W ignore compile.py --dataset=exist --corpus=2021-es```
    ```python -W ignore compile.py --dataset=exist --corpus=2021-en```
    
5. Finetune BERT. 
    ```python -W ignore train.py --dataset=exist --corpus=2021-es --model=transformers```
    ```python -W ignore train.py --dataset=exist --corpus=2021-en --model=transformers```
    
6. Feature selection. 
    ```python -W ignore feature-selection.py --dataset=exist --corpus=2021-es```
    ```python -W ignore feature-selection.py --dataset=exist --corpus=2021-en```
    
7. Generate BF features: 
    ```python -W ignore generate-bf.py --dataset=exist --corpus=2021-es```
    ```python -W ignore generate-bf.py --dataset=exist --corpus=2021-en```
    
8. Feature selection for the BF features. 
    ```python -W ignore feature-selection.py --dataset=exist --corpus=2021-es```
    ```python -W ignore feature-selection.py --dataset=exist --corpus=2021-en```
    
9. Train (available features: lf, se, bf, we and combinations. For example: lf-bf)
    ```python -W ignore train.py --dataset=exist --corpus=2021-es --model=deep-learning --features=lf```
    ```python -W ignore train.py --dataset=exist --corpus=2021-en --model=deep-learning --features=lf```
    
10. Evaluate. 
    ```python -W ignore evaluate.py --dataset=exist --corpus=2021-es --model=deep-learning --features=lf --source=val```
