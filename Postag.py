
# coding: utf-8

# BIO tags

# In[1]:

import nltk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tree import Tree

style.use('fivethirtyeight')

# Process text 
def process_text(txt_file):
    raw_text = open(txt_file).read()
    token_text = word_tokenize(raw_text)
    return token_text

# Stanford NER tagger
def stanford_tagger(token_text):
    # Change the path according to your system
    stanford_classifier = 'C:\\Users\\dell\\Desktop\\chunk\\stanford\\stanford\stanford-ner-2017-06-09\\classifiers\\english.all.3class.distsim.crf.ser.gz'
    stanford_ner_path = 'C:\\Users\\dell\\Desktop\\chunk\\stanford\\stanford\\stanford-ner-2017-06-09\\stanford-ner.jar'
    # Creating Tagger Object
    st = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')
    ne_tagged = st.tag(token_text)
    return(ne_tagged)

# NLTK POS and NER taggers
def nltk_tagger(token_text):
    tagged_words = nltk.pos_tag(token_text)
    ne_tagged = nltk.ne_chunk(tagged_words)
    return(ne_tagged)

# Tag tokens with standard NLP BIO tags
def bio_tagger(ne_tagged):
    bio_tagged = []
    prev_tag = "O"
    for token, tag in ne_tagged:
        if tag == "O": #O
            bio_tagged.append((token,tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Ajacent NE
            bio_tagged.append((token, "B-"+tag))
            prev_tag = tag
    return bio_tagged

# Create tree
def stanford_tree(bio_tagged):
    tokens, ne_tags = zip(*bio_tagged)
    pos_tags = [pos for token, pos in pos_tag(tokens)]
    
    conlltags = [(token, pos, ne) for token, pos, ne in zip(tokens, pos_tags, ne_tags)]
    ne_tree = conlltags2tree(conlltags)
    return ne_tree

# Parse named entities from tree
def structure_ne(ne_tree):
    ne = []
    for subtree in ne_tree:
        if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token,pos in subtree.leaves()])
            ne.append((ne_string, ne_label))
    return ne


# output for bio tags

# In[2]:

# Group all additional functions together

txt_file = r"E:\company\2companies\ADMA\allWords.txt"

def stanford_main():
    print(structure_ne(stanford_tree(bio_tagger(stanford_tagger(process_text(txt_file))))))
    
# Call the functions
if __name__ == '__main__':

    stanford_main()
    nltk_main()


# output for nltk tags

# In[3]:

# Group all additional functions together

txt_file = r"E:\company\2companies\ADMA\allWords.txt"

    
def nltk_main():
    print(structure_ne(nltk_tagger(process_text(txt_file))))
    
# Call the functions
if __name__ == '__main__':
    #stanford_main()
    nltk_main()


# Entity-BIO_tags table

# In[4]:
"""
import pandas as pd
b = structure_ne(stanford_tree(bio_tagger(stanford_tagger(process_text(txt_file)))))
dfBIO = pd.DataFrame({'Entity':[b[i][0] for i in range(len(b))],
                      'BIO tags':[b[i][1] for i in range(len(b))]})

dfBIO = dfBIO.reindex(columns=pd.Index(['Entity','BIO tags']))
dfBIO


# In[5]:

dfBIO = dfBIO.drop_duplicates()
dfBIO


# In[6]:

dfBIO.to_csv('BIO_tags.csv')


# Entity-NLTK_tags table

# In[7]:

import pandas as pd
s = structure_ne(nltk_tagger(process_text(txt_file)))
dfnltk = pd.DataFrame({'Entity':[s[i][0] for i in range(len(s))],
                       'NLTK tags':[s[i][1] for i in range(len(s))]})
dfnltk


# In[8]:

dfnltk = dfnltk.drop_duplicates()
dfnltk


# In[9]:

dfnltk.to_csv('NLTK_tags.csv')
"""
