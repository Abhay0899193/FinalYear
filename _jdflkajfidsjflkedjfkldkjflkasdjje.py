
from nltk.corpus import wordnet

word = input("word:")

syns = wordnet.synsets(word)
            
syns1 = []
k=0
for n in syns:
    a = syns[k].lemmas()[0].name()
    print (a)
    syns1.extend(a)
    k=k+1
