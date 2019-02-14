import nltk
#import wikipedia
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union






train_text = state_union.raw("2005-GWBush.txt")
_text = input("Enter a text:\n")
#sample_text=wikipedia.summary(_text)
custom_SentTok = PunktSentenceTokenizer(train_text)
tokenized = custom_SentTok.tokenize(_text)

namedEnt = []


def process_content () :
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()
#        for k in  namedEnt:
#            if (namedEnt.label()=="NE"):
#                print(namedEnt.leaves())
    except Exception as e:
         print(str(e))

def traverse(tree41):
    try:
        tree41.label()
    except AttributeError:
        print(tree41)
    else:
        # Now we know that t.node is defined
        j = tree41.label()
        if(j == "NE"):
            print(tree41.leaves())
        #print('(', tree41.label)
        for child in tree41:
            traverse(child)
            
process_content() 
traverse(namedEnt)
#My name is Abhay.