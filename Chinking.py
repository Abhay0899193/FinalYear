import nltk
import wikipedia
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union






train_text = state_union.raw("2005-GWBush.txt")
_text = input("Enter a text:")
sample_text=wikipedia.summary(_text,sentences=1)
custom_SentTok = PunktSentenceTokenizer(train_text)
tokenized = custom_SentTok.tokenize(sample_text)

def process_content () :
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            
            chunkGram = r"""Chunk: {<.*>+}
                                   }<VB.?|IN|DT|TO>+{"""
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            chunked.draw()
            
    except Exception as e:
         print(str(e))
         
process_content() 