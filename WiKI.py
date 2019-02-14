import wikipedia
import nltk
_text=input("What do you want to search: ")

for word,pos in nltk.pos_tag(nltk.word_tokenize(str(_text))):
         if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
             cxz=word

#for word,ne in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(str(_text)))):
#         if (ne == 'NE'):
#             cxz=word
print(cxz)
abh = wikipedia.summary(cxz)
print(abh)