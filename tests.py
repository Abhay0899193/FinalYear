import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet

stemmer = LancasterStemmer()

training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"I like you"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"bye"})


classes = list(set([a['class'] for a in training_data]))

print (classes)

class_words = {}
corpus_words = {}
#temp_words = {}
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []
    


for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore a some things
        if word not in ["?", "'s"]:
            
#             stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
# have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
        
            syns = wordnet.synsets(word)
            
            syns1 = []
            k=0
            for n in syns:
                a = syns[k].lemmas()[0].name()
                print (a)
                syns1.extend(a)
                k=k+1
            
            for w in syns1 :
                if w not in corpus_words :
                    stemmed_words = stemmer.stem(w.lower())
                    if stemmed_words not in corpus_words:
                        corpus_words[stemmed_words] = 1
                   
                    class_words[data['class']].extend([stemmed_words])
            class_words[data['class']].extend([stemmed_word])
#             add the word to our words in class list
            

print (corpus_words)
print (class_words)