import nltk
from nltk.stem.lancaster import LancasterStemmer
from sklearn.externals import joblib

stemmer = LancasterStemmer()

training_data = []
training_data.append({"class":"greeting", "sentence":"hi"})
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"hello"})
training_data.append({"class":"greeting", "sentence":"hey"})
training_data.append({"class":"greeting", "sentence":"Good morning"})
training_data.append({"class":"greeting", "sentence":"Good afternoon"})
training_data.append({"class":"greeting", "sentence":"how do you do?"})
training_data.append({"class":"greeting", "sentence":"How are things?"})
training_data.append({"class":"greeting", "sentence":"How’s it goin?"})

training_data.append({"class":"alarm", "sentence":"set alarm clock for six tomorrow."})
training_data.append({"class":"alarm", "sentence":"set alarm for 6 AM"})
training_data.append({"class":"alarm", "sentence":"set alarm for 6 PM"})
training_data.append({"class":"alarm", "sentence":"please set alarm for tommarow evening"})
training_data.append({"class":"alarm", "sentence":"remind me tomarrow morning"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"bye"})
training_data.append({"class":"goodbye", "sentence":"catch you later"})
training_data.append({"class":"goodbye", "sentence":" it's time to head off."})
training_data.append({"class":"goodbye", "sentence":" it's time to leave you"})
training_data.append({"class":"goodbye", "sentence":"See you later"})
training_data.append({"class":"goodbye", "sentence":"Talk to you later!"})
training_data.append({"class":"goodbye", "sentence":"Have a good day"})


training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})

training_data.append({"class":"weather", "sentence":"What's it like outside? "})
training_data.append({"class":"weather", "sentence":"How's the weather?"})
training_data.append({"class":"weather", "sentence":"Do you have rain?"})
training_data.append({"class":"weather", "sentence":"What's the temperature in Manchester?"})
training_data.append({"class":"weather", "sentence":"It's snowing here in Manchester, what's it doing there? "})
training_data.append({"class":"weather", "sentence":"It is a Beautiful day for a walk?"})
training_data.append({"class":"weather", "sentence":"What's the weather forecast for the rest of the week?"})




corpus_words = {}
class_words = {}

classes = list(set([a['class'] for a in training_data]))
for c in classes:
   
    class_words[c] = []


for data in training_data:
   
    for word in nltk.word_tokenize(data['sentence']):
       
        if word not in ["?", "'s","."]:
           
            stemmed_word = stemmer.stem(word.lower())

            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            
            class_words[data['class']].extend([stemmed_word])



filename = 'finalized_model.sav'

joblib.dump(corpus_words, filename)

filename1 = 'finalized_model1.sav'

joblib.dump(class_words, filename1)