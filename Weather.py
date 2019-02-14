# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import pyowm
import sys
path = 'E://Python/imgCls/tensorflow-for-poets-2-master/tensorflow-for-poets-2/scripts/'   
sys.path.append(path)
import faceDetector as fd
#from sklearn.externals import joblib
# word stemmer
my = pyowm.OWM('f1147045a3b5c94f7bc496789168c3f6') 
stemmer = LancasterStemmer()

# we can now calculate a score for a new sentence
#sentence = "Weather of Ghaziabad."
# 3 classes of training data
training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"I like you"})
training_data.append({"class":"greeting", "sentence":"Hi"})
training_data.append({"class":"greeting", "sentence":"hello"})
training_data.append({"class":"greeting", "sentence":"what's up"})



training_data.append({"class":"alarm", "sentence":"set alarm clock for six tomorrow."})
training_data.append({"class":"alarm", "sentence":"set alarm for 6 AM"})
training_data.append({"class":"alarm", "sentence":"set alarm for 6 PM"})
training_data.append({"class":"alarm", "sentence":"please set alarm for tommarow evening"})
training_data.append({"class":"alarm", "sentence":"remind me tomarrow morning"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"goodbye"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"bye"})
training_data.append({"class":"goodbye", "sentence":"no thanks"})
training_data.append({"class":"goodbye", "sentence":"good night"})
training_data.append({"class":"goodbye", "sentence":"take care"})



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
training_data.append({"class":"weather", "sentence":"is there cold or hot?"})

#training_data.append({"class":"weather", "sentence":"Do you have rain?"})
#print ("%s sentences of training data" % len(training_data))

# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}
# turn a list into a set (of unique items) and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []

# loop through each sentence in our training data
for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore a some things
        if word not in ["?", "'s","."]:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
# have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])



#filename = 'finalized_model.sav'

#joblib.dump(corpus_words, filename)

#filename1 = 'finalized_model1.sav'

#joblib.dump(class_words, filename1)


# we now have each stemmed word and the number of occurances of the word in our training corpus (the word's commonality)
#print ("Corpus words and counts: %s \n" % corpus_words)
# also we have all words in each class
#print ("Class words: %s" % class_words)

# calculate a score for a given class taking into account word commonality
def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # treat each word with relative weight
            score += (1 / corpus_words[stemmer.stem(word.lower())])

#            if show_details:
#               print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score



# now we can find the class with the highest score
#for c in class_words.keys():
#    print ("Class: %s  Score: %s \n" % (c, calculate_class_score(sentence, c)))
    
    # return the class with highest score for sentence
def classify(sentence):
    high_class = None
    high_score = 0
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score(sentence, c, show_details=False)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score

    return high_class, high_score

#print(classify(sentence))

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            print(current_chunk)   
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
        if continuous_chunk == []:
            continuous_chunk = current_chunk
    return continuous_chunk

#my_sent = input("Enter a sentence:")
#print(get_continuous_chunks(my_sent))

def get_weather(sentence):
    closer = ''
    if classify(sentence)[0] == 'weather':
        ne = get_continuous_chunks(sentence)
        if ne != []:
            observation = my.weather_at_place(ne[0])
            w = observation.get_weather()
            k = w.get_temperature('celsius')
            print('Temperature: '+str(k['temp'])) 
        else:
            a = input("Enter a location:")
            observation = my.weather_at_place(a)
            w = observation.get_weather()
            k = w.get_temperature('celsius')
            print('Temperature: '+str(k['temp']))  
    elif classify(sentence)[0] == 'greeting':
        x,y,z = fd.main1()
        x = x[0].upper()+x[1:]
#        print(x,y)
        if y>0.2 and z == True:
            x = "Abhay"
            print("Hello {} how are you?".format(x))
        else:
            print('hello, How are you')

    elif classify(sentence)[0] == 'goodbye':
        print('Bye, have a nice day')
        closer = 'goodbye'
    if closer != 'goodbye':
        print("\nDo you want anything else?\n")
    return closer

closer = ''
print('Hello how can I help you?')
while (closer != 'goodbye'):
    sentence = input("")
    closer = get_weather(sentence)

#what is the weather of Gorakhpur.
