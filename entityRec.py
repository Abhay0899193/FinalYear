# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet,stopwords

# word stemmer
stemmer = LancasterStemmer()

# we can now calculate a score for a new sentence
sentence = input("check:")
# 3 classes of training data
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

for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore a some things
        stop_words = set(stopwords.words("english"))
#        if word not in stop_words:
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

            if show_details:
               print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score



# now we can find the class with the highest score
for c in class_words.keys():
    print ("Class: %s  Score: %s \n" % (c, calculate_class_score(sentence, c)))
    
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