import numpy as np
import pandas as pd

CODE_LOC = 'C://Users/Abhay/Downloads/NLPBot-master/NLPBot-master/'   
DATA_LOC = 'C://Users/Abhay/Downloads/NLPBot-master/NLPBot-master/analysis/sentences.csv' 

sentences = pd.read_csv(filepath_or_buffer = DATA_LOC)

## Extract some patterns of PoS sequences
#import nltk
#from nltk import word_tokenize
#
#list_of_triple_strings = []  # triple sequence of PoS tags
#sentence = "Can a dog see in colour?"
#
#sentenceParsed = word_tokenize(sentence)
#pos_tags = nltk.pos_tag(sentenceParsed)
#pos = [ i[1] for i in pos_tags ]
##print("Words mapped to Part of Speech Tags:",pos_tags)
##print("PoS Tags:", pos)
#
#n = len(pos)
#for i in range(0,n-3):
#    t = "-".join(pos[i:i+3]) # pull out 3 list item from counter, convert to string
#    list_of_triple_strings.append(t)
    
#print("sequences of triples:", list_of_triple_strings)
#
import sys
sys.path.append(CODE_LOC)  
import features  
#sentence = "Can a dog see in colour?"
#
#sentence = features.strip_sentence(sentence)
#print(sentence)
#pos = features.get_pos(sentence)
#triples = features.get_triples(pos)
#print(triples)


sentences = ["Can a dog see in colour?",
             "Hey, How's it going?",
             "Oracle 12.2 will be released for on-premises users on 15 March 2017",
             "When will Oracle 12 be released"]
id = 1
for s in sentences:
    features_dict = features.features_dict(str(id),s)
    features_string,header = features.get_string(str(id),s)
#    print(features_dict)
#    print(features_string)
    id += 1
    

from sklearn.ensemble import RandomForestClassifier

FNAME = 'C://Users/Abhay/Downloads/NLPBot-master/NLPBot-master/analysis/featuresDump.csv' 

df = pd.read_csv(filepath_or_buffer = FNAME, )   
#print(str(len(df)), "rows loaded")


df.columns = df.columns[:].str.strip()
df['class'] = df['class'].map(lambda x: x.strip())

width = df.shape[1]
    
#split into test and training (is_train: True / False col)
np.random.seed(seed=1)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]
#print(str(len(train)), " rows split into training set,", str(len(test)), "split into test set.")

features = df.columns[1:width-1]  #remove the first ID col and last col=classifier
#print("FEATURES = {}".format(features))

# Fit an RF Model for "class" given features
clf = RandomForestClassifier(n_jobs=2, n_estimators = 100)
clf.fit(train[features], train['class'])


# Predict against test set
preds = clf.predict(test[features])
predout = pd.DataFrame({ 'id' : test['id'], 'predicted' : preds, 'actual' : test['class'] })

#print(predout)

# Cross-check accuracy ##
#print(pd.crosstab(test['class'], preds, rownames=['actual'], colnames=['preds']))
#print("\n",pd.crosstab(test['class'], preds, rownames=['actual']
#                       , colnames=['preds']).apply(lambda r: round(r/r.sum()*100,2), axis=1) )

from sklearn.metrics import accuracy_score
print("\n\nAccuracy Score: ", round(accuracy_score(test['class'], preds),3) ) # https://en.wikipedia.org/wiki/Jaccard_index


FNAME = 'C://Users/Abhay/Downloads/NLPBot-master/NLPBot-master/analysis/pythonFAQ.csv' # !! Modify this to the CSV data location

import csv
import hashlib 



fin = open(FNAME, 'rt')
reader = csv.reader(fin)

keys = ["id",
"wordCount",
"stemmedCount",
"stemmedEndNN",
"CD",
"NN",
"NNP",
"NNPS",
"NNS",
"PRP",
"VBG",
"VBZ",
"startTuple0",
"endTuple0",
"endTuple1",
"endTuple2",
"verbBeforeNoun",
"qMark",
"qVerbCombo",
"qTripleScore",
"sTripleScore",
"class"]

rows = []


textout = {'Q': "QUESTION", 'C': "CHAT", 'S':"STATEMENT"}

mySentence = "Scikit-learn is a popular Python library for Machine Learning."
#mySentence = "The cat is dead"
#mySentence = "Is the cat dead"

myFeatures = features.features_dict(str(id),mySentence)

values=[]
for key in keys:
    values.append(myFeatures[key])

s = pd.Series(values)
width = len(s)
myFeatures = s[1:width-1]  #All but the last item (this is the class for supervised learning mode)
predict = clf.predict([myFeatures])

print("\n\nPrediction is: ", textout[predict[0].strip()])
next(reader)  #Assume we have a header 
for line in reader:
    sentence = line[0]  
    c = line[1]        #class-label
    id = hashlib.md5(str(sentence).encode('utf-8')).hexdigest()[:16] # generate a unique ID
    
    f = features.features_dict(id,sentence, c)
    row = []
    
    for key in keys:
        value = f[key]
        row.append(value)
    rows.append(row)
    
faq = pd.DataFrame(rows, columns=keys)
fin.close()

# Predict against FAQ test set
featureNames = faq.columns[1:width-1]  #remove the first ID col and last col=classifier
faqPreds = clf.predict(faq[featureNames])

predout = pd.DataFrame({ 'id' : faq['id'], 'predicted' : faqPreds, 'actual' : faq['class'] })

## Cross-check accuracy ##
print(pd.crosstab(faq['class'], faqPreds, rownames=['actual'], colnames=['preds']))

print("\n",pd.crosstab(faq['class'], faqPreds, rownames=['actual'],
                       colnames=['preds']).apply(lambda r: round(r/r.sum()*100,2), axis=1) )

print("Accuracy Score:", round(accuracy_score(faq['class'], faqPreds) ,3) )

