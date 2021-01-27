import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def preProcessing(text):
    import re
    import string
    # Convert text to lowercase
    outText = text.lower()
    # Remove numbers
    outText = re.sub(r'\d+', '', outText)
    # Remove punctuation

    outText = outText.translate(str.maketrans("", "", string.punctuation))
    # Remove whitespaces
    outText = outText.strip()
    # Remove stopwords
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(outText)
    outText = [i for i in tokens if not i in stop_words]

    # Lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    result = []
    for word in outText:
        result.append(lemmatizer.lemmatize(word))

    return result


from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
categories = data.target_names
# Training the data on these categories\n",
train = fetch_20newsgroups(subset='train', categories=categories)
# Testing the data for these categories\n",
test = fetch_20newsgroups(subset='test', categories=categories)
# print(train.target_names)

import time

start_time = time.time()

word_dict = {}
train_words = []
test_words = []
count = 0
#print(train.data[10000])
for item in train.data:
    k = preProcessing(item)

    train_words.append(k)
    for item1 in k:
        word_dict[item1] = count

for item in test.data:
    k = preProcessing(item)
    test_words.append(k)
    for item1 in k:
        word_dict[item1] = count

train_occurence = {}
index = 0
for item in train_words:

    train_occurence[index] = {}
    for item1 in item:
        train_occurence[index][item1] = 0

    for item1 in item:
        train_occurence[index][item1] += 1

    index = index + 1

class_occurence = [0] * 20

for item in train.target:
    class_occurence[item] += 1

priors = []
for i in class_occurence:
    priors.append(i/len(train.data))


print(priors)

totalX = 20*[0]
j = 0
for item in train.target:
    totalX[item] += sum(list(train_occurence[j].values()))

    j = j + 1

indice = 0
classes_and_totals = {} # per word, there will be different classes and each class in a word=>key has occurence value.

for item in train.target:
    for word in train_occurence[indice].keys():
        if word not in classes_and_totals.keys():
            classes_and_totals[word] = {}
        if item not in classes_and_totals[word].keys():
            classes_and_totals[word][item] = 0

    for word in train_occurence[indice].keys():
        classes_and_totals[word][item] += train_occurence[indice][word]
    indice = indice + 1


for item in classes_and_totals.keys(): # bir kelime bir classta hic olmamaissa bile, o classi nestedDict'de index olarak tanimlayip occurence i 0 yapiyoruz
    for item1 in range(20):
        if item1 not in classes_and_totals[item].keys():
            classes_and_totals[item][item1] = 0



for item in classes_and_totals.keys():
    for item1 in classes_and_totals[item].keys():
        classes_and_totals[item][item1] = (classes_and_totals[item][item1]+1)/(totalX[item1]+len(word_dict))*15000

test_probs = {}
p = 0 # doc id in test documents

for item in test_words:
    test_probs[p] = np.array([])
    for c in range(20): # choose a class
        mul = priors[c]
        for item1 in item: # choose a word from doc
            if item1 in classes_and_totals.keys():
                mul = mul*classes_and_totals[item1][c]
            else: # class_and_totals depends on train datas, here we examine test datas. So the word may not be seen in this dict.
                mul = mul*1/(totalX[c]+len(word_dict))*15000
        test_probs[p] = np.append(test_probs[p],[mul])

    p = p + 1

p = 0 #doc id
hit = 0 #dogru tahmin sayisi
#print(test.target[0])

conf_matrix = []

for i in range(20): # initialize confusion matrix
    temp = []
    for j in range(20):
        temp.append(0)
    conf_matrix.append(temp)

for item in test_probs.keys():
    maxIndex = np.argmax(test_probs[item])
    conf_matrix[test.target[p]][maxIndex] += 1
    if maxIndex == test.target[p]:
        hit = hit + 1
    p = p + 1



print('hit = %d' % hit)
accuracy = hit/len(test.data)
print('accuracy is %f ' % accuracy)
print('dictionary size = %d' % len(word_dict))
elapsed_time = time.time() - start_time
print(elapsed_time)

df_cm = pd.DataFrame(conf_matrix, index=[i for i in test.target_names],
                     columns=[i for i in test.target_names])
plt.figure(figsize = (10,7))
plt.title('confusion matrix')
sns.heatmap(df_cm, annot=True)
plt.show()