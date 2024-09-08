import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data3.pickle', 'rb')) #our data is in a dictionary

# Determine the desired length of the data (length of 84 breaks it)
desired_length = 42

# Filter the data and labels
filtered_data = []
filtered_labels = []

for i, item in enumerate(data_dict['data']):
    if len(item) == desired_length:
        filtered_data.append(item)
        filtered_labels.append(data_dict['labels'][i])

# Convert to numpy arrays
data = np.array(filtered_data)
labels = np.array(filtered_labels)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) #stratify makes sure that same proportions of each letter is in the test and the train set

model = RandomForestClassifier() #the premade model we are using

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print("{}% of samples were classified correctly".format(score*100))


f = open('model3.p', 'wb') #write and binary

pickle.dump({'model' : model}, f) # a dictionary

f.close()