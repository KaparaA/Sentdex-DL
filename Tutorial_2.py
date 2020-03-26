import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle
import os

DIR="F:\ML\Projects\Cat vs Dog\PetImages"

CATEGORIES=['Dog','Cat']

IMG_SIZE=60

training_data = []

def create_training_data():

    for category in CATEGORIES:
        path=os.path.join(DIR,category)
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
                try:
                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass

create_training_data()
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0])
print(np.array(X[0]).reshape((-1,IMG_SIZE,IMG_SIZE,1)))

X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

pickle_out = open('X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out1 = open('Y.pickle','wb')
pickle.dump(y , pickle_out1)
pickle_out1.close()

print(os.getcwd())

pickle_in = open("X.pickle", 'rb')
X=pickle.load(pickle_in)

pickle_in1 = open("Y.pickle", 'rb')
Y=pickle.load(pickle_in1)