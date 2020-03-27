
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Sequential
import numpy as np

pickle_in = open(" X.pickle","rb")
X=pickle.load(pickle_in)
pickle_in.close()

pickle_in1 = open("Y.pickle","rb")
Y=pickle.load(pickle_in1)
pickle_in.close()

X=X/255.0

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:],activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

Y=np.array(Y)
model.fit(X,Y,batch_size=32,epochs=3,validation_split=0.3)