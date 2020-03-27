import tensorflow as td
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import time
import os
import numpy as np

NAME='Dog VS CAT_{}'.format(str(time.time()))
print(str(time.time()))

tensorboard=TensorBoard(log_dir='logs\{}'.format(NAME))

print(os.getcwd())
X_in=open('X.pickle','rb')
X=pickle.load(X_in)
X_in.close()


Y_in=open('Y.pickle','rb')
Y=pickle.load(Y_in)
Y_in.close()

X=X/255.0

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:],activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']
              )
Y=np.array(Y)
model.fit(X,Y,epochs=3,validation_split=0.2,callbacks=[tensorboard])
