#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.utils import shuffle
import tensorflow
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


# In[2]:


#Just for monitoring
import wandb
from wandb.keras import WandbCallback

# In[3]:


#C:\Users\ing_l\Final SO\Breast Cancer


# In[4]:


#The data is stores as ints


# In[5]:


data = np.load('X.npy')


# In[6]:


targets = np.load('Y.npy')


# In[7]:


# In[8]:


data, targets = shuffle(data, targets)


# In[9]:


# plot some images  



# In[10]:



# In[11]:


#Normalize data and categorize targets
data = data / 255.0


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, stratify=targets)


# In[13]:


#CNN model
i1 = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
d1 = Conv2D(150, (3, 3), activation='relu')(i1)
d1 = BatchNormalization()(d1)
d1 = MaxPooling2D(pool_size=(2,2))(d1)
d1 = Conv2D(300, (5, 5), activation='relu')(d1)
d1 = BatchNormalization()(d1)
d1 = MaxPooling2D(pool_size=(2,2))(d1)
d1 = Dropout(0.5)(d1)
d1 = Conv2D(200, (3, 3), activation='relu')(d1)
d1 = MaxPooling2D(pool_size=(2,2))(d1)
d1 = Flatten()(d1)
d1 = Dropout(0.5)(d1)
d1 = Dense(1, activation='sigmoid')(d1)

#Dense model
i2 = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
d2 = Flatten()(i2)
d2 = Dense(4096, activation='relu')(d2)
d2 = Dense(8192, activation='relu')(d2)
d2 = Dense(2048, activation='relu')(d2)
d2 = Dropout(0.3)(d2)
d2 = Dense(4096, activation='relu')(d2)
d2 = Dropout(0.5)(d2)
d2 = Dense(1024, activation='relu')(d2)
d2 = Dense(1, activation='sigmoid')(d2)


# In[14]:


#For the weigths and biases graphs
def model_fitter(batch_size, optimizer, model, epochs=100):
    
    wandb.init(project="cnn-graph-so-tensorflow-gpu", name='model {}, batch {}, optimizer{}'.format(model[2], batch_size, optimizer), reinit=True)

    model = Model(inputs=model[0], outputs=model[1])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

    h = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[WandbCallback()])


# In[15]:


model=[[i1,d1,'CNN']]
batch_size=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
optimizer=['SGD','Adam']#, 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
for m in model:
    for batch in batch_size:
        for op in optimizer:
            model_fitter(batch, op, m)


# In[ ]:




