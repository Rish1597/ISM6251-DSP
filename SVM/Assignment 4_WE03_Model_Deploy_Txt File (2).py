#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle


# In[14]:


model_ownership = pickle.load(open('C:/Users/risha/Downloads/model_ownership.pkl', "rb"))


# In[15]:


print("\n*****************************************************")
print("* The USF Super Simple Ownership Prediction Model *")
print("*****************************************************\n")


# In[23]:


Income = float(input("Enter the Income: "))
Lot_Size= float(input("Enter the Lot Size: "))
df = pd.DataFrame({'Income': [Income],'Lot_Size' : [Lot_Size]})

result = model_ownership.predict(df)
probability = model_ownership.predict_proba(df)
ownership = ('owner', 'non-owner')
print(f"\nThe USF Super Simple Ownership Prediction model indicates probability of ownership at {probability[0][1]:.4f}, therefore it implies that the person is {result[0]}.\n")

