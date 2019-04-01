#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd

data_path = "/Volumes/External/data_sink/FacebookRecruiting/train.csv"
data = pd.read_csv(data_path)

# In[2]:
data_clean = data.groupby('source_node').filter(lambda x: len(x) >= 10)

# In[3]:
import numpy as np
split_mask = np.random.rand(len(data)) < 0.8
train = data[split_mask]
test = data[~split_mask]

# In[4]:

train_clean = train.groupby('source_node').filter(lambda x: len(x) >= 10)

# In[5]:
a = train_clean['source_node'].unique()
b = train_clean['destination_node'].unique()
c = list(set(a.tolist() + b.tolist()))
c.sort()

ind, ids = zip(*enumerate(c))
old_id2new = dict(zip(ids, ind))

# In[6]:
train_ = train_clean.applymap(lambda x: old_id2new[x])

# In[7]:
test_ = test.applymap(lambda x: old_id2new.get(x,None)).dropna()
