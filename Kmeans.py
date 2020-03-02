#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Generate data in a 2 dimentional space

# In[27]:


X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, edgecolor ='b')


# ## Use Scikit_learn

# In[28]:


from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)


# ## Finding the centroid

# In[29]:


Kmean.cluster_centers_


# In[33]:


plt.scatter(X[ : , 0], X[ : , 1], s =50)
plt.scatter(-0.99252583, -0.95787172, s=200)
plt.scatter(1.91530391, 2.09680822, s=200)


# ## Testing the algorithm

# In[34]:


Kmean.labels_


# ####  As you can see, 50 data points belong to the 0 cluster while the rest belong to the 1 cluster

# In[41]:


sample_test=np.array([0.0, -1.5])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)


# #### Shows that the test data point [0.0, -1.5] belongs to the 0 (orange centroid) cluster

# In[42]:


sample_test=np.array([3.5, 1.7])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)


# #### Shows that the test data point [3.5, 1.7] belongs to the 1 (green centroid) cluster

# ## Another example using 4 clusters

# In[44]:


from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=600, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[45]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[48]:


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.4);


# In[ ]:




