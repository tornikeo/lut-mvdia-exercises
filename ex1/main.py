#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat


# ## Task 1 - Image registration

# In[2]:


a = Image.open('ex21a.tif')
b = Image.open('ex21b.tif')


# In[3]:


a


# In[4]:


b


# In[5]:


from skimage.transform import estimate_transform, AffineTransform, warp
import numpy as np

img1 = np.array(a)
img2 = np.array(b)[...,0]


# In[42]:


from skimage.transform import estimate_transform
from skimage.transform import AffineTransform, warp

points1 = np.array([
  [327, 392], # tail end of a
  [540, 340], # end pipe down
  [151, 618],  # largest square br corner
  [640, 530], # second to last low a tail
])

points2 = np.array([
  [337, 391],
  [542, 354],
  [204, 604],
  [683, 551]
])

tfm = estimate_transform('affine', points1, points2)
img2_gen = (warp(img1, tfm.inverse) * 255).astype('uint8')
plt.imshow(img2_gen != img2)


# ## Task 2

# In[72]:


data = loadmat('Data for Exercise 1.mat')['data']


# In[75]:


data.shape, data.mean(0), data.std(0)


# ### 1. Naive bayes

# In[120]:


# Adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.inspection import DecisionBoundaryDisplay

x_full = data[:,:2]
y_full = data[:,2]

x_full = StandardScaler().fit_transform(x_full)
n_classes = 2
plot_colors = "ryb"
target_names = ['salmon', 'sea bass']

x_train, x_test, y_train, y_test = \
    train_test_split(x_full, y_full, random_state=42, test_size=.3)

for clf in  [ GaussianNB(), KNeighborsClassifier(5), MLPClassifier(max_iter=200, learning_rate='adaptive'), SVC(C=.5) ]:
    clf.fit(x_train, y_train)

    # y_pred = clf.predict(x_test)
    # print(classification_report(y_pred=y_pred, y_true=y_test))

    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(2, 3, 1)

    ax.set_title(f"{clf.__class__.__name__}")
    DecisionBoundaryDisplay.from_estimator(
        clf,
        x_full,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel='brightness',
        ylabel='length',
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_full == i)
        plt.scatter(
            x_full[idx,0],
            x_full[idx,1],
            c=color,
            label=target_names,
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )


# ## C-Means clustering

# In[124]:


# Adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
from sklearn.cluster import KMeans

clust = KMeans(2)
y_pred = clust.fit_predict(x_full);
for i, color in zip(set(y_pred), plot_colors):
    
    idx = np.where(y_pred == i)
    plt.scatter(
        x_full[idx,0],
        x_full[idx,1],
        c=color,
        # label=target_names,
        cmap=plt.cm.RdYlBu,
        edgecolor="black",
        s=15,
    )
    plt.xlabel('brightness')
    plt.ylabel('length')


# ## Self-organizing map

# In[125]:


get_ipython().system('pip install minisom')


# In[147]:


# Adapted from https://github.com/JustGlowing/minisom/blob/master/examples/Clustering.ipynb
from minisom import MiniSom    
som_shape = 1,2
som = MiniSom(*som_shape, 2, 
              sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(x_train, 100) # trains the SOM with 100 iterations


# In[152]:


# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in x_train]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

for c in np.unique(cluster_index):
    plt.scatter(x_train[cluster_index == c, 0],
                x_train[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=10, linewidths=15, color='k', label='centroid')
plt.legend();

