#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


data=pd.read_csv("C:/Users/DELL/Desktop/msvn/retail.csv")


# In[20]:


data.head(5)


# In[21]:


print (data.columns)
data.describe().T



# In[22]:


#remove null values 
data.dropna(axis=0, subset=["CustomerID"], inplace=True)
data.isnull().sum()



# In[23]:


print("number of duplicated entriees :{}".format(data.duplicated().sum()))
data.drop_duplicates(inplace=True)



# In[24]:


nbTran= len(data.InvoiceNo.unique())
nbProd = len(data.StockCode.unique())
nbCustom= len(data.CustomerID.unique())
plt.bar(["transactions", "products", "customers"],[nbTran,nbProd , nbCustom], color=["blue", "green", "red"])
plt.title('total number of transactions, products, customers')
plt.ylabel('Total number')


# In[25]:


vtrans=len(data.InvoiceNo.unique())
ctrans= len([i for i in data.InvoiceNo.unique() if str(i).startswith('C')])
plt.pie([vtrans, ctrans], labels=["valided transactions", "canceled transactions"], autopct="%1.2f%%")
plt.legend()
plt.title("total number of valid and canceled transactions")


# In[26]:


#remove canceled transactions
data.drop(data[data["InvoiceNo"].apply(lambda x:str(x).startswith('C'))].index, axis=0, inplace=True)


# In[28]:


# create Price feature
data['Totalutility'] = data['UnitProfit']*data['Quantity']
data[:15]



# In[30]:


# average price per transactions and customer 
totalutility= data.groupby(['StockCode', 'Quantity'])[['Totalutility']].sum()
df1 =totalutility.groupby(['StockCode']).sum()
scaler = StandardScaler()
totalutility_scaled = scaler.fit_transform(df1)
df1 =pd.DataFrame(data= totalutility_scaled, index=df1.index, columns=df1.columns)


# In[32]:


df1['Freq']= data.groupby(['StockCode', 'Quantity'])['Quantity'].count().groupby(['StockCode']).count()
df1.head()


# In[33]:


data.loc[:10,['StockCode', 'Description', 'Totalutility']]


# In[46]:


# find the optimal number of clusters 
wcss= []
#X= df1[["TotalPrice" , "Duration", "freq" ]]
X= data[["CustomerID" , "Totalutility"]]
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# In[45]:


##Fitting kmeans to the dataset with k=4
km4=KMeans(n_clusters=4,init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km4.fit_predict(X)
print(y_means)


# In[49]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=50, c='purple',label='Cluster1')
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=50, c='blue',label='Cluster2')
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=50, c='green',label='Cluster3')
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=50, c='cyan',label='Cluster4')
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=50, c='pink',label='Cluster5')

plt.scatter(km4.cluster_centers_[:,0], km4.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('High Utility Itemsets')
plt.xlabel('totalutility:')
plt.ylabel('totalsales')
plt.legend()
plt.show()


# In[ ]:




