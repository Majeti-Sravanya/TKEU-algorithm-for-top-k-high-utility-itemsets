#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np, pandas as pd, re, scipy as sp, scipy.stats


# In[6]:


#Importing Dataset
pd.options.mode.chained_assignment = None
#dataset = 'C:/Users/DELL/Desktop/msvn/retail.csv'
Data = pd.read_csv("C:/Users/DELL/Desktop/msvn/retail.csv")
Data.head()


# In[12]:


#Listing Some Irrelevant StockCodes
Irrelevant = Data['StockCode'].unique()
Irrelevant.sort()
print('Irrelevant Transactions: \n',Irrelevant[::-1][:4])
#Quantity and UnitPrice Summary
Data.describe().iloc[:,:2]


# In[13]:


#Outliers and Irrelevant Values
#Dropping all stockcodes that contain only strings
CodeTypes = list(map(lambda codes: any(char.isdigit() for char in codes), Data['StockCode']))
IrrelevantCodes = [i for i,v in enumerate(CodeTypes) if v == False]
Data.drop(IrrelevantCodes , inplace = True)
#Removing Outliers Based on Z-score
Data = Data[(np.abs(sp.stats.zscore(Data['UnitPrice']))<3) & (np.abs(sp.stats.zscore(Data['Quantity']))<5)]


# In[14]:


# Missing & Incorrect Values
Data.drop(Data[(Data.Quantity>0) & (Data.InvoiceNo.str.contains('C') == True)].index, inplace = True)
Data.drop(Data[(Data.Quantity<0) & (Data.InvoiceNo.str.contains('C') == False)].index, inplace = True)
Data.drop(Data[Data.Description.str.contains('?',regex=False) == True].index, inplace = True)
Data.drop(Data[Data.UnitPrice == 0].index, inplace = True)

for index,value in Data.StockCode[Data.Description.isna()==True].items():
    if pd.notna(Data.Description[Data.StockCode == value]).sum() != 0:
        Data.Description[index] = Data.Description[Data.StockCode == value].mode()[0]
    else:
        Data.drop(index = index, inplace = True)
        
Data['Description'] = Data['Description'].astype(str)


# In[15]:


#Incorrect Prices
StockList = Data.StockCode.unique()
CalculatedMode = map(lambda x: Data.UnitPrice[Data.StockCode == x].mode()[0],StockList)
StockModes = list(CalculatedMode)
for i,v in enumerate(StockList):
    Data.loc[Data['StockCode']== v, 'UnitPrice'] = StockModes[i]


# In[16]:


#Customers with Different Countries
Customers = Data.groupby('CustomerID')['Country'].unique()
Customers.loc[Customers.apply(lambda x:len(x)>1)]


# In[17]:


#Fixing Duplicate CustomerIDs
for i,v in Data.groupby('CustomerID')['Country'].unique().items():
    if len(v)>1:
        Data.Country[Data['CustomerID'] == i] = Data.Country[Data['CustomerID'] == i].mode()[0]

#Adding Desired Features
Data['FinalPrice'] = Data['Quantity']*Data['UnitPrice']
Data['InvoiceMonth'] = Data['InvoiceDate'].apply(lambda x: x.strftime('%B'))
Data['Day of week'] = Data['InvoiceDate'].dt.day_name()

#Exporting Processed Data
Data.to_csv("Data.to_csv("", date_format = '%Y-%m-%d %H:%M', index = False)
", date_format = '%Y-%m-%d %H:%M', index = False)


# In[32]:


pip install apyori


# In[36]:


import pandas as pd, numpy as np
from apyori import apriori, association_rules



# In[40]:


import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


Data = Data_Cleaned.copy()
Data['Country'] = Data['Country'].map(lambda x: 'UK' if x=='UNITED KINGDOM' else 'non-UK')
CustomerData = Data.groupby(['CustomerID','Country'], sort=False).agg({'Quantity':'mean','UnitPrice':'mean','InvoiceNo':'nunique','Description':'nunique'})
CustomerData.reset_index(inplace=True)
CustomerData.columns = ['CustomerID', 'UK?', 'Average Quantity', 'Average Price', 'Repeats', 'Product Variety']
CustomerData.head()


# In[41]:


#scaling the numerical features for clustering
Scaler = StandardScaler()
CustomerData.iloc[:,2:] = Scaler.fit_transform(CustomerData.iloc[:,2:])
syms = CustomerData.iloc[:,0].values.astype(str)
X = CustomerData.iloc[:,1:].values.astype(object)
#finding the optimal cluster_number k
for n in range(2,8):
    kproto = KPrototypes(n_clusters = n, init = 'Cao')
    clusters = kproto.fit_predict(X, categorical = [0])
    silhouette = silhouette_score(X[:,1:],clusters)
    print('number of clusters:', n)
    print('  cost: ',kproto.cost_)
    print('  average silhouette score: ',silhouette)


# In[42]:


#clustering with kprototypes with k = 3
kproto = KPrototypes(n_clusters = 3, init = 'Cao')
clusters = kproto.fit_predict(X, categorical = [0])
print('Cluster Centers:\n', kproto.cluster_centroids_)


# In[43]:


#comparison plots
sns.pairplot(Clustered.drop(columns=['UK?','CustomerID']), hue='Cluster')
plt.suptitle('Scatter Matrix Within Clusters', fontsize = 15, y = 1.05)
plt.show()


# In[45]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


df = pd.read_csv("C:/Users/DELL/Desktop/msvn/OnlineRetail.csv")
print(df.shape)
df.head()


# In[47]:


customer_country=df[['Country','CustomerID']].drop_duplicates()
customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[48]:


# Drop cancelled orders
df = df.loc[df['Quantity'] > 0]

# Drop records with missing Customer ID
df = df[pd.notnull(df['CustomerID'])]

# Drop incomplete month
df = df.loc[df['InvoiceDate'] < '2011-12-01']

# Calculate total sales from the Quantity and UnitPrice
df['Sales'] = df['Quantity'] * df['UnitPrice']


# In[49]:


# use groupby to aggregate sales by CustomerID
customer_df = df.groupby('CustomerID').agg({'Sales': sum, 
                                            'InvoiceNo': lambda x: x.nunique()})

# Select the columns we want to use
customer_df.columns = ['TotalSales', 'OrderCount'] 

# create a new column 'AvgOrderValu'
customer_df['AvgOrderValue'] = customer_df['TotalSales'] / customer_df['OrderCount']

customer_df.head(20)


# In[50]:


ranked_df = customer_df.rank(method='first')
ranked_df.head()


# In[51]:


# Center values around 0 with standard deviation of 1
ranked_df = customer_df.rank(method='first')
normalized_df = (ranked_df - ranked_df.mean()) / ranked_df.std()
normalized_df.head(10)


# In[52]:


normalized_df.describe()


# In[53]:


# Use the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
krange = list(range(2,11))
X = normalized_df[['TotalSales','OrderCount','AvgOrderValue']].values
for n in krange:
    model = KMeans(n_clusters=n, random_state=3)
    model.fit_predict(X)
    # Store labels of clusters in label_ attribute of kmeans model object
    cluster_assignments = model.labels_
    # Store cluster centers of clusters in cluster_center_ attribute of kmeans model object
    centers = model.cluster_centers_
    wcss.append(np.sum((X - centers[cluster_assignments]) ** 2))

plt.plot(krange, wcss)
plt.title('The Elbow Method')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of Squares (WCSS)")
plt.show()


# In[54]:


model = KMeans(n_clusters=4).fit(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])

four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)

# Store labels of clusters in labels_ attribute of the kmeans model object
four_cluster_df['Cluster'] = model.labels_

four_cluster_df.head(10)


# In[65]:


cluster1_metrics = KMeans.cluster_centers_[0]
cluster2_metrics = kmeans.cluster_centers_[1]
cluster3_metrics = kmeans.cluster_centers_[2]
cluster4_metrics = kmeans.cluster_centers_[3]

data = [cluster1_metrics, cluster2_metrics, cluster3_metrics, cluster4_metrics]
cluster_center_df = pd.DataFrame(data)

cluster_center_df.columns = four_cluster_df.columns[0:3]
cluster_center_df


# In[66]:


plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['TotalSales'],
    c='red', label='Cluster 1')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['TotalSales'],
    c='blue', label='Cluster 2')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['TotalSales'],
    c='green', label='Cluster 3')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['TotalSales'],
    c='magenta', label='Cluster 4')

plt.title('TotalSales vs. OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')
# plt.legend()
plt.grid()
plt.show()


plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['AvgOrderValue'],
    c='red', label='Cluster 1')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['AvgOrderValue'],
    c='blue', label='Cluster 2')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['AvgOrderValue'],
    c='green', label='Cluster 3')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['AvgOrderValue'],
    c='magenta', label='Cluster 4')

plt.title('AvgOrderValue vs. OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')
# plt.legend()
plt.grid()
plt.show()


plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['TotalSales'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['AvgOrderValue'],
    c='red', label='Cluster 1')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['TotalSales'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['AvgOrderValue'],
    c='blue', label='Cluster 2')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['TotalSales'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['AvgOrderValue'],
    c='green', label='Cluster 3')

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['TotalSales'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['AvgOrderValue'],
    c='magenta', label='Cluster 4')

plt.title('AvgOrderValue vs. TotalSales Clusters')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')
plt.legend()
plt.grid()
plt.show()


# In[67]:


high_val_cluster = four_cluster_df.loc[four_cluster_df['Cluster'] == 2]

pd.DataFrame(df.loc[df['CustomerID'].isin(high_val_cluster.index)].groupby(
    'Description').count()['StockCode'].sort_values(ascending=False).head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




