#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train=pd.read_csv("C:/Users/DELL/Desktop/msvn/Train.csv")
test=pd.read_csv("C:/Users/DELL/Desktop/msvn/Test.csv")


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


Sales.describe(include="all")


# In[5]:


train.describe()


# In[6]:


idsunique=len(set(train.Item_Identifier))
idstotal=train.shape[0]
idsdupli=idstotal-idsunique
print("There are "  + str(idsdupli)  +  " duplicate IDs for "  +  str(idstotal) + " total entries. ")


# In[8]:


Sales["Qty_Sold"] = (Sales["Item_Outlet_Sales"]/Sales["Item_MRP"])
Sales.head()


# In[7]:


#item outlet sales.
plt.figure(figsize=(10,5))
sns.distplot(train['Item_Outlet_Sales'],bins=25)
plt.xlabel('Item_outlet_sales')
plt.ylabel('Number of Sales')
plt.title('Item Outlet Sales Distribution')
plt.grid()


# In[8]:


#the distribution is right skewed.
print('Skew is:', train.Item_Outlet_Sales.skew())
print('Kurtosis :' , train.Item_Outlet_Sales.kurt())


# In[9]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[10]:


#correltion between target and predictor varibales.
corr=numeric_features.corr()
corr



# In[11]:


print(corr['Item_Outlet_Sales'].sort_values(ascending=False))


# In[12]:


#correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(corr, vmax=.8, square=True, annot=True)


# In[ ]:





# In[14]:


#Item_Fat_Content
sns.countplot(train.Item_Fat_Content)



# In[15]:


#Item_type
sns.countplot(train.Item_Type, palette ='Blues')
plt.xticks(rotation=90)


# In[16]:


#Outlet sales
sns.countplot(train.Outlet_Size)


# In[17]:


#Outlet Type
sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)


# In[18]:


numeric_features.dtypes


# In[19]:


#Item weight and Item Outlet Sales
plt.figure(figsize=(10,8))
sns.scatterplot(x=train.Item_Weight,y=train.Item_Outlet_Sales ,alpha=0.3)
plt.title('Item Weight Vs Outlet Sales')
plt.grid()


# In[20]:


#Item Visiblity vs Item Outlet Sales
plt.figure(figsize=(10,8))
sns.scatterplot(x=train.Item_Visibility,y=train.Item_Outlet_Sales,alpha=0.3)
plt.title('Item Visibility Vs Item Outlet Sales')
plt.grid()


# In[22]:


plt.figure(figsize=(7,7))
sns.barplot(y=train.Item_Type,x=train.Item_Outlet_Sales,palette='Blues')
plt.title('Impact of Item Type on Item Outlet Sales')


# In[31]:


plt.figure(figsize=(7,7))
sns.barplot(y=train.Item_Type,x=train.Item_Visibility,palette='Reds')
plt.title('Item Type Impact on Item Visibility')


# In[24]:


#item fat content vs outlet sales
sns.barplot(x=train.Item_Fat_Content,y=train.Item_Outlet_Sales,palette='Greens')
plt.title('Impact of Item Fat Content on Item Outlet Sales')


# In[30]:


#outlet identifier vs item oulet sales
plt.figure(figsize=(7,4))
sns.barplot(x=train.Outlet_Identifier,y=train.Item_Outlet_Sales, palette = 'Greens')
plt.title('Impact of Outlet Identifier on Item Oulet sales')


# In[29]:


Sales['Item_Type'].value_counts()


# In[26]:


train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())


# In[28]:


train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())


# In[29]:


sns.barplot(x=train.Outlet_Size,y=train.Item_Outlet_Sales,palette='Greens')
plt.title('Impact of Outlet Location on Item Outlet sales')


# In[32]:


Sales = pd.get_dummies(Sales,columns=["Item_Type","Outlet_Identifier","Outlet_Location_Type","Outlet_Type"],drop_first=True)


# In[33]:


Sales.head()


# In[34]:


Sales.columns


# In[35]:


X_Cols = ['Item_Fat_Content', 'Item_MRP',
       'Item_Visibility', 'Item_Weight',
       'Qty_Sold', 'Outlet_Age', 'Item_Type_DBBS',
       'Item_Type_Drinks', 'Item_Type_Frozen_Canned', 'Item_Type_Fruit_Veg',
       'Item_Type_HH_HH', 'Item_Type_Others', 'Item_Type_Seafood_Meat',
       'Outlet_Identifier_OUT013', 'Outlet_Identifier_OUT017',
       'Outlet_Identifier_OUT018', 'Outlet_Identifier_OUT019',
       'Outlet_Identifier_OUT027', 'Outlet_Identifier_OUT035',
       'Outlet_Identifier_OUT045', 'Outlet_Identifier_OUT046',
       'Outlet_Identifier_OUT049', 'Outlet_Location_Type_Tier 2',
       'Outlet_Location_Type_Tier 3', 'Outlet_Type_Supermarket Type1',
       'Outlet_Type_Supermarket Type2', 'Outlet_Type_Supermarket Type3']
y_cols = 'Outlet_Size'


# In[36]:


from sklearn.model_selection import train_test_split
X = Sales.loc[(Sales[y_cols].notnull()) & (Sales['Type'] == "train"), X_Cols]
y = Sales.loc[(Sales[y_cols].notnull()) & (Sales['Type'] == "train"), y_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)


# In[37]:


print("Shape of X_train: ",X_train.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of X_test: ",X_test.shape)
print("Shape of y_test: ",y_test.shape)


# In[38]:


Sales[y_cols].value_counts()


# In[39]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier()


# In[43]:


def IdentifyKValueCrossValidation(X,Y,startK,endK,cv,scoring):
    k_range = list(range(startK, endK+1))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, Y, cv=cv, scoring=scoring)
        k_scores.append(scores.mean())
    z = [i for i, j in enumerate(k_scores) if j == max(k_scores)]
    
    print("Location for Max Accuaracy is:")
    
    for i in z:
        print(k_range[i])
    
    # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    
    return k_range[i]

def metrices(Predicted,Actual):
    print("Confusion Matrix for the model is:\n\n {}".format(metrics.confusion_matrix(y_pred=Predicted,y_true=Actual)))
    print("\nAccuracy for the model is: {}".format(metrics.accuracy_score(y_pred=Predicted, y_true= Actual)))
    print("\nArea under the curve for the model is: {}".format(metrics.roc_auc_score(y_score=Predicted,y_true=Actual)))
    print("\nClassification Report for the model is:\n {}".format(metrics.classification_report(y_pred=Y_Predict,y_true=y_test)))


# In[50]:


from sklearn.cross_validation import cross_val_score
K = IdentifyKValueCrossValidation(X=X,Y=y,cv=5,startK=1,endK=50,scoring="accuracy")
print("Value of K with is: {}".format(K))


# In[45]:


knn = KNeighborsClassifier(n_neighbors=3)
y_predicted = knn.fit(X_train,y_train).predict(X_test)
print("Test Accuracy: ", (y_predicted == y_test).astype(int).sum()/y_test.shape[0])


# In[46]:


pd.Series(knn.predict(X=Sales.loc[(Sales[y_cols].isnull()) & (Sales['Type'] == "train"), X_Cols])).value_counts()


# In[49]:


sns.distplot(sales_train['Item_Weight'],bins=20,rug=True,kde=True)
sns.despine()


# In[ ]:




