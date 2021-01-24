#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


retail=pd.read_csv("C:/Users/DELL/Desktop/msvn/retail.csv")
retail.head()
retail.tail()


# In[3]:


retail.info()


# In[4]:


retail.shape


# In[7]:


retail['Description'].value_counts()


# In[8]:


retail['totalutility'] = retail['Quantity'] * retail['UnitProfit']


# In[9]:


retail.head()


# In[10]:


nbTran= len(retail.Tid.unique())
nbProd = len(retail.totalutility.unique())
nbCustom= len(retail.Description.unique())
plt.bar(["transactions", "totalutility", "Description"],[nbTran,nbProd , nbCustom], color=["blue", "green", "red"])
plt.title('total number of transactions, totalutility, Description')
plt.ylabel('Total number')


# In[11]:


len(retail['Tid'].unique())


# In[12]:


len(retail[retail['CustomerID'].isnull()]['Tid'].unique())


# In[13]:


len(retail['CustomerID'].unique())


# In[14]:


sum(retail['totalutility'])


# In[17]:


display(retail[retail['totalutility'] <= 1])


# In[18]:


vtrans=len(retail.Tid.unique())
ctrans= len([i for i in retail.Tid.unique() if str(i).startswith('C')])
plt.pie([vtrans, ctrans], labels=["high utility items", "low utility items"], autopct="%1.2f%%")
plt.legend()
plt.title("total number of high and low transactions")


# In[19]:


retail.drop(retail[retail["Tid"].apply(lambda x:str(x).startswith('C'))].index, axis=0, inplace=True)


# In[20]:


retail.head()


# In[21]:


display(retail[(retail.Quantity <= 0) & (retail.UnitProfit <= 0)].shape[0])


# In[54]:


retail[(retail.UnitProfit==0)  & ~(retail['CustomerID'].isnull())]

display(retail[(retail.UnitProfit == 0)  & ~(retail['CustomerID'].isnull())].shape[0])


# In[22]:


retail[retail['CustomerID'].isnull()].shape[0]


# In[23]:


def rstr(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1, sort=True)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=True)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str


# In[25]:


retail_cl = retail[~(retail['totalutility'].isnull())]


# Remove negative or return transactions
retail_cl = retail_cl[~(retail_cl.Quantity < 0)]
retail_cl = retail_cl[retail_cl.UnitProfit > 0]

# passing dataframe to rstr function to display statistics of the dataframe
details = rstr(retail_cl)

display(details.sort_values(by='distincts', ascending=False))


# In[26]:


item utility = retail_cl.groupby('Description')['totalutility'].agg(np.sum).sort_values(ascending=False)
item utility.head(20)


# In[27]:


retail_cl.columns = retail_cl.columns.str.replace(' ', '').str.lower()
retail_cl.columns


# In[29]:


ret_df = retail_cl.groupby(["itemid", "description"]).count().reset_index()

display(ret_df.itemid.value_counts()[ret_df.itemid.value_counts()> 1].reset_index().head())

retail_cl[retail_cl['itemid'] == ret_df.itemid.value_counts()[ret_df.itemid.value_counts() > 1]
      .reset_index()['index'][4]]['description'].unique()


# In[30]:


from pandasql import sqldf

pysqldf = lambda q:sqldf(q,globals())

unique_desc = retail_cl[["itemid", "description"]].groupby(by=["itemid"]).                apply(pd.DataFrame.mode).reset_index(drop=True)
q = '''
select df.tid, df.itemid, un.description, df.quantity, df.invoicedate,
       df.unitprofit, df.customerid, df.country, df.totalutility
from retail_cl as df INNER JOIN 
     unique_desc as un on df.itemid = un.itemid
'''

retail_cl = pysqldf(q)


# In[ ]:


retail_cl.invoicedate = pd.to_datetime(retail_cl.invoicedate)

retail_cl.customerid = retail_cl.customerid.astype('Int64')

details = rstr(retail_cl)
display(details.sort_values(by='distincts', ascending=False))


# In[40]:


customer = retail_cl.groupby('itemid')['totalutility'].agg(np.sum).sort_values(ascending=False)
customer.head(20)


# In[41]:


sim_stock = retail_cl.groupby('description')
stock_util = sim_stock['unitprofit'].agg(np.sum).sort_values(ascending=False)
stock_util.head(20)


# In[43]:


st = stock_util.reset_index()
st
st.description[:10]


# In[ ]:


item = retail[0]
item_uti = retail[2]
trans_uti = retail[1]

item=item.str.split()
item=item.tolist()
item=[list( map(int,i) ) for i in item]

item_uti=item_uti.str.split()
item_uti=item_uti.tolist()
item_uti=[list( map(int,i) ) for i in item_uti]


trans_uti=trans_uti.tolist()

thmin=100

#phase1


ubtu=[]

for i in item_uti:
    ubtuq=0
    for j in i:
        if j>0:
            ubtuq=ubtuq+j
    ubtu.append(ubtuq)
    
#tid
    
def index_2d(myList, v):
    p=[]
    for i in myList:
        if v in i:
            p.append(myList.index(i))
    return (p)
    
tidi=[]
tidt=[]
htwui=[]

for i in item:
    for j in i:
        if j not in tidi:
            tidi.append(j)
            tidt.append(index_2d(item,j))
#tid
            
rules = apriori(item, min_support=0.5,  min_confidence=1)

result=rules[0]
l=[]
j=[]
[l.extend([v]) for k,v in result.items()]
for i in l:
    j=list(i)
    for k in j:
        htwui.append(list(k))
        
htwuis=[]

for i in htwui:
    ubtwu=0
    for j in item:
        if i in j:
            ubtwu=ubtwu+ubtu[item.intdex(j)]
    
    if ubtwu>thmin:
       htwuis.append(i)
       
#phase 2:
    
huis=[]
 
for i in htwuis:
    u=0
    for j in item:
        uj=0
        if i in j:
            for k in i:
                indj=item.index(j)
                indk=j.index(k)
                p=item_uti[j]          
            uj=uj+p[k]
        u=u+uj
        
    if u>thmin:
        huis.append(i)
    
print(huis)       


# In[44]:


st = stock_util.reset_index()

plt.figure(figsize=(10,3))
stock_util.head(10).plot.bar(color='firebrick')
plt.bar(stock_util.head(10),height=0.5 ,color='firebrick' )

plt.xlabel("Item Description",fontsize=16)
plt.xticks(rotation=75)
plt.ylabel("totalutility",fontsize=16)

plt.title("Distribution of high utility itemsets", fontsize=20)
plt.show()


# In[45]:


len(retail_cl[retail_cl['description'].isnull()])


# In[46]:


for i, d in retail_cl[retail_cl['description'].isnull()].iterrows():
  retail_cl['description'][i] = 'Code-' + str(d['itemid'])
  
retail_cl.head()


# In[48]:


import datetime

ref_date = retail_cl.invoicedate.max() + datetime.timedelta(days = 1)
print('Reference Date:', ref_date)

retail_cl['days_since_last_purchase'] = (ref_date - retail_cl.invoicedate).astype('timedelta64[D]')
cust_history =  retail_cl[['tid', 'days_since_last_purchase']].groupby("tid").min().reset_index()
cust_history.rename(columns={'days_since_last_purchase':'recency'}, inplace=True)
cust_history.describe().transpose()


# In[49]:


from scipy.stats import norm, probplot, skew

def QQ_plot(data, measure):
    fig = plt.figure(figsize=(20,7))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.distplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('utility')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()

QQ_plot(cust_history.recency, 'Recency')


# In[ ]:




