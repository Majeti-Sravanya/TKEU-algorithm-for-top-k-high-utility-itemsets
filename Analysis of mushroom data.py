#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd 


# In[35]:


data=pd.read_csv("C:/Users/DELL/Desktop/msvn/mushrooms.csv")


# In[36]:


data.head()


# In[37]:


data.info()


# In[38]:


for key, value in data.iteritems():
    print(data[key].value_counts(), "\n")


# In[39]:


X = data.drop("class", axis=1)
y = data["class"].values.copy()

X.describe()


# In[40]:


y


# In[41]:


from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
print(len(X_test))

print(len(y_train))
print(len(y_test))


# In[42]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

cat_pipeline = Pipeline([
    ("cat_encoder", OneHotEncoder(sparse=False)),
    #("cat_encoder", OneHotEncoder()),
])


# In[43]:


X_train = cat_pipeline.fit_transform(X_train)
X_train.shape


# In[44]:


X_test = cat_pipeline.transform(X_test)
X_test


# In[45]:


y_train_p = (y_train == 'p')
y_test_p = (y_test == 'p')


# In[46]:


from sklearn.linear_model import SGDClassifier

sgdc = SGDClassifier(random_state=42)
sgdc.fit(X_train, y_train_p)


# In[47]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgdc, X_test, y_test_p, cv=3, scoring="roc_auc")


# In[48]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_test_pred = cross_val_predict(sgdc, X_test, y_test_p, cv=3)
confusion_matrix(y_test_p, y_test_pred)


# In[49]:


from sklearn.metrics import precision_score, recall_score

print(precision_score(y_test_p, y_test_pred))
print(recall_score(y_test_p, y_test_pred))


# In[50]:


from sklearn.metrics import f1_score

f1_score(y_test_p, y_test_pred)


# In[51]:


# plot within jupyter
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# get the scores
y_scores = cross_val_predict(sgdc, X_test, y_test_p, cv=3, method="decision_function")

# metrics for plotting the curve
precisions, recalls, thresholds = precision_recall_curve(y_test_p, y_scores)

# from ageron/handson-ml2
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown

plt.figure(figsize=(8, 4))                      # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([7813, 7813], [0., 0.9], "r:")         # Not shown
plt.plot([-50000, 7813], [0.9, 0.9], "r:")      # Not shown
plt.plot([-50000, 7813], [0.4368, 0.4368], "r:")# Not shown
plt.plot([7813], [0.9], "ro")                   # Not shown
plt.plot([7813], [0.4368], "ro")                # Not shown
plt.show()


# In[19]:


from sklearn.metrics import roc_curve

# get the metrics for the roc curve
fpr, tpr, thresholds = roc_curve(y_test_p, y_scores)

# from ageron/handson-ml2
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") # Not shown
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  # Not shown
plt.plot([4.837e-3], [0.4368], "ro")               # Not shown
plt.show()


# In[20]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# function to print out classification model report
def classification_report(model_name, test, pred, label="1"):
    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.4f}'.format(accuracy_score(test, pred)))
    print("     Precision: ", '{:,.4f}'.format(precision_score(test, pred, pos_label=label)))
    print("        Recall: ", '{:,.4f}'.format(recall_score(test, pred, pos_label=label)))
    print("      F1 score: ", '{:,.4f}'.format(f1_score(test, pred, pos_label=label)))


# In[21]:


from sklearn.neighbors import KNeighborsClassifier

knnc = KNeighborsClassifier(weights='distance', n_neighbors=4)
knnc.fit(X_train, y_train)


# In[22]:


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_test, y_test, cv=10)
forest_scores.mean()


# In[26]:


y_forest_pred = knnc.predict(X_test)
classification_report("Test data - Random Forest Classifier report", y_test, y_forest_pred, "p")


# In[27]:


from sklearn.naive_bayes import GaussianNB

clf_GNB = GaussianNB()
clf_GNB = clf_GNB.fit(X_train, y_train)


# In[28]:


import seaborn as sns

y_pred_GNB = clf_GNB.predict(X_test)
cfm = confusion_matrix(y_test, y_pred_GNB)

sns.heatmap(cfm, annot=True,  linewidths=.4, cbar=None)
plt.title('Gaussian Naive Bayes confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[23]:


from sklearn.preprocessing import LabelEncoder

# split the set again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding the variable
label_encoder = LabelEncoder()
X_train[X_train.columns] = X_train[X_train.columns].apply(lambda col: label_encoder.fit_transform(col))
X_train.head()


# In[25]:




# fit the random forest model using the new X_train, label encoded
forest_clf.fit(X_train, y_train)


import pandas as pd

# get the importances
feat_importances = pd.Series(forest_clf.feature_importances_, index=X_train.columns)


# In[30]:


# set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# set the style of the axes and the text color
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'

# create dataframe then sort feature importances (top 5)
df = pd.DataFrame({'utility' : feat_importances.nlargest(10)})
df = df.sort_values(by='utility')

# we first need a numeric placeholder for the y axis
my_range=list(range(1,len(df.index)+1))

fig, ax = plt.subplots(figsize=(15,8))

# create for each importance type an horizontal line that starts at x = 0 with the length 
plt.hlines(y=my_range, xmin=0, xmax=df['utility'], color='#007ACC', alpha=0.2, linewidth=20)

# create for each expense type a dot at the level of the expense percentage value
plt.plot(df['utility'], my_range, "o", markersize=20, color='#007ACC', alpha=0.6)

# set labels
ax.set_xlabel('utility', fontsize=15, fontweight='black', color = '#333F4B')
ax.set_ylabel('')

# set axis
ax.tick_params(axis='both', which='major', labelsize=12)
plt.yticks(my_range, df.index)

# add an horizonal label for the y axis 
fig.text(-0.23, 0.96, 'Features', fontsize=15, fontweight='black', color = '#333F4B')

# change the style of the axis spines
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

# set the spines position
ax.spines['bottom'].set_position(('axes', -0.04))
ax.spines['left'].set_position(('axes', 0.015))

plt.savefig('hist2.png', dpi=300, bbox_inches='tight')


# In[52]:


from sklearn.decomposition import PCA


# In[ ]:





# In[ ]:




