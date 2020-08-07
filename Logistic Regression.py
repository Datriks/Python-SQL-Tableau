#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT ALL THE LIBRARIES BELOW
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',200)
print('StandardImport Completed')


# In[2]:


data_preprocessed = pd.read_csv('D:\\OneDrive - office365hubs.com\\!Python + SQL + Tableau\\Absenteeism_preprocessed.csv')


# In[3]:


data_preprocessed.head()


# ## Create the targets

# In[4]:


data_preprocessed['Absenteeism Time in Hours'].median()


# In[5]:


targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(),1,0)


# In[6]:


targets


# In[7]:


data_preprocessed['ExcesiveAbsenteeism'] = targets


# In[8]:


data_preprocessed


# In[9]:


(targets.sum() / targets.shape[0])*100


# In[10]:


data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'],axis = 1)


# In[11]:


data_with_targets is data_preprocessed


# In[12]:


data_with_targets.head()


# ## Select the inputs for the regression

# In[13]:


data_with_targets.shape


# In[14]:


#Select the imnputs for our regression
data_with_targets.iloc[:,0:14]


# In[15]:


data_with_targets.iloc[:,:-1]


# In[16]:


unscaled_inputs = data_with_targets.iloc[:,:-1]


# In[17]:


unscaled_inputs.head()


# ## Standardise the data

# In[18]:


#here we prepare the scaling mechanism
from sklearn.preprocessing import StandardScaler
#absenteeism _scaler will be used to substract the mean and divide
#by the standard deviation variablewise
absenteeism_scaler = StandardScaler()


# In[19]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler=StandardScaler(copy,with_mean,with_std)
        self.columns=columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self,X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_=np.mean(X[self.columns])
        self.var_=np.var(X[self.columns])
        return self
    
    def transform(self,X,y=None,copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]),columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled,X_scaled],axis=1)[init_col_order]


# In[20]:


unscaled_inputs.columns.values


# In[21]:


#columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',
       #'Age', 'Daily Work Load Average', 'Body Mass Index','Children', 'Pets']
columns_to_omit = ['Reason_1','Reason_2','Reason_3','Reason_4','Education']    


# In[22]:


columns_to_scale = [x for x in unscaled_inputs.columns.values 
                    if x not in columns_to_omit]


# In[23]:


absenteeism_scaler = CustomScaler(columns_to_scale)


# In[24]:


#this will calculate and store the mean and standard deviation for each element
absenteeism_scaler.fit(unscaled_inputs)


# In[25]:


#transform the unscaled inputs:substarct the mean and divide bi standard deviation
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[26]:


#new_dsata_raw = pd.read_csv('new_data.csv')
#new_data_scaled = absenteeism_scale.transform(new_data_raw)


# In[27]:


scaled_inputs


# In[28]:


scaled_inputs.shape


# ## Split the data into train and test and shufle

# ## Import the relevant module

# In[29]:


from sklearn.model_selection import train_test_split


# ## Split

# In[30]:


train_test_split(scaled_inputs,targets)


# In[31]:


#x_train,x_test,y_train,y_test = train_test_split(scaled_inputs,targets, train_size=0.9)
#means that 90% used for trainin and only 10% for testing
x_train,x_test,y_train,y_test = train_test_split(scaled_inputs,targets,train_size = 0.8,random_state=20)


# In[32]:


print(x_train.shape,y_train.shape)


# In[33]:


print(x_test.shape,y_test.shape)


# In[34]:


#sklearn.mode_selection.train_test_split(inputs ,targets,train_size,
#shuffle=True,random_state)
#split arrays or matrices into random train and test subsets


# ## Logistic Regression with sklearn

# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### Training the model

# In[36]:


reg = LogisticRegression()


# In[37]:


#sklearn.linear_model.LogisticRegression.fit(x,y)
#fits the model according to the given training data
reg.fit(x_train,y_train)


# In[38]:


#sklearn.linear_model.LogisticRegression.score(inputs,targets)
#returns the mean accuracy on the given test data and labels
reg.score(x_train,y_train)
#the model is 78.3928 correct
#the model learn to clasify 78% of our data correctly


# ### Manually check the accuracy

# In[39]:


#sklearn.linear_model.LogisticRegression.predict(inputs)
# predicts class labels (logistic regression outputs)for given input samples
model_outputs = reg.predict(x_train)


# In[40]:


#predictions
model_outputs


# In[41]:


# targets
y_train


# In[42]:


model_outputs == y_train


# In[43]:


np.sum(model_outputs==y_train)


# In[44]:


#Accuracy = Correct Predictions / nr. observation


# In[45]:


model_outputs.shape[0]


# In[46]:


#Accuracy
np.sum((model_outputs==y_train))/model_outputs.shape[0]


# ### Finding the intercept and coefficients

# In[47]:


reg.intercept_


# In[48]:


reg.coef_


# In[49]:


unscaled_inputs.columns.values


# In[50]:


feature_name = unscaled_inputs.columns.values


# In[51]:


#Create a dataframe to contain the intercept,the feature_name,coeficients
summary_table = pd.DataFrame(columns=['Feature name'],
                             data=feature_name)

summary_table['Coefficient']=np.transpose(reg.coef_)

summary_table


# In[52]:


#shift the whole data frame down one row
summary_table.index = summary_table.index+1
summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# ### Interpreting the coefficients

# In[53]:


#log(odds) =intercept +b1x1+b2x2+...+b14x14
#where b are the coeficients


# In[54]:


summary_table['Odds_ratio']=np.exp(summary_table.Coefficient)
summary_table


# In[55]:


# DataFrame.sort_values(Series) sort the values in a data frame in respect to a given column(Series)
summary_table.sort_values('Odds_ratio',ascending=False)


# In[56]:


# if coef is close to 0 means not important: will be multiplied with 0 
# with this reselts that Daily Work load Average is close to 0  
#and becomes non important as does Distance to Work and Day of the Week
#based on the features given these do not make any difference
#Reason_0 is no reason has been chosen as the base for our model- no reason


# In[57]:


#need to go back to where we standardised the data
#and put the code in comments with #


# ### Testing the Model

# In[58]:


#Find the accuracy
reg.score(x_test,y_test)


# In[59]:


#based on the data that the test has not seen before in 74,2%
#of the cases the model will predict that a person is going to 
#be excessively absent
#test accuracy is going to be sless than train accuracy due to overfiting


# In[60]:


#instead of 0 and 1 we can get the probability of an output being 0 and 1
predicted_proba = reg.predict_proba(x_test)
predicted_proba
#first column probability of being 0
#second column probability of being 1


# In[61]:


predicted_proba.shape


# In[62]:


predicted_proba[:,1]


# In[63]:


#in reality logistic regression models calc these probabilities in background
#if the probability is below 0.5 it places a 0
#if prob is above 0.5 it places a 1


# In[64]:


#1. Save the model
#2.Create module
#Get new data and pass it through SQL and analise it in Tableau


# # Save the Model

# In[65]:


#save the reg object
import pickle 


# In[66]:


with open('model','wb') as file:
    pickle.dump(reg, file)
    #model is the file name; wb is write bytes and dump is save the file    


# In[68]:


# pickle the scaler file
with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler, file)


# In[ ]:


#Ceate a mechanism to load the model and deploy it or make predictions


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
