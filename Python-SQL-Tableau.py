#!/usr/bin/env python
# coding: utf-8

# In[180]:


import pandas as pd


# In[181]:


raw_csv_data = pd.read_csv('D:\\OneDrive - office365hubs.com\\!Python + SQL + Tableau\\Course_Resources\\5. Preprocessing\Original\\Absenteeism_data.csv')


# In[182]:


raw_csv_data


# In[183]:


df = raw_csv_data.copy()


# In[184]:


df


# In[185]:


type(df)


# In[186]:


df.info()


# In[187]:


display(df)


# In[188]:


df.drop(['ID'],axis = 1)


# In[189]:


df = df.drop(['ID'],axis=1)


# In[190]:


df


# In[13]:


raw_csv_data


# In[191]:


df['Reason for Absence']


# In[15]:


#Sorting a list of numbers
sorted(df['Reason for Absence'].unique())


# In[192]:


#Splitting a column in multiple dummies
#converts categorical variable into dummy variable get_dummies()
reason_columns = pd.get_dummies(df['Reason for Absence'])
reason_columns.head(10)


# In[193]:


reason_columns['check'] = reason_columns.sum(axis=1)


# In[194]:


reason_columns.head(1)


# In[195]:


reason_columns['check'].sum(axis=0)


# In[196]:


reason_columns['check'].unique()


# In[197]:


reason_columns = reason_columns.drop(['check'],axis=1)


# In[198]:


reason_columns.head(2)


# In[199]:


#Dummy variable and their statistical importance
#We are gona grop the first column with the reason 0
reason_columns = pd.get_dummies(df['Reason for Absence'],drop_first=True)


# In[200]:


reason_columns


# In[201]:


#Grouping-Transforming dummy var in categorical variable
#group the reason of Absence
df.columns.values


# In[202]:


reason_columns.columns.values


# In[203]:


df = df.drop(['Reason for Absence'],axis=1)


# In[204]:


df.head()


# In[205]:


#Adding reason_columns data set to df data est
#Clasification or grouping
#Split the reason_columns data frame
reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:,22:].max(axis=1)


# In[206]:


#Concatenating columns in Python
df.columns.values


# In[207]:


reason_columns.columns.values


# In[208]:


#joinnig the columns 
df = pd.concat([df,reason_type_1,reason_type_2,
                reason_type_3,reason_type_4],axis=1)


# In[209]:


df.head()


# In[210]:


df.columns.values


# In[211]:


column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']


# In[213]:


df.columns =column_names


# In[214]:


df.head()


# In[215]:


#Reorder the columns of a data frame
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                          'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']


# In[216]:


df = df[column_names_reordered]


# In[217]:


df.head()


# ## Create a checkpoint

# In[218]:


#Create a checkp[oint]
#For a checkpoint you have to create a copy of the data frame in the state that is now
df_reason_mod = df.copy()


# In[219]:


df_reason_mod.head()


# In[220]:


df_reason_mod.columns.values


# In[221]:


df_reason_mod['Date']


# In[222]:


type(df_reason_mod['Date'][0])


# In[223]:


#using pandas timestamp
#when doing the conversion from text to date specify the proper 
#format of the date you will be working on
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'],
                                       format='%d/%m/%Y')


# In[224]:


type(df_reason_mod['Date'])


# In[225]:


df_reason_mod.info()


# # Extract the Month value

# In[226]:


df_reason_mod['Date'][0]


# In[227]:


df_reason_mod['Date'][0].month


# In[228]:


df_reason_mod.shape


# In[243]:


list_months = []
for i in range(700):
    list_months.append(df_reason_mod['Date'][i].month)


# In[231]:


list_months


# In[232]:


len(list_months)


# In[233]:


df_reason_mod['Month Value'] = list_months


# In[234]:


df_reason_mod.head()


# In[235]:


df_reason_mod.columns.values


# ## Extracting the day of the week

# In[236]:


df_reason_mod['Date'][699].weekday()


# In[237]:


df_reason_mod['Date'][699]


# In[240]:


def date_to_weekday(date_value):
    return date_value.weekday()


# In[241]:


df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)


# In[242]:


df_reason_mod.head()


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




