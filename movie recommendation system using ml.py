#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[7]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[8]:


df.head()


# In[5]:


df.shape


# In[9]:


df['user_id']


# In[10]:


df['user_id'].nunique()


# In[11]:


df['item_id'].nunique()


# In[12]:


movies_title=pd.read_csv('u.item',sep="\|",header=None)


# In[13]:


movies_title.shape


# In[17]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]
movies_titles.head()


# In[18]:


df=pd.merge(df,movies_titles,on="item_id")


# In[19]:


df


# In[21]:


df.tail()


# In[22]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[23]:


ratings.head()


# In[24]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# #Create the Recommendar System

# In[25]:


df.head()


# In[26]:


moviemat=df.pivot_table(index="user_id",columns="title",values="rating")


# In[27]:


moviemat.head()


# In[31]:


starwars_user_ratings=moviemat['Star Wars (1977)']


# In[33]:


starwars_user_ratings.head(20)


# In[34]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)


# In[35]:


similar_to_starwars


# In[41]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])


# In[37]:


corr_starwars.dropna(inplace=True)


# In[38]:


corr_starwars


# In[39]:


corr_starwars.head()


# In[42]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[43]:


ratings


# In[44]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[45]:


corr_starwars


# In[46]:


corr_starwars.head()


# In[48]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False)


# In[57]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)
    
    return predictions


# In[59]:


predict_my_movie=predict_movies("Titanic (1997)")


# In[60]:


predict_my_movie.head()


# In[ ]:




