import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

#Data Acquisition 
Master_Data = pd.read_csv('C:/Users/PRIYA/Desktop/S6/Movielens.csv', sep = ',', engine = 'python')

Master_Data.head(2)

Master_Data = Master_Data.loc[:, ~Master_Data.columns.str.contains('^Unnamed')]

Master_Data.head(2)

Master_Data.info()

Master_Data.describe()

#visualize user age distribution

Master_Data.groupby('Age')['UserID'].count()

Master_Data.groupby('Age')['UserID'].count().plot(kind = 'bar', color = 'green',figsize = (8,7))
plt.xlabel('Age')
plt.ylabel('Number of Users in Population')
plt.title('Visualization of User Age Distribution')
plt.show

sns.distplot(Master_Data['Age'], color = 'g', bins = 45)
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.title('User Age Distribution')
plt.show

#visualize rating of the movie Toy Story

Master_Data[Master_Data.Title == 'Toy Story (1995)'].groupby('Rating')['UserID'].count()

#viewership of the movie Toy Story by age Group
Master_Data[Master_Data.Title == 'Toy Story (1995)'].groupby('Age')['MovieID'].count()

#Top 25 movies by viewers rating

Master_Data.groupby('MovieID')['Rating'].count().sort_values(ascending = False)[:25]

Master_Data.groupby('MovieID')['Rating'].count().nlargest(25)

Master_Data.groupby('MovieID')['Rating'].count().sort_values(ascending = False)[:25].plot(kind ='barh', color = 'g', x = 'Rating', y = 'Number of Users', title = 'User Rating of Toy Story (1995) Movie', figsize = (10,9))
plt.xlabel('Rating')
plt.ylabel('Top 25 MovieID')
plt.title('Visualization of Top 25 Movies Viewership Rating')
plt.show

#rating for a particular user of userid=2696

Master_Data[Master_Data.UserID == 2696].groupby('Rating')['MovieID'].count()

user2696 = Master_Data[Master_Data.UserID == 2696]
user2696

user_id = Master_Data[Master_Data.UserID == 2696].groupby('Rating')['MovieID'].count().plot(kind = 'pie',figsize = (6,5))
plt.title('Movie Ratings of UserID 2696')
plt.show

#Perform machine learning on first 500 extracted records
ml_Data = Master_Data.head(500)
ml_Data

ml_Data.shape

ml_Data.describe()

ml_Data['Age'].unique()#using age
ml_Data['Occupation'].unique()#using occupation
ml_Data.head()

f = ml_Data.iloc[:,[5,2,3]]
f.head(2)

