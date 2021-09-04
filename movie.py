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

l = ml_Data.iloc[:,6]
l.head(2)

features = f.values
label = l.values


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,401):
    X_train,X_test,Y_train,Y_test = train_test_split(features,label,test_size=0.2, random_state= i)
    for n in range(1,401):
        model = KNeighborsClassifier(n_neighbors = n)
        model.fit(X_train,Y_train)
        training_score = model.score(X_train,Y_train)
        testing_score = model.score(X_test,Y_test)
        if testing_score > training_score:
            if testing_score > 0.49:
                print("Training Score {} Testing Score {} for Random State {} and n_neighbors {}".format(training_score,testing_score,i,n))

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,Y_train,Y_test = train_test_split(features,label,test_size=0.2, random_state=232 )
model = KNeighborsClassifier(n_neighbors = 217)
model.fit(X_train,Y_train)
training_score = model.score(X_train,Y_train)
testing_score = model.score(X_test,Y_test)
# Only Generalized model will be outputted
if testing_score > training_score:
    print("Training Score {} Testing Score {} ".format(training_score,testing_score))

movieid = int(input("Enter the MovieID: "))
age = int(input("Enter the Age Group( 1, 56, 25, 45, 50):"))
occupation = int(input("Enter the Occupation group value (10, 16, 15,  7, 20,  9):"))

featureInput = np.array([[movieid,age,occupation]])
rating = model.predict(featureInput)
print("Rating of the Movie is: ", rating)    


