#Imported pandas (package providing fast, flexible, and expressive data structures)
#library and matlab through pip in terminal file path

import pandas as pd
import matplotlib.pyplot as plt

 # Daraframe - Training file
df = pd.read_csv("train.csv")
fig = plt.figure(figsize=(18,6))

#subplot2grid allows for flexible display
#normalize to see values in percent
#plotted survival rate
plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survival Rate")
plt.xlabel('0 = Dead  1 = Survived')
plt.ylabel('Percent of Passengers')

#examined survival rate by age in scatter
plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title("Survival Rate by Age")
plt.xlabel('Percent of Passengers')
plt.ylabel('Age')

#examined class level
plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Class Distribution")
plt.xlabel('Class Level')
plt.ylabel('Percent of Passengers')

#examined age in regard to class using  kernel density estimation
#used a four loop to filter by age, passenger class
plt.subplot2grid((2,3),(1,0), colspan=2)
for x in [1,2,3]:
  df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Class by Age")
plt.legend(("1st Class", "2nd Class", "3rd Class"))

#examined where people embarked
plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Embarked")
plt.xlabel('|Southhampton | Cherbourg | Queenstown|')
plt.ylabel('Percent of Passengers')


plt.show()
