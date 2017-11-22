import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
fig = plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,0))
df.Embarked[df.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5,)
plt.title("Survival by Location")
plt.ylabel('Percent of Passengers')
plt.xlabel('|Southhampton | Cherbourg | Queenstown|')

plt.subplot2grid((2,3),(0,1))
df.Pclass[df.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5,)
plt.title("Survival by Passenger Class")
plt.ylabel('Percent of Passengers')

#Exammined survival rate bu class and by age. For loops to filter by age and survival.
plt.subplot2grid((2,3),(1,0), colspan=4)
for x in [1,2,3]:
  df.Survived[df.Pclass == x].plot(kind="kde")
plt.title("Survival Rate by Class")
plt.legend(("1st Class", "2nd Class", "3rd Class"))
plt.xlabel('0=Died, 1=Survived')

plt.subplot2grid((2,3),(0,2), colspan=2)
for x in [0,1,]:
  df.Age[df.Survived == x].plot(kind="kde")
plt.title("Survival by Age")
plt.legend(("1 = Survived", "0 = Dead"))

plt.show()
