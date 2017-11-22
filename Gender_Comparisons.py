import pandas as pd
import matplotlib.pyplot as plt

female_color = "#ff1493"
male_color = "#228B22"

df = pd.read_csv("train.csv")
fig = plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survival Rate")
plt.xlabel('1 = Survived, 0 = Died')
plt.ylabel('Percent of Passengers')

plt.subplot2grid((3,4),(0,1))
df.Survived[df.Sex == "male"].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=male_color)
plt.title("Men Survived")
plt.xlabel('1 = Survived, 0 = Died')
plt.ylabel('Percent of Male Passengers')

plt.subplot2grid((3,4),(0,2))
df.Survived[df.Sex == "female"].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=female_color)
plt.title("Women Survived")
plt.xlabel('1 = Survived, 0 = Died')
plt.ylabel('Percent of Female Passengers')

plt.subplot2grid((3,4),(0,3))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=[female_color, male_color])
plt.title("Sex of Survivers")
plt.ylabel('Percent of Passengers')

plt.subplot2grid((3,4),(2,0))
df.Survived[(df.Sex == "male") & (df.Pclass ==1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=male_color)
plt.title("Survival of Men in First Class")
plt.xlabel('1 = Survived, 0 = Died')

plt.subplot2grid((3,4),(2,1))
df.Survived[(df.Sex == "male") & (df.Pclass ==3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=male_color)
plt.title("Survival of Men in Third Class")
plt.xlabel('1 = Survived, 0 = Died')

plt.subplot2grid((3,4),(2,2))
df.Survived[(df.Sex == "female") & (df.Pclass ==1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=female_color)
plt.title("Survival of Women in First Class")
plt.xlabel('1 = Survived, 0 = Died')

plt.subplot2grid((3,4),(2,3))
df.Survived[(df.Sex == "female") & (df.Pclass ==3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=female_color)
plt.title("Survival of Women in Third Class")
plt.xlabel('1 = Survived, 0 = Died')

plt.show()
