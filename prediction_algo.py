import pandas as pd

df = pd.read_csv("train.csv")

#testing hypothesis (set default to death) using method loc (indexing survival and condition)
#if sex is female she will survive
df["Hyp"] = 0
df.loc[df.Sex == "female", "Hyp"] = 1

#if hypothesis is same as survived column update result in 1
df["Result"] = 0
df.loc[df.Survived == df["Hyp"], "Result"] = 1

#normalize to see percent
print df["Result"].value_counts(normalize=True)
