#file paths. Imported Sklearn through pip in terminal
import pandas as pd
import Utils as utils
from sklearn import tree, model_selection

df = pd.read_csv("train.csv")
utils.clean_data(df)

#target is desired output
#passed the desired features through the target
target = df["Survived"].values
feature_names = ["Pclass", "Age", "Fare", "Sex", "SibSp", "Parch"]
features = df[feature_names].values

#Classify the the data with decision tree
decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree_ = decision_tree.fit(features, target)

print decision_tree_.score(features, target)

score = model_selection.cross_val_score(decision_tree, features, target, scoring ='accuracy', cv=50)

print score
print score.mean()

#allowing the tree to be large and generalized
generalized_decision_tree = tree.DecisionTreeClassifier(
     random_state = 1,
     max_depth = 7,
     min_samples_split = 2)

generalized_decision_tree = tree.DecisionTreeClassifier(random_state = 1)
generalized_decision_tree_ = decision_tree.fit(features, target)

print generalized_decision_tree_.score(features, target)

#building a picture of the how the algorithm makes decision_tree
#imported graphviz through pip in terminal
tree.export_graphviz(generalized_decision_tree_, feature_names=feature_names, out_file="tree.dot")
