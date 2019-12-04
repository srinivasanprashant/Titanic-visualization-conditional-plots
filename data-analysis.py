import pandas as pd
import seaborn as sns    # seaborn is commonly imported as `sns`
import matplotlib.pyplot as plt


titanic = pd.read_csv("train.csv")
cols_to_keep = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Select columns with data to analyze and remove rows containing missing values
# Row count drops from 891 to 712 when inspected with .describe()
titanic = titanic[cols_to_keep].dropna()
# print(titanic.describe())
print(titanic.head(5))

# Generate a histogram of a column using the seaborn.distplot()
# Seaborn uses a technique called kernel density estimation, or KDE
# This creates a smoothed line chart over the histogram
# sns.distplot(titanic["Age"])

# Set up the style for clean looking plots to focus on data
sns.set_style("white")
# Generate just the kernel density plot using the seaborn.kdeplot()
# sns.kdeplot(titanic["Age"], shade=True)
# sns.despine(left=True, bottom=True)
# Set the x-axis label to "Age"
# plt.xlabel("Age")

# Visualize the differences in age distributions between passengers who survived and those who didn't
# by creating a pair of kernel density plots

# # Condition on unique values of the "Survived" column.
# g = sns.FacetGrid(titanic, col="Survived", height=5)
# # For each subset of values, generate a kernel density plot of the "Age" columns.
# g.map(sns.kdeplot, "Age", shade=True)

# # Use a FacetGrid instance to generate three plots on the same row
# g = sns.FacetGrid(titanic, col="Pclass", height=5)
# # For each subset of values, generate a kernel density plot of the "Age" columns.
# g.map(sns.kdeplot, "Age", shade=True)
# # Remove all of the spines
# sns.despine(left=True, bottom=True)

# titanic["Sex_female"] = (titanic["Sex"] == "female")
# print(titanic.head(5))
print(titanic["Age"].describe())

# Subsets the dataframe on different combinations of unique values in both the Pclass and Survived columns
# Plot the Sex column using different hues
g = sns.FacetGrid(titanic, col="Survived", row="Pclass", hue="Sex", height=3)
g.map(sns.kdeplot, "Age", shade=True).add_legend()
sns.despine(left=True, bottom=True)
# Clearly, the several factors seem to be correlated with whether one survived or not during the Titanic's accident
# Younger women and children seemed to have the most correlation with survival
plt.show()
