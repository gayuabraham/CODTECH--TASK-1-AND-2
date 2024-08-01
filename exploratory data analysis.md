import pandas as pd
import numpy as np
df= pd.read_csv('/content/student_scores.csv')
df
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/student_scores.csv')

# Descriptive statistics
print(df.describe())

# Distribution of scores
sns.histplot(df['Scores'])
plt.title('Distribution of Scores')
plt.show()

# Correlation between hours and scores
sns.scatterplot(x='Hours', y='Scores', data=df)
plt.title('Correlation between Hours and Scores')
plt.show()

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Check for outliers using box plot
sns.boxplot(df['Hours'])
plt.title('Box Plot for Hours')
plt.show()

sns.boxplot(df['Scores'])
plt.title('Box Plot for Scores')
plt.show()
