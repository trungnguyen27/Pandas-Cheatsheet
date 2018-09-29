# Machine learning cheat-sheet (basic)

**Sampling**

> Generate random samples and then break the data into tests and training sets

- 2/3 of your dataset -> training set
- 1/3 of your dataset -> test set

**Methods**

- Supervised: making prediction from labeled data
- Unsupervised: clustering data

**Factor analysis**

A method used to explore datasets to find root causes that explain why data is acting a certain way

_factors(laten vars)_ = vars that are quite meaningful but are inferred and not directly observables

> **Assumptions**
> - Features are metrics
> - Continuous or ordinal
> r > 0.3 correlation between the features in your dataset
> N > 100 observation and > 5 observation per feature
> Sample is homogenous

**Single value decomposition (SVD)**

A linear algebra method that decomposes a matrix into three resultant to reduce information redundancy and noise

commonly used for principal component analysis

> Anatomy of SVD
> - A = u * s * v (matrices)
> - A = original matrix
> - u = left orthogonal matrix; holds important, non redundant info about observations
> - v = right orthogonal matrix; holds important info on features
> - S = diagonal matrix; contains all of the info about decomposition processes during the compression 

**Principal components**

Uncorrelated features that embody a dataset's important information (its variance) with the redundancy, noise, and outliers _stripped out_

> Usage
> - Fraud detection
> - Spam detectioin
> - Image and speech recognition

## Discover latent variables with Scikit-learn

### Factor analysis

**Imports**

```py
import pandas as pd
import numpy as np

import sklearn
from sklearn.decomposition import FactorAnalysis

from sklearn import datasets
```

**Implementations**

```py
factor = FactorAnalysis().fit(X)

pd.DataFrame(factor.components_, columns=variable_names)
```

### Principle components

**imports**

```py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb
from IPython.display import Image
from IPython.core.display import HTML 
from pylab import rcParams

import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets
```

Sample data
```py
iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names
X[0:10,]
```

**Explained variance ratio**
```py
pca = decomposition.PCA()
iris_pca = pca.fit_transform(X)

pca.explained_variance_ratio_

# sum
pca.explained_variance_ratio_.sum()
```

**Why ?**
- Explained variance ratio tells how much infomation is compressed into the first few components
- Helpful when deciding how many components to keep, looking at the percent of cummulative variance


### Extreme Value analysis

**Set up**
```py
df = pd.read_csv(
    filepath_or_buffer='C:/Users/Lillian Pierson/Desktop/Exercise Files/Ch05/05_01/iris.data.csv',
    header=None,
    sep=',')
df.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width', 'Species']

X = df.ix[:,0:4].values
y = df.ix[:,4].values

df[:5]
```

#### Identifying outliers from Tukey boxplots

**Unvariate method**

```py
df.boxplot(return_type='dict')
plt.plota()

# from the boxplot, identify the quartile and normal ranges, then filter out potential outliers
# in this case, Sepal_width > 4 seems to be an outlier

Sepal_Width = X[:,1]
iris_outliers = (Sepal_Width > 4)
df[iris_outliers]

## apply Tukey outlier labeling

pd.options.display.float_format = '{:.1f}'.format
X_df = pd.DataFrame(X)
print X_df.describe()
```

**Variate method**

```py
sb.boxplot(x='Species', y='Sepal Length', data=df, palette='hls')

sb.pairplot(df, hue='Species', palette='hls')
```

#### Linear project methods for multi-variate data analysis

**DBSCAN for outlier detection**

Important DBSCAN model parameters
    - esps: the maximum distancec between two samples for them to be clustered in the same neighborhood (stat at esp = 0.1)
    - min_samples: the minimum number of samples in a neighborhood for a data point to qualify as a core point(stsart with very low sample size)
   

**imports**
```py
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter


df.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width', 'Species']
data = df.ix[:,0:4].values
target = df.ix[:,4].values
df[:5]
```

**Implementation**

```py
model = DBSCAN(eps=0.8, min_samples=19).fit(data)
print model
```

**Visualization**

```py
outliers_df = pd.DataFrame(data)

print Counter(model.labels_)

print outliers_df[model.labels_ ==-1]

fig = plt.figure()
ax = fig.add_axes([.1, .1, 1, 1]) 

colors = model.labels_

ax.scatter(data[:,2], data[:,1], c=colors, s=120)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
plt.title('DBScan for Outlier Detection')
```


### K-mean clustering

**Base**
- Number of cluster center present (k)
- Nearest mean values (measure in Euclidian distance between observations) 
  
**Use cases**
- Market price and cost modelling
- Customer segmentation
- Insurance claim detection
- Hedge fund classification

**Keep in mind**
- Scale the variables
- Look at the scatterplot or the data table to estimate the number of centroid, cluster centers, to set for the k parameter for the model

**Set up**

```py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm # evaluate model
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report #use to evaluate the model

iris = dataset.load_iris()

X = scale(iris.data)
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names
```

**Implementation**

```py
clustering = KMeans(n_clusters = 3, random_state = 5)
clustering.fit(X)

# plot the model

iris_df = pf.DataFrame(iris.data)
irs_df.columns = []
y.columns= ["Targets"]

color_theme = np.array(["darkgray", "lightsalmon", "powderblue"])

# first subplot
plot.subplot(1,2,1) # 1 row, 2 columns, first position

plot.scatter(x=iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[iris.target], s=50) 
plt.title("Ground Truth Classification")

plt.subplot(1,2,2)
plot.scatter(x=iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[clustering.labels], s=50) 
plt.title("K-Means classification")

# Relabel
relabel = np.choose(clustering.labels_, [2,0,1].astype(np.int64))
plot.subplot(1,2,1) # 1 row, 2 columns, first position

plot.scatter(x=iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[iris.target], s=50) 
plt.title("Ground Truth Classification")

plt.subplot(1,2,2)
plot.scatter(x=iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[clustering.labels], s=50) 
plt.title("K-Means classification")
```

**Evaluation**

```py
print classification_report(y, relabel)
```

- **Presicion**: a measure of the model's relevancy
- **Recall**: a measure of the model's completeness


### Hierachical Clustering
Hierachical clustering predict subgroups within data by
- finding distance between each data point and its nearest neighbors
- linking the most nearby neighbors
