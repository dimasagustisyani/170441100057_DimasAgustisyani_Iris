**Data Description**

Iris plants dataset is a well-known benchmark commonly used to test supervised machine learning problems. The four first attributes are numerical attributes, which gives measurements in centimeters of the variables sepal length and width, and petal length and width, respectively, for 50 flowers from three different species of iris. They will be used for the clustering and as predictive attributes, but the class (the fifth attribute) can be used to test the validity of the clustering.

In [1]:

```
# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#system("ls ../input")
names(iris)
data<- iris[1:4] # four attribute('Sepal.Length' 'Sepal.Width' 'Petal.Length' 'Petal.Width')
class<-as.matrix(iris[5]) # the class ('Species')
```



1. 'Sepal.Length'
2.  

3. 'Sepal.Width'
4. 'Petal.Length'
5.  

6. 'Petal.Width'
7. 'Species'

for more details:

In [2]:

```
summary (iris)
help (iris)
```



```
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
       Species  
 setosa    :50  
 versicolor:50  
 virginica :50  
                
                
                
```

We can see what are the most correlated attributes:

In [3]:

```
cor(data)
```



|              | Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |
| :----------- | :----------- | :---------- | :----------- | :---------- |
| Sepal.Length | 1.0000000    | -0.1175698  | 0.8717538    | 0.8179411   |
| Sepal.Width  | -0.1175698   | 1.0000000   | -0.4284401   | -0.3661259  |
| Petal.Length | 0.8717538    | -0.4284401  | 1.0000000    | 0.9628654   |
| Petal.Width  | 0.8179411    | -0.3661259  | 0.9628654    | 1.0000000   |

As the correlation matrix shows the most correlated attributes are Petal.Length and Petal.Width (96%)

**Visualisation**

Ploting the pairwise graph of the iris dataset with the 4 first continuous attributes to study the effect of each attribute on the species of the iris:

In [4]:

```
pairs(iris[1:4], pch=21, bg= c("red","green3","blue")[unclass(iris$Species)])
```



![img](assets/images/1.png)

**Principal Component Analysis**

Using the **prcomp** function for computing the principal component analysis (PCA) of the dataset

In [5]:

```
iris.pca<- prcomp(iris[1:4])
iris.pca
```



```
Standard deviations (1, .., p=4):
[1] 2.0562689 0.4926162 0.2796596 0.1543862

Rotation (n x k) = (4 x 4):
                     PC1         PC2         PC3        PC4
Sepal.Length  0.36138659 -0.65658877  0.58202985  0.3154872
Sepal.Width  -0.08452251 -0.73016143 -0.59791083 -0.3197231
Petal.Length  0.85667061  0.17337266 -0.07623608 -0.4798390
Petal.Width   0.35828920  0.07548102 -0.54583143  0.7536574
```

The attribute that is the best represented by the first component (PC1) is Petal.Length (0.86 on the first component PC1)

In [6]:

```
summary(iris.pca)
```



```
Importance of components:
                          PC1     PC2    PC3     PC4
Standard deviation     2.0563 0.49262 0.2797 0.15439
Proportion of Variance 0.9246 0.05307 0.0171 0.00521
Cumulative Proportion  0.9246 0.97769 0.9948 1.00000
```

The reduced representation space (in 2D) is a good representation of the four attributes of the dataset: the cumulative proportion with the first component (1D) represents 92% of the dataset variance and with the two components (2D) represents 98% of the datset variance.

In [7]:

```
pairs(iris.pca$x, main = "PCA PLOT", font.main = 4, pch = 21,bg= c("red","green3","blue")[unclass(iris$Species)])
plot(iris.pca$x, main = "PCA PLOT", font.main = 4, pch = 21, bg= c("red","green3","blue")[unclass(iris$Species)])
```



![img](assets/images/2.png)



![img](assets/images/3.png)

The 3 iris species are well separated on a 2D representation, especially with the first component (on x-axis)

**PCA with princomp function**

In [8]:

```
iris.pca2 <- princomp(iris[1:4])
iris.pca2
biplot(iris.pca2)
```



```
Call:
princomp(x = iris[1:4])

Standard deviations:
   Comp.1    Comp.2    Comp.3    Comp.4 
2.0494032 0.4909714 0.2787259 0.1538707 

 4  variables and  150 observations.
```



![img](assets/images/4.png)

The attribute which is the best represented in the plot is Petal.Length (it is the largest arrow), this arrow is nearly parallel to x-axis and covers the Petal.Width arrow (due to the high correlation between the 2 attributes). We can easily find 2 groups in iris dataset The class "setosa" is easily separated from the two others (the cluster composed by "versicolor" and "virginica" classes) We expect that the results obtained by applying clustering and classification methods will not be bad, especially for predicting the iris from setosa class.

**PART2**

**CLUSTERING**

- Data Preparation

In [9]:

```
irisn <- t(apply(iris[1:4], 1, as.numeric))
normalize<- function(row){
  (row - mean(row))/ sd(row)
}
iris_zs <- apply(irisn,2,normalize)
```

**K-Means**

In [10]:

```
#For obtaining a k-means clustering with k = 3 clusters on the third and fifth attributes
c1 <- kmeans(iris[1:4], 3)
print(c1)
plot(iris[3:4], col = c1$cluster)
```



```
K-means clustering with 3 clusters of sizes 50, 62, 38

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.006000    3.428000     1.462000    0.246000
2     5.901613    2.748387     4.393548    1.433871
3     6.850000    3.073684     5.742105    2.071053

Clustering vector:
  [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [38] 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [75] 2 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 3 3 3 3 2 3 3 3 3
[112] 3 3 2 2 3 3 3 3 2 3 2 3 2 3 3 2 2 3 3 3 3 3 2 3 3 3 3 2 3 3 3 2 3 3 3 2 3
[149] 3 2

Within cluster sum of squares by cluster:
[1] 15.15100 39.82097 23.87947
 (between_SS / total_SS =  88.4 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"      
```



![img](assets/images/5.png)

With the z-score transformation, the results seem to be less stable. This is due to the effect of the two first attributes (with sepals) which can add more noise on the distances computed between the points.

**PART3**

**Classification**

**Decision Tree**

The decision tree can be computed on a dataset after loading the **rpart** package.

In [11]:

```
library(rpart)
library("rpart.plot")
"dividing the dataset in two random samples: one for training the model and the other one for testing the
model."
set.seed(2586)
n <- nrow(iris)
train <- sort(sample(1:n, floor(n/2)))
iris.train <- iris[train, ]
iris.test <- iris[-train, ]
```



'dividing the dataset in two random samples: one for training the model and the other one for testing the\nmodel.'

There are many packages in R for modeling decision trees: rpart, party, RWeka, ipred, randomForest, gbm, C50. The R package **rpart** implements recursive partitioning. As iris dataset contains 3 classes of 150 instances each, where each class refers to the type of the iris plant. We'll try to find a tree, which can tell us if an Iris flower species belongs to one of following classes: setosa, versicolor or virginica. As the response variable is categorial, the resulting tree is called classification tree. The default criterion, which is maximized in each split is the Gini index.

In [12]:

```
library(rpart)
library(rpart.plot)
tree <- rpart(Species ~., data = iris.train, method = "class")
tree
rpart.plot(tree)
```



```
n= 75 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

1) root 75 49 virginica (0.3333333 0.3200000 0.3466667)  
  2) Petal.Length< 2.6 25  0 setosa (1.0000000 0.0000000 0.0000000) *
  3) Petal.Length>=2.6 50 24 virginica (0.0000000 0.4800000 0.5200000)  
    6) Petal.Length< 5.05 27  3 versicolor (0.0000000 0.8888889 0.1111111) *
    7) Petal.Length>=5.05 23  0 virginica (0.0000000 0.0000000 1.0000000) *
```



![img](assets/images/6.png)

Evaluation of the Decision Tree on the Test Set

In [13]:

```
pred.rep <- predict(tree, newdata = iris[-train,], type = "class" )
#pred.rep
```

For having more information:

" predict(tree, newdata = iris[-train,], type = "prob") predict(tree, newdata = iris[-train,], type = "vector") predict(tree, newdata = iris[-train,], type = "matrix") "

In [14]:

```
table(class[-train], pred.rep)
```



```
            pred.rep
             setosa versicolor virginica
  setosa         25          0         0
  versicolor      0         25         1
  virginica       0          6        18
```

The cross table shows a few errors (e.g., 2 examples are predicted virginica but are versicolor), the setosa class is predicted without error

k-Nearest Neighbours (k-NN):

It is possible to use k-NN algorithm with knn function from class package. Here, we will use the 25 first elements for the training set and the next 25 for the test test.

In [15]:

```
library(class)
data(iris3)
str(iris3)
```



```
 num [1:50, 1:4, 1:3] 5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
 - attr(*, "dimnames")=List of 3
  ..$ : NULL
  ..$ : chr [1:4] "Sepal L." "Sepal W." "Petal L." "Petal W."
  ..$ : chr [1:3] "Setosa" "Versicolor" "Virginica"
```

In [16]:

```
train <- rbind(iris3[1:25,,1],iris3[1:25,,2],iris3[1:25,,3])
test <-  rbind(iris3[26:50,,1],iris3[26:50,,2],iris3[26:50,,3])
cl <- factor (c(rep("Setosa",25), rep("Versicolor",25), rep("Virginica",25)))
pred <- knn(train, test, cl, k=3)
table(pred, cl)
```



```
            cl
pred         Setosa Versicolor Virginica
  Setosa         25          0         0
  Versicolor      0         23         4
  Virginica       0          2        21
```