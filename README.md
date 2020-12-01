# Machine Learning

This repo includes a collection of both supervised and unsupervised machine learning algorithms, all in python, and leveraging Open Source packages like [scikitlearn](https://scikit-learn.org/stable/index.html).  
The goal of this repo is to familiarize ourselves with these algorithms, applied to various data sets and use cases and see if there are good candidates to onboard into our projects or platform(s).

To note I am also capturing additional notebooks that leverage Amazon SageMaker and their own improved implementation of some of these algorithms in this [aws-sagemaker-notebooks](https://github.com/FabG/aws-sagemaker-notebooks) repo.

### 1. Installation & How to Run
It is highly recommended to utilize a virtual environment like [virtualenv](https://pypi.org/project/virtualenv/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) due to this project having several libraries to import.
Also please use [Python 3](https://www.python.org/download/releases/3.0/) (and not python 2.X)

Install the python packages (after setting up your virtual environment):
`pip3 install -r requirements.txt`


Finally, you will need [Jupyter](https://jupyter.org/) to run the notebooks:
 - Jupyter Lab install (recommended): `pip install jupyterlab`
 - Classic Jupyter Notebook install: `pip install notebook`

Once installed, run:
 - Jupyter Lab: `jupyter-lab`
 - or Jupyter Notebook: `jupyter notebook`


### 2. Collection of Notebooks showcasing different ML algorithms

#### 2.1 Supervised algorithms
##### 2.1.1 Naive Bayes
**Naive Bayes** models are a group of extremely **fast and simple classification algorithm that are often suitable for very high-dimensional datasets**. Because they are so fast and have so few tunable parameters, they end up being very useful as a quick-and-dirty **baseline for a classification problem**.

When to use Naive Bayes?
Because naive Bayesian classifiers make such stringent assumptions about data, they will generally not perform as well as a more complicated model. That said, they have several advantages:

- They are extremely fast for both training and prediction
- They provide straightforward probabilistic prediction
- They are often very easily interpretable
- They have very few (if any) tunable parameters

These advantages mean a naive Bayesian classifier is often a good choice as an initial baseline classification. If it performs suitably, then congratulations: you have a very fast, very interpretable classifier for your problem. If it does not perform well, then you can begin exploring more sophisticated models, with some baseline knowledge of how well they should perform.

Naive Bayes classifiers tend to perform especially well in one of the following situations:

- When the naive assumptions actually match the data (very rare in practice)
- For very well-separated categories, when model complexity is less important
- For very high-dimensional data, when model complexity is less important

###### Naive Bayes Notebooks
- [In Depth - Naive Bayes](supervised-ml/naive-bayes/naive-bayes.ipynb)
- [Naive Bayes on titanic Kaggle dataset](supervised-ml/naive-bayes/naive-bayes-titanic.ipynb)


##### 2.1.2 SVM (Support vector machines)
###### SVM Intro
[SVM](https://scikit-learn.org/stable/modules/svm.html) Support vector machines (SVMs) are a set of supervised learning methods used for **classification**, **regression** and **outliers detection**.

The **advantages** of support vector machines are:
- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The **disadvantages** of support vector machines include:
- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).


###### SVM Notebooks
- [In Depth - Support Vector Machine](supervised-ml/svm/support-vector-machine.ipynb)
- [SVM binary classifier on Cancer data](supervised-ml/svm/svm_classifier_breast_cancer/svm_classifier_cancer.ipynb)
- [SVM and other binary classifiers on Cancer Data](supervised-ml/svm/svm_classifier_breast_cancer/multiple_classifiers_cancer.ipynb) => SVM, Logistic Regression, KNeighbor, Naïve Bayes, Random Forest, Decision Tree
- [SVM on Credit Card Fraud data](supervised-ml/svm/svm_classifier_credit_card_fraud/svm_credit_card_fraud.ipynb)



##### 2.1.3 Random Forest
Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble.
It can be used for **classification**, **regression** and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees.
For example, in a regression problem, each decision tree in the forest considers a random subset of features when forming questions and only has access to a random set of the training data points. This increases diversity in the forest leading to more robust overall predictions and the name ‘random forest.’ When it comes time to make a prediction, the random forest takes an average of all the individual decision tree estimates.


###### RandomForest Notebooks
 - [RandomForest as a Multi-classifier Vs Decision Tree and GuassianNB on Wine data](supervised-ml/random-forest/random-forest-game-of-wines.ipynb)


##### 2.2 Supervised algorithms - comparing Classifiers
 - [Classifier Comparison plot](supervised-ml/classifiers/classifier_comparison_plot.ipynb)

Here is how classifiers compares based on 3 different synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries of different classifiers. This should be taken with a grain of salt, as the intuition conveyed by these examples does not necessarily carry over to real datasets
![classifier_comparison_plot](images/classifier_comparison_plot.png)



#### 2.3 Unsupervised algorithms

##### 2.3.1 PCA - Principal Component Analysis
PCA is fundamentally a dimensionality reduction algorithm, but it can also be useful as a tool for visualization, for noise filtering, for feature extraction and engineering, and much more.

###### PCA Notebooks
 - [Principal Component Analysis](unsupervised-ml/pca/principal-component-analysis.ipynb) => Dimension reduction, Visualization and Noise filtering

#### 3 Resources
 - [Kaggle game of wines](https://www.kaggle.com/booleanhunter/game-of-wines)
 - [In depth Principal Component Analysis](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
 - [Python Data Science handbook](https://github.com/jakevdp/PythonDataScienceHandbook)
