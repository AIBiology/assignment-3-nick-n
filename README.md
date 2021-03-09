# Assignment 3: Machine learning with regression methods


You may work independently or with **1** other partner. If you work with a partner, you should work together on all parts rather than splitting up the work. If you *really, really* want to work with a group of 3, let Matt or Brian know and we can discuss.

## Problem

The file [data/housing_data.csv](data/housing_data.csv) contains a dataset with 100 observations of 37 variables describing various attributes of houses.  Your goal is to build a model that can predict the price of a home (provided in the variable "SalePrice") using any of the information in the remaining 36 variables.  Descriptions of each variable are provided in [data/variable_descriptions.md](data/variable_descriptions.md).


## Turning in your work

Your final results should be submitted as a _single Jupyter notebook file_ named `assignment_3-FIRSTNAME_LASTNAME.ipynb`. If you are working with a partner, the file should be named `assignment_3-FIRSTNAME_LASTNAME-FIRSTNAME_LASTNAME.ipynb`.  This notebook should be:
  * well organized and
  * clearly documented: show your work, and explain your modeling decisions.

_**Very important:**_ The _final_ cell of your notebook should contain all of the code needed to build your final model, including Python imports, data loading, and modeling.  Specifically, it should have:
  * All required imports.
  * A function called `get_x_matrix()` that takes the full dataset and returns the x matrix to be used for building the model.  This makes your code much easier to test if you do things like exclude certain variables or apply custom data transformations "by hand".
  * Code to build your final model and save it to a variable named `model`.

Here is an example of what this final cell might look like:
```python
import pandas as pd
#
# Other required imports go here.
#

def get_x_matrix(raw_data):
    """
    Takes a raw DataFrame of a housing prices dataset and generates the x
    DataFrame (matrix) required for modeling.
    """
    # At a minimum, just remove the y variable, but you might want to do other
    # things, too (e.g., drop other variables or add custom transformations or
    # interactions.
    x_mat = raw_data.drop(columns=['SalePrice'])

    return x_mat


df = pd.read_csv('data/housing_data.csv')

x = get_x_matrix(df)
y = df.SalePrice

model = # Code to fit the model goes here.

```

Turn in your final notebook file by uploading it as your assignment submission on Canvas.  Please also paste the URL of your assignment repository on Github in the "comments" when you submit your assignment.


## Evaluation

The performance of your final model will be evaluated on a large test dataset from the same data source used to generate the training data.  Two evaluation criteria will be used:
  * bias [= mean(y_hat - y)]
  * mean absolute error [= mean(abs(y_hat - y))]

We will compare and discuss the (anonymized) test performance of all models in class.

Your grade will be determined as follows:
  * Notebook named and formatted as described above, with self-contained modeling code in the final cell (5 points).
  * Code that is syntactically correct (i.e., the code runs) (5 points).
  * Well-organized notebook that is clearly documented and easy to read (10 points).
  * Thoughtful, clearly explained modeling process (10 points).


## Tips

As always, to build a good model, you need to do more than just dump the data into a modeling procedure or mechanically repeat the code you learned in class (or on the web).  It is important to:
  1. Understand the data. Explore the dataset visually: look at variable distributions and how variables might be related to each other. Think about each variable and how it might make sense to include it in a model (if at all).
  2. Understand how the modeling methods you use work. For example, the dataset includes nominal, ordinal, and continuous variables; would it make sense to include all of these in PolynomialFeatures? What are the prediction error tradeoffs of the various modeling procedures? Under what circumstances does one approach work better than another?
  3. Assess the performance of your model(s) on data _outside_ of the training dataset. This is absolutely essential for building good statistical and machine learning models. We've discussed several strategies for doing this. Point 2 is important here, as well: understanding how these strategies work and the tradeoffs they involve is important for making wise modeling decisions.
  4. Sometimes it is useful to apply a scikit-learn transformation (e.g., `StandardScaler`) to only _part_ of a dataset. For that, [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) is very helpful and usually much easier than trying to manually slice and dice the dataframe.

