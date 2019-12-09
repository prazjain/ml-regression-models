# Regression Models in Machine Learning

## Concepts

#### Mean 
`∑ X(i) / N`

Mean is Average value of data points. But Mean does not show how the values are spread in dataset. For this we have Range.

#### Range 
`X(max) - X(min)`

This shows how wide apart the values are in dataset,
 but this is heavily dependent on min and max values in dataset. For this we have Variance.

#### Variance 
`∑ ( X(i) - X(mean) )^2 / N`

Variance shows how the values in dataset jump around the mean value. As variance does square of difference, we need square root of variance, which is Standard Deviation. 

#### Standard Deviation 
`SquareRoot (Variance)`

Standard Deviation is also a measure of how values jump around in dataset.

#### Standardize the dataset
So to standardize values in a matrix like below, we need to centre the value around 0, and reduce the variance to 1 : 

<code>
[

[x<sub>11</sub>, | x<sub>12</sub>, | x<sub>13</sub>]

[x<sub>21</sub>, | x<sub>22</sub>, | x<sub>23</sub>]

[x<sub>31</sub>, | x<sub>32</sub>, | x<sub>33</sub>]

]
</code>

(When represending data in matrix, data in a column will belong to a feature)
We can centre the dataset values around 0 by substracting the mean value of the column, from that column's data. And to reduce variance to 1, we can divide this value by standard deviation for that feature (column data)

<code>
[

[(x<sub>11</sub> - mean(x<sub>i1</sub>)) / std_dev(x<sub>i1</sub>), (x<sub>12</sub> - mean(x<sub>i2</sub>)) / std_dev(x<sub>i2</sub>), (x<sub>13</sub> - mean(x<sub>i3</sub>)) / std_dev(x<sub>i3</sub>)]

[(x<sub>21</sub> - mean(x<sub>i1</sub>)) / std_dev(x<sub>i1</sub>), (x<sub>22</sub> - mean(x<sub>i2</sub>)) / std_dev(x<sub>i2</sub>), (x<sub>23</sub> - mean(x<sub>i3</sub>)) / std_dev(x<sub>i3</sub>)]

[(x<sub>31</sub> - mean(x<sub>i1</sub>)) / std_dev(x<sub>i1</sub>), (x<sub>32</sub> - mean(x<sub>i2</sub>)) / std_dev(x<sub>i2</sub>), (x<sub>33</sub> - mean(x<sub>i3</sub>)) / std_dev(x<sub>i3</sub>)]

]
</code>

## Regression Models

#### Linear Regression

In Linear Regression, given a set of data points, it finds a line, that will pass through all the data points : 

`y = A + Bx`

(A is intersect on y axis. B is slope of line)

Because all the points ( (x<sub>1</sub>,y<sub>1</sub>), (x<sub>2</sub>,y<sub>2</sub>)...(x<sub>n</sub>,y<sub>n</sub>)), may not lie on same line, we will need to fit these points ( (x<sub>1</sub>,y'<sub>1</sub>), (x<sub>2</sub>,y'<sub>2</sub>)...(x<sub>n</sub>,y'<sub>n</sub>)) (f stands for fitted), on a line. So 

Error / Residual for a point will be = |y<sub>i</sub>| - |y'<sub>i</sub>|

*Residual of Regression* is difference in value of dependent variable (y) (label in Machine Learning terminology) for actual data point and fitted data point.

While fitting these points on the line, how do we know if we have a good fit?

This is checked with *Mean Square Error (MSE)* / *Variance of Residual* of data points

∑ ( (|y<sub>i</sub>| - |y'<sub>i</sub>|) <sup>2</sup>) / n

R<sup>2</sup> = Variance of fitted dependent variable / Variance of actual dependent variable

Higher R<sup>2</sup> value means better the line fits and variance of fitted values is more in line with variance of actual values. Upper bound value for R<sup>2</sup> is 100%.

#### Lasso Regression

To minimize large coefficients (weights on features). Objective function is in L1 form.

√MSE + α( |A| + |B| )


#### Ridge Regression

To minimize large coefficients (weights on features). Objective function is in L2 form.

√MSE + α( |A|<sup>2</sup> + |B|<sup>2</sup> )

Here α is the *hyper parameter* that can be tuned to find best fit for our model & data.

#### Regularization

Regularization is needed to reduce variance error. But this in turn introduces Bias.

|Low Bias|High Bias|
|---|---|
|Line is fitted closely to data| Line is not overfitted with training data|
|Not much impact of model parameter changes|Model parameter changes have impact|

|High Variance|Low Variance|
|---|---|
|Complex line, as we are not making any assumption about the data |Relatively Simple line by making assumption about data we have|
|Any changes to training data, could result in dramatically different model|Model is not tightly fitted to training data|

#### Gradient Boosting Regressor

In this model, machine generates lot of weak learning decision trees. (Keeping low value of max_depth hyperparameter, or keeping decision tree depth low)


You can see how Machine Learning Model Comparision fares for a particular data set with different hyper parameter values.