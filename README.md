# CM146-Problem-Set-2-Perceptron-and-regression-solved

Download Here: [CM146 Problem Set 2: Perceptron and regression solved](https://jarviscodinghub.com/assignment/problem-set-2-perceptron-and-regression-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

1 Perceptron [2 pts]
Design (specify θ for) a two-input perceptron (with an additional bias or offset term) that computes
the following boolean functions. Assume T = 1 and F = −1. If a valid perceptron exists, show
that it is not unique by designing another valid perceptron (with a different hyperplane, not simply
through normalization). If no perceptron exists, state why.
(a) OR (b) XOR
2 Logistic Regression [10 pts]
Consider the objective function that we minimize in logistic regression:
J(θ) = −
X
N
n=1
[yn log hθ (xn) + (1 − yn) log (1 − hθ (xn))]
(a) Find the partial derivatives ∂J
∂θj
.
(b) Find the partial second derivatives ∂
2J
∂θj∂θk
and show that the Hessian (the matrix H of second
derivatives with elements Hjk =
∂
2J
∂θj∂θk
) can be written as H =
PN
n=1 hθ (xn) (1 − hθ (xn)) xnx
T
n
.
(c) Show that J is a convex function and therefore has no local minima other than the global
one.
Hint: A function J is convex if its Hessian is positive semi-definite (PSD), written H  0. A
matrix is PSD if and only if
z
THz ≡
X
j,k
zjzkHjk ≥ 0.
for all real vectors z.
2
3 Locally Weighted Linear Regression [14 pts]
Consider a linear regression problem in which we want to “weight” different training instances
differently because some of the instances are more important than others. Specifically, suppose we
want to minimize
J(θ0, θ1) = X
N
n=1
wn (θ0 + θ1xn,1 − yn)
2
.
Here wn > 0. In class, we worked out what happens for the case where all the weights (the wn’s)
are the same. In this problem, we will generalize some of those ideas to the weighted setting.
(a) Calculate the gradient by computing the partial derivatives of J with respect to each of the
parameters (θ0, θ1).
(b) Set each partial derivatives to 0 and solve for θ0 and θ1 to obtain values of (θ0, θ1) that
minimize J.
(c) Show that J(θ) can also be written
J(θ) = (Xθ − y)
TW(Xθ − y)
for an appropriate diagonal matrix W, and where X =


1 x1,1
1 x2,1
.
.
.
.
.
.
1 xN,1


and y =


y1
y2
.
.
.
yN


and
θ =

θ0
θ1

. State clearly what W is.
3
4 Implementation: Polynomial Regression [20 pts]
In this exercise, you will work through linear and polynomial regression. Our data consists of
inputs xn ∈ R and outputs yn ∈ R, n ∈ {1, . . . , N}, which are related through a target function
y = f(x). Your goal is to learn a linear predictor hθ(x) that best approximates f(x). But this time,
rather than using scikit-learn, we will further open the “black-box”, and you will implement the
regression model!
code and data
• code : regression.py
• data : regression_train.csv, regression_test.csv
This is likely the first time that many of you are working with numpy and matrix operations within
a programming environment. For the uninitiated, you may find it useful to work through a numpy
tutorial first.1 Here are some things to keep in mind as you complete this problem:
• If you are seeing many errors at runtime, inspect your matrix operations to make sure that
you are adding and multiplying matrices of compatible dimensions. Printing the dimensions
of variables with the X.shape command will help you debug.
• When working with numpy arrays, remember that numpy interprets the * operator as elementwise multiplication. This is a common source of size incompatibility errors. If you want
matrix multiplication, you need to use the dot function in Python. For example, A*B does
element-wise multiplication while dot(A,B) does a matrix multiply.
• Be careful when handling numpy vectors (rank-1 arrays): the vector shapes 1 × N, N ×
1, and N are all different things. For these dimensions, we follow the the conventions of
scikit-learn’s LinearRegression class2
. Most importantly, unless otherwise indicated (in
the code documentation), both column and row vectors are rank-1 arrays of shape N, not
rank-2 arrays of shape N × 1 or shape 1 × N.
Visualization [1 pts]
As we learned last week, it is often useful to understand the data through visualizations. For this
data set, you can use a scatter plot to visualize the data since it has only two properties to plot (x
and y).
(a) Visualize the training and test data using the plot_data(…) function. What do you observe? For example, can you make an educated guess on the effectiveness of linear regression
in predicting the data?
1Try out SciPy’s tutorial (https://wiki.scipy.org/Tentative_NumPy_Tutorial), or use your favorite search engine to find an alternative. Those familiar with Matlab may find the “Numpy for Matlab Users” documentation
(https://wiki.scipy.org/NumPy_for_Matlab_Users) more helpful.
2
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
4
Linear Regression [12 pts]
Recall that linear regression attempts to minimize the objective function
J(θ) = X
N
n=1
(hθ(xn) − yn)
2
.
In this problem, we will use the matrix-vector form where
y =


y1
y2
.
.
.
yN


, X =


x
T
1
x
T
2
.
.
.
x
T
N


, θ =


θ0
θ1
θ2
.
.
.
θD


and each instance xn =

1, xn,1, . . . , xn,DT
.
In this instance, the number of input features D = 1.
Rather than working with this fully generalized, multivariate case, let us start by considering a
simple linear regression model:
hθ(x) = θ
Tx = θ0 + θ1×1
regression.py contains the skeleton code for the class PolynomialRegression. Objects of this
class can be instantiated as model = PolynomialRegression (m) where m is the degree of the
polynomial feature vector where the feature vector for instance n,

1, xn,1, x2
n,1
, . . . , xm
n,1
T
. Setting
m = 1 instantiates an object where the feature vector for instance n,

1, xn,1
T
.
(b) Note that to take into account the intercept term (θ0), we can add an additional “feature” to
each instance and set it to one, e.g. xi,0 = 1. This is equivalent to adding an additional first
column to X and setting it to all ones.
Modify PolynomialRegression.generate_polynomial_features(…) to create the matrix
X for a simple linear model.
(c) Before tackling the harder problem of training the regression model, complete
PolynomialRegression.predict(…) to predict y from X and θ.
(d) One way to solve linear regression is through gradient descent (GD).
Recall that the parameters of our model are the θj values. These are the values we will adjust
to minimize J(θ). In gradient descent, each iteration performs the update
θj ← θj − 2α
X
N
n=1
(hθ(xn) − yn) xn,j (simultaneously update θj for all j).
With each step of gradient descent, we expect our updated parameters θj to come closer to
the parameters that will achieve the lowest value of J(θ).
• As we perform gradient descent, it is helpful to monitor the convergence by computing
the cost, i.e., the value of the objective function J. Complete PolynomialRegression.cost(…)
to calculate J(θ).
If you have implemented everything correctly, then the following code snippet should
return 40.234.
train_data = load_data(‘regression_train.csv’)
model = PolynomialRegression()
model.coef_ = np.zeros(2)
model.cost(train_data.X, train_data.y)
• Next, implement the gradient descent step in PolynomialRegression.fit_GD(…).
The loop structure has been written for you, and you only need to supply the updates
to θ and the new predictions ˆy = hθ(x) within each iteration.
We will use the following specifications for the gradient descent algorithm:
– We run the algorithm for 10, 000 iterations.
– We terminate the algorithm ealier if the value of the objective function is unchanged
across consecutive iterations.
– We will use a fixed step size.
• So far, you have used a default learning rate (or step size) of η = 0.01. Try different
η = 10−4
, 10−3
, 10−2
, 0.0407, and make a table of the coefficients, number of iterations
until convergence (this number will be 10, 000 if the algorithm did not converge in a
smaller number of iterations) and the final value of the objective function. How do the
coefficients compare? How quickly does each algorithm converge?
(e) In class, we learned that the closed-form solution to linear regression is
θ = (XTX)
−1XT y.
Using this formula, you will get an exact solution in one calculation: there is no “loop until
convergence” like in gradient descent.
• Implement the closed-form solution PolynomialRegression.fit(…).
• What is the closed-form solution? How do the coefficients and the cost compare to those
obtained by GD? How quickly does the algorithm run compared to GD?
(f) Finally, set a learning rate η for GD that is a function of k (the number of iterations) (use
ηk =
1
1+k
) and converges to the same solution yielded by the closed-form optimization (minus
possible rounding errors). Update PolynomialRegression.fit_GD(…) with your proposed
learning rate. How long does it take the algorithm to converge with your proposed learning
rate?
Polynomial Regression[7 pts]
Now let us consider the more complicated case of polynomial regression, where our hypothesis is
hθ(x) = θ
T φ(x) = θ0 + θ1x + θ2x
2 + . . . + θ
mx
m.
6
(g) Recall that polynomial regression can be considered as an extension of linear regression in
which we replace our input matrix X with
Φ =


φ(x1)
T
φ(x2)
T
.
.
.
φ(xN )
T


,
where φ(x) is a function such that φj (x) = x
j
for j = 0, . . . , m.
Update PolynomialRegression.generate_polynomial_features(…) to create an m + 1
dimensional feature vector for each instance.
(h) Given N training instances, it is always possible to obtain a “perfect fit” (a fit in which all
the data points are exactly predicted) by setting the degree of the regression to N − 1. Of
course, we would expect such a fit to generalize poorly. In the remainder of this problem, you
will investigate the problem of overfitting as a function of the degree of the polynomial, m.
To measure overfitting, we will use the Root-Mean-Square (RMS) error, defined as
ERMS =
p
J(θ)/N,
where N is the number of instances.3
Why do you think we might prefer RMSE as a metric over J(θ)?
Implement PolynomialRegression.rms_error(…).
(i) For m = 0, . . . , 10, use the closed-form solver to determine the best-fit polynomial regression
model on the training data, and with this model, calculate the RMSE on both the training
data and the test data. Generate a plot depicting how RMSE varies with model complexity
(polynomial degree) – you should generate a single plot with both training and test error, and
include this plot in your writeup. Which degree polynomial would you say best fits the data?
Was there evidence of under/overfitting the data? Use your plot to justify your answer.
3Note that the RMSE as defined is a biased estimator. To obtain an unbiased estimator, we would have to divide
by n − k, where k is the number of parameters fitted (including the constant), so here, k = m + 1.
