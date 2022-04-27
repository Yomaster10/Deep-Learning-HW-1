r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. **FALSE**, the test set allows us to estimate the generalization error.
The in-sample error can be estimated using the training set
2. **FALSE**, the goal of the training-set is to be large eanough to represent the underlying
joint distribution of data and labels. The test-set also has to be large eanough to get a good
estimate of the generalization error. Thus not every split would constitute an equally useful train-test split.
3. **TRUE**, using the test-set in cross validation could make us choose the model whose hyper-parameters
fit the test-set the best and cause the estimation of the generalization error to be irrelevant.
4. **FALSE**, we use it to estimate the in-sample loss to choose the best hyper-parameters.


"""

part1_q2 = r"""
**not justified**, if the hyper-parameters are tuned using the test-set,
We don't have a unbiased estimation of the generalization error.


"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Increasing K does not necessarily improve generalization** for extremely high values of k we basically
get a majority class classifier. for extremely small values of k the model is very sensitive to noise.

"""

part2_q2 = r"""
1. In the case of train-set accuracy we might overfit the training data resulting in a large generalization error.
2. In the case of test-set accuracy if we choose the model that best fits the test-set, we can say very little about the generalization error of the model.


"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
The choice is arbitrary for the SVM loss because the derivative of a constant is 0 so the gradient is not affected.

"""

part3_q2 = r"""
**Your answer:**
1. It seems that the classifier is learning the patterns of the gray scale values of the pixels of each digit.
The weights resemble an average of the trining data for each image.
We can see for example that the weights for the digits 2 and 7 are pretty similar and that explains some of the confusion.

2.  In both cases, it seems that the classifier is measuring some kind of "distance".
    In the case of KNN the classifier measures the euclidian distance from the training examples and chooses the majority class of the closest neighbors.
    In the case of MC-SVM the it seems that in some sense the classifier measures the "distance"
    from the learned pattern for each digit and classifies for the most similar pattern.
    The classifier are similar in that sense.

"""

part3_q3 = r"""
1. **GOOD**
a. In the case that the learning rate is too low, the slope of the loss graph is smaller and the graph is not as steep.
    It means that it would take more epochs to reach the same loss level and we might not get to saturation of the loss graph in a reasonable number of epochs.
b. In the case that the learning rate is too high there are local spikes in the loss graph when we over-shoot the local minimum of
   the loss function
2. Based on the accuracy graph we think that the model is slightly overfitted to the training set.
   At the end of the training the accuracy of the model on the training-set is slightly better than on the test set, buy all
   along the training we see pretty similar accuracy levels on the training and test set so we think that the model is not highly overfitted
   or highly underfitted to the training set.


"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
