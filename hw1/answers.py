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
**Your answer:**

**Not justified**, if the hyper-parameters are tuned using the test-set,
We don't have a unbiased estimation of the generalization error.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

**Increasing K does not necessarily improve generalization** for extremely high values of k we basically
get a majority class classifier. for extremely small values of k the model is very sensitive to noise.
"""

part2_q2 = r"""
**Your answer:**

1. In the case of train-set accuracy we might overfit the training data resulting in a large generalization error.
2. In the case of test-set accuracy if we choose the model that best fits the test-set, we can say very little about the generalization error of the model.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

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
**Your answer:**

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

The ideal pattern to see in a residual plot is a horizontal line at y=0, with all the data points concentrated close to it. This means that our error is minimal (ideally zero). The final plot after CV is close to ideal, since most of the points are concentrated near the y=0 line, whereas the top-5 features plot was much more scattered and thus contained more error.
"""

part4_q2 = r"""
**Your answer:**

(1) **Yes**, this is still a linear regression model. While the data was pre-processed in a non-linear fashion, the model that we are using still belongs to the hypothesis class of linear regression. All we have done is added pre-processing to the data, we have not changed the type of model we are training.

(2) **Yes**, we can do whatever we please to the data during the pre-processing phase (to an extent, of course); this still will not change the class of model. What it will affect, though, is the performance of the model. If we choose these non-linear functions poorly, we could get a worse fit from our linear regression model.

(3) Adding non-linear features would not affect the decision boundary of such a classifier, in the sense that it would still remain a hyperplane. Our model is still linear, and our weights are still calculated as such - they still describe a linear boundary. In fact, the decision boundaries of non-linear SVM's are also hyperplanes (though only in the transformed feature space, not in the original feature space). The only effect is that the weight values themselves will be different, since we are essentially training the linear regression model on different data.
"""

part4_q3 = r"""
**Your answer:**

(1) When it comes to the regularization parameter, values of lambda that are on the same order of magnitude will tend to produce more similar results than values of lambda that are on different orders of magnitude. Thus, using np.logspace is preferable for CV since it allows us to investigate a wider range of lambda values that will have much more varying results. If we wanted to tune our model further, we could use np.linspace after already using np.logspace, in order to narrow our search first to the optimal order of magnitude and then finally we can find the best value on that order.

(2) Our grid search went over 20 values of lambda and 3 values for the degree, and it did so 3 times each during the cross-validation. Thus, the total number of times the model was fitted to the data was 20 * 3 * 3 = **180 times** total!
"""

# ==============
