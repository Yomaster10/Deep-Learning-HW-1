r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1) **FALSE**. The in-sample error is obtained from averaging the pointwise losses of the model on the training set only! The test set has no effect on the in-sample error.

2) **FALSE**. Every random unstratified train-test split could result in train and test sets that have different distributions for their data. Thus, some splits would be less "useful" than others, since their train and test sets do not both have distributions that reflect the actual data. Models trained on different unstratified train-test splits would show different levels of generalization and performance as a result.

3) **TRUE**. The cross-validation step should only ever include the training set and the validation sets produced during the process. This is because we are
utilizing this step to minimize the in-sample error and optimize the model solely based on data available within the training set. We only use the test set after cross-validation in order to assess the generalization performance of the model.

4) **FALSE**. The validation set is used to determine the in-sample error of the model (for each fold), whereas the generalization error is actually determined by model's performance on the test set.

"""

part1_q2 = r"""
**Your answer:**

No, his approach is definitely **unjustified**. We *never* use the test set to tune hyperparameters (doing so would introduce bias/information leakage into our model); we only use it to make conclusions about the generalization of the model. To tune the hyperparameter, he should use k-fold cross-validation and find the hyperparameter that minimizes the in-sample error.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

We see that increasing the value of k actually decreases the accuracy of our model, and this indicates to us that we are essentially overfitting our model to the training data the more we increase k (and thus we see the decrease in accuracy / increase in generalization error).

"""

part2_q2 = r"""
**Your answer:**

Explain why (i.e. in what sense) using k-fold CV, as detailed above, is better than:

(1) Training on the entire train-set with various models and selecting the best model with respect to train-set accuracy.
(2) Training on the entire train-set with various models and selecting the best model with respect to test-set accuracy.


(1) This is an erroneous approach to machine learning by itself; it could easily result in model overfitting to the training data, and thus such a model would generalize poorly. While k-fold CV does make use of accuracy scores obtained within the training set, it only does so on validation subsets that are separated/hidden during each instance of training. Thus, k-fold CV implements a sort of mini-generalization check, by checking each model on an unseen subset of the training data and then choosing the best model based on the validation accuracy. The fold process ensures better generalization, by comparing accuracy values across a set of validation datasets without much overfitting to any single training subset.

(2) This is an erroneous approach to machine learning by itself; we should never use the test set to choose thWhile this method is superior to the previous one, it still might not generalize as well as k-fold CV. This is because it is only ever training on one single training set, and so it is more likely that the best model will be overfitted on the training set. 

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
