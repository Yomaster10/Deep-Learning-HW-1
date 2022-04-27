import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        shape = (n_features, n_classes)
        self.weights = torch.normal(mean=0, std=weight_std, size=shape)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = x @ self.weights
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        N = y.shape[0]
    
        d = y - y_pred
        correct = d[d==0]
        num_correct = correct.shape[0]
    
        acc = num_correct / N
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            # Training Set
            num_of_batches = 0
            for x_batch, y_batch in dl_train:
                # predict
                y_pred, class_scores = self.predict(x_batch)
                # update accuracy
                total_correct += self.evaluate_accuracy(y_batch, y_pred)
                # update loss
                data_dependent_curr_loss_term = loss_fn.loss(x_batch, y_batch, class_scores, y_pred)
                regularization_curr_loss_term = weight_decay * (1/2) * (torch.norm(self.weights).item()**2)
                total_curr_loss = data_dependent_curr_loss_term + regularization_curr_loss_term
                average_loss += total_curr_loss
                # update weights
                self.weights -= learn_rate * loss_fn.grad()
                # up num of batches by 1
                num_of_batches += 1
            # update avg accuracy and loss
            train_res[0].append(total_correct / num_of_batches)
            train_res[1].append(average_loss / num_of_batches)

            # Validation Set
            total_correct = 0
            average_loss = 0
            num_of_batches = 0
            for x_batch, y_batch in dl_valid:
                # predict
                y_pred, class_scores = self.predict(x_batch)
                # update accuracy
                total_correct += self.evaluate_accuracy(y_batch, y_pred)
                # update loss
                data_dependent_curr_loss_term = loss_fn.loss(x_batch, y_batch, class_scores, y_pred)
                regularization_curr_loss_term = weight_decay * (1/2) * (torch.norm(self.weights).item()**2)
                total_curr_loss = data_dependent_curr_loss_term + regularization_curr_loss_term
                average_loss += total_curr_loss
                # update weights - don't need this in validation right?
                    # self.weights -= learn_rate * loss_fn(grad)
                # up num of batches by 1
                num_of_batches += 1
            # update avg accuracy and loss
            valid_res[0].append(total_correct / num_of_batches)
            valid_res[1].append(average_loss / num_of_batches)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        N = self.n_classes
        W = self.weights
        
        if has_bias is True:
            W = W[1:,:]
            
        w_images = torch.transpose(W,0,1)
        new_shape = (N, img_shape[0], img_shape[1], img_shape[2])
        
        w_images = w_images.reshape(new_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.001
    hp['learn_rate'] = 0.001
    hp['weight_decay'] = 0.001
    # ========================

    return hp
