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
        self.weights = torch.normal(0, weight_std, (self.n_features,self.n_classes))

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
        class_scores = torch.matmul(x, self.weights)
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
        sub = y - y_pred
        zeros = sub[sub == 0]
        acc = zeros.shape[0] / y.shape[0]
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
            num_of_batches = 0
            for x_batch, y_batch in dl_train:
                '''predict for each mininbatch, calc loss and update weights with SGD '''
                y_pred, scores = self.predict(x_batch)
                total_correct += self.evaluate_accuracy(y_batch, y_pred)
                loss_no_r = loss_fn.loss(x_batch, y_batch, scores, y_pred)
                reg = weight_decay * (1/2) * (torch.norm(self.weights).item()**2)
                total_temp_loss = loss_no_r + reg
                average_loss += total_temp_loss
                self.weights -= learn_rate * loss_fn.grad()
                
                
                num_of_batches += 1
                
            # avg acc and loss
            train_res[0].append(total_correct / num_of_batches)
            train_res[1].append(average_loss / num_of_batches)

            # Evaluate validation set
            total_correct = 0
            average_loss = 0
            num_of_batches = 0
            for x_batch, y_batch in dl_valid:
                y_pred, scores = self.predict(x_batch)
                total_correct += self.evaluate_accuracy(y_batch, y_pred)
                loss_no_r = loss_fn.loss(x_batch, y_batch, scores, y_pred)
                reg = weight_decay * (1/2) * (torch.norm(self.weights).item()**2)
                total_temp_loss = loss_no_r + reg
                average_loss += total_temp_loss
                num_of_batches += 1
            
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
        if has_bias is True:
            weights = self.weights[1:,:]
            w_images = torch.transpose(weights,0,1)
            shape_to_reshape_to = (self.n_classes, img_shape[0], img_shape[1], img_shape[2])
            w_images = w_images.reshape(shape_to_reshape_to)
        else:
            w_images = torch.transpose(self.weights,0,1)
            shape_to_reshape_to = (self.n_classes, img_shape[0], img_shape[1], img_shape[2])
            w_images = w_images.reshape(shape_to_reshape_to)
            
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
