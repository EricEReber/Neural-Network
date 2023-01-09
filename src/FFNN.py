import math
import autograd.numpy as np
from Schedulers import *
from activationFunctions import *
from costFunctions import *
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy
from typing import Tuple, Callable
from sklearn.utils import resample


class FFNN:
    """
    Description: 

        Feed Forward Neural Network with interface enabling flexible design of a 
        nerual networks architecture and the specification of activation function 
        in the hidden layers and output layer respectively. This model can be used 
        for both regression and classification problems, depending on the output function.

    Attributes:

        I.  dimensions (list[int]): A list of positive integers, which specifies the 
            number of nodes in each of the networks layers. The first integer in the array 
            defines the number of nodes in the input layer, the second integer defines number 
            of nodes in the first hidden layer and so on until the last number, which 
            specifies the number of nodes in the output layer.
        II. hidden_func (Callable): The activation function for the hidden layers
        III.output_func (Callable): The activation function for the output layer
        IV. cost_func (Callable): Our cost function
        V.  weights (list): A list of numpy arrays, containing our weights
    """
    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = lambda x: x,
        cost_func: Callable = CostOLS,
        seed: int = None, # Used for reproducibility 
     ):
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()

        # The initialization of weights for the Neural Network is done 
        # the moment an object of FFNN class is created
        for i in range(len(self.dimensions) - 1):
            if self.seed is not None:
                np.random.seed(self.seed)
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )
            # Matrix containing weights for the i-th layer in the network
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.1
            
            self.weights.append(weight_array)


    def fit(
        self,
        X: np.ndarray, 
        t: np.ndarray,
        scheduler_class: Scheduler, 
        *scheduler_args: list(),
        batches: int = 1,
        epochs: int = 100,
        lam: float = 0,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ):
        """
        Description: 
            
            This function performs the training the neural network by performing the feedforward and backpropagation
            algorithm to update the networks weights. 

        Parameters: 

            I   :param X: training data
            II  :param t: target data
            III :param scheduler_class: specified scheduler (algorithm for optimization of gradient descent)
            IV  :param scheduler_args: list of all arguments necessary for scheduler
            V   :param batches: number of batches the datasets are split into, default equal to 1
            VI  :param epochs: number of iterations used to train the network, default equal to 100
            VII :param lam: regularization hyperparameter lambda
            VIII:param X_val validation set
            IX  :param t_val validation target set
            X   :return: scores dictionary containing test and train error amongst other things

        """

        # --------- setup ---------
        if self.seed is not None:
            np.random.seed(self.seed)

        classification = False
        if (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            classification = True 

        test_set = False
        if X_val is not None and t_val is not None:
            test_set = True

        # --- Creating arrays for score metrics ----
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        test_errors = np.empty(epochs)
        test_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        test_accs = np.empty(epochs)
        test_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches

        best_test_error = 10e20
        best_test_acc = 0
        best_train_error = 10e20
        best_train_acc = 0

        X, t = resample(X, t)

        # this function/method returns a function valued only at X
        cost_function_train = self.cost_func(t)  # used for performance metrics
        if test_set:
            cost_function_test = self.cost_func(t_val)

        # create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(scheduler_class(*scheduler_args))
            self.schedulers_bias.append(scheduler_class(*scheduler_args))

        print(f"{scheduler_class.__name__}: Eta={scheduler_args[0]}, Lambda={lam}")
        # this try allows to abort the code early by pressing Ctrl+c
        try:
            for e in range(epochs):
                for i in range(batches):
                    # -------- minibatch gradient descent ---------
                    if i == batches - 1:
                        # If the for loop has reached the last batch, take all thats left
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]
                    
                    self._feedforward(X_batch)
                    self._backpropagate(X_batch, t_batch, lam)

                # reset schedulers for each epoch (some schedulers pass in this call)
                for scheduler in self.schedulers_weight:
                    scheduler.reset()

                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # --------- Computing performance metrics ---------
                prediction = self.predict(X, raw=True)
                train_error = cost_function_train(prediction)
                if train_error > 10e20:
                    # Indicates a problem with the training
                    length = 10
                    train_error = None
                    test_error = None
                    train_acc = None
                    test_acc = None
                    raise OverflowError
                if test_set:
                    prediction_test = self.predict(X_val, raw=True)
                    test_error = cost_function_test(prediction_test)
                    best_test_error = test_error
                    best_train_error = train_error

                else:
                    test_errors = np.nan

                train_acc = None
                test_acc = None
                if classification:
                    train_acc = self._accuracy(self.predict(X, raw=False), t)
                    train_accs[e] = train_acc
                    if test_set:
                        test_acc = self._accuracy(self.predict(X_val, raw=False), t_val)
                        test_accs[e] = test_acc
                        best_test_acc = test_acc
                        best_train_acc = train_acc

                train_errors[e] = train_error
                if not test_set:
                    progression = e / epochs

                    # ----- printing progress bar ------------
                    length = self._progress_bar(
                        progression,
                        train_error=train_error,
                        train_acc=train_acc,
                        test_acc=test_acc,
                    )

                if test_set:
                    test_errors[e] = test_error
                    progression = e / epochs

                    # ----- printing progress bar ------------
                    length = self._progress_bar(
                        progression,
                        train_error=train_error,
                        test_error=test_error,
                        train_acc=train_acc,
                        test_acc=test_acc,
                    )
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        # overwrite last print so that we dont get 99.9 %
        print(" " * length, end="\r")
        if not test_set:
            self._progress_bar(
                1,
                train_error=train_error,
                train_acc=train_acc,
            )
            print()
        else:
            self._progress_bar(
                1,
                train_error=train_error,
                test_error=test_error,
                train_acc=train_acc,
                test_acc=test_acc,
            )
            print()

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors
        scores["final_train_error"] = best_train_error

        if test_set:
            scores["test_errors"] = test_errors
            scores["final_test_error"] = best_test_error

        if classification:
            scores["train_accs"] = train_accs
            scores["final_train_acc"] = best_train_acc

            if test_set:
                scores["test_accs"] = test_accs
                scores["final_test_acc"] = best_test_acc

        return scores


    def predict(self, X: np.ndarray, *, raw=False, threshold=0.5):
        """
        Performs prediction after training of the network has been finished,
        labelling the output as either 1 or 0 in the case of classification, 
        or decimal numbers in the case of regression.  

        Parameters:
        
        I.  X (np.ndarray): The design matrix, with n rows of p features each

        Returns:
        I.  z (np.ndarray): A prediction vector (row) for each row in our design matrix
            This vector is thresholded if we are dealing with classification and raw if not True

        """

        predict = self._feedforward(X)
        if raw:
            return predict
        elif (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            return np.where(predict > threshold, 1, 0)
        else:
            return predict

    def reset_weights(self):
        """
        Resets weights in order to train the neural network for better
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    def _feedforward(self, X: np.ndarray):
        """
        Return a prediction vector for each row in X
        Parameters:
            X (np.ndarray): The design matrix, with n rows of p features each
            Returns:
            z (np.ndarray): A prediction vector (row) for each row in our design matrix
        """

        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # if X is just a vector, make it into a design matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # put a coloumn of ones as the first coloumn of the design matrix, so that
        # we have a bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # a^0, the nodes in the input layer (one a^0 for each row in X)
        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        # the feed forward part
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = a @ self.weights[i]
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                a = np.hstack([np.ones((a.shape[0], 1)), a])
                self.a_matrices.append(a)
            else:
                # a^L, the nodes in our output layer
                z = a @ self.weights[i]
                a = self.output_func(z)
                self.a_matrices.append(a)
                self.z_matrices.append(z)

        # this will be a^L
        return a

    def _backpropagate(self, X, t, lam):
        """
        Perform backpropagation
        Parameters:
            X (np.ndarray): The design matrix, with n rows of p features each
            t (np.ndarray): The target vector, with n rows of p targets
        Returns:
            does not return anything, but updates the weights
        """
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)
        update_list = list()

        for i in range(len(self.weights) - 1, -1, -1):
            # creating the delta terms
            if i == len(self.weights) - 1:
                if (
                    self.output_func.__name__ == "softmax"
                    and self.cost_func.__name__ == "CostCrossEntropy"
                ):
                    # here we just assume that if softmax, our cost function is cross entropy loss
                    delta_matrix = self.a_matrices[i + 1] - t
                else:
                    cost_func_derivative = grad(self.cost_func(t))
                    delta_matrix = out_derivative(
                        self.z_matrices[i + 1]
                    ) * cost_func_derivative(self.a_matrices[i + 1])

            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.z_matrices[i + 1])

            gradient_weights_matrix = np.zeros(
                (
                    self.a_matrices[i][:, 1:].shape[0],
                    self.a_matrices[i][:, 1:].shape[1],
                    delta_matrix.shape[1],
                )
            )

            for j in range(len(delta_matrix)):
                gradient_weights_matrix[j, :, :] = np.outer(
                    self.a_matrices[i][j, 1:], delta_matrix[j, :]
                )

            gradient_weights = np.mean(gradient_weights_matrix, axis=0)
            delta_accumulated = np.mean(delta_matrix, axis=0)

            gradient_weights += self.weights[i][1:, :] * lam

            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(
                        delta_accumulated.reshape(1, delta_accumulated.shape[0])
                    ),
                    self.schedulers_weight[i].update_change(gradient_weights),
                ]
            )

            update_list.insert(0, update_matrix)

        self._update_w_and_b(update_list)

    def _update_w_and_b(self, update_list):
        """
        Updates weights and biases using a list of arrays that matches
        self.weights
        """
        for i in range(len(self.weights)):
            self.weights[i] -= update_list[i]

    def _accuracy(self, prediction: np.ndarray, target: np.ndarray):
        """
        Calculates accuracy of given prediction to target
        """
        assert prediction.size == target.size
        return np.average((target == prediction))

    def _progress_bar(self, progression, **kwargs):
        """
        Displays progress of training
        """
        length = 40
        num_equals = int(progression * length)
        num_not = length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._fmt(progression * 100, N=5)
        line = f"  {bar} {perc_print}% "

        for key in kwargs:
            if kwargs[key]:
                value = self._fmt(kwargs[key], N=4)
                line += f"| {key}: {value} "
        print(line, end="\r")
        return len(line)

    def _fmt(self, value, N=4):
        """
        Formats decimal numbers for progress bar
        """
        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1
        n = 1 + math.floor(math.log10(v))
        if n >= N - 1:
            return str(round(value))
            # or overflow
        return f"{value:.{N-n-1}f}"
