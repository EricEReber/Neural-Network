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
    ------------
        Feed Forward Neural Network with interface enabling flexible design of a 
        nerual networks architecture and the specification of activation function 
        in the hidden layers and output layer respectively. This model can be used 
        for both regression and classification problems, depending on the output function.

    Attributes:
    ------------
        I.  dimensions (tuple[int]): A list of positive integers, which specifies the 
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
        ------------ 
            This function performs the training the neural network by performing the feedforward and backpropagation
            algorithm to update the networks weights. 

        Parameters: 
        ------------
            I    X (np.ndarray) : training data
            II   t (np.ndarray) : target data
            III  scheduler_class (Scheduler) : specified scheduler (algorithm for optimization of gradient descent)
            IV   scheduler_args (list[int]) : list of all arguments necessary for scheduler

        Optional Parameters:
        ------------
            V    batches (int) : number of batches the datasets are split into, default equal to 1
            VI   epochs (int) : number of iterations used to train the network, default equal to 100
            VII  lam (float) : regularization hyperparameter lambda
            VIII X_val (np.ndarray) : validation set
            IX   t_val (np.ndarray) : validation target set

        Returns: 
        ------------
            I.  scores (dict) : A dictionary containing the performance metrics of the model. The number of the metrics 
                depends on the parameters passed to the fit-function.

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

        val_set = False
        if X_val is not None and t_val is not None:
            val_set = True

        # --- Creating arrays for score metrics ----
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches

        best_val_error = 10e20
        best_test_acc = 0
        best_train_error = 10e20
        best_train_acc = 0

        X, t = resample(X, t)

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(t)  # used for performance metrics
        if val_set:
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
                prediction = self.predict(X, regression=True)
                train_error = cost_function_train(prediction)
                if train_error > 10e20:
                    # Indicates a problem with the training
                    length = 10
                    train_error = None
                    val_error = None
                    train_acc = None
                    val_acc = None
                    raise OverflowError
                if val_set:
                    prediction_test = self.predict(X_val, regression=True)
                    val_error = cost_function_test(prediction_test)
                    best_val_error = val_error
                    best_train_error = train_error

                else:
                    val_errors = np.nan

                train_acc = None
                val_acc = None
                if classification:
                    train_acc = self._accuracy(self.predict(X, regression=False), t)
                    train_accs[e] = train_acc
                    if val_set:
                        val_acc = self._accuracy(self.predict(X_val, regression=False), t_val)
                        val_accs[e] = val_acc
                        best_test_acc = val_acc
                        best_train_acc = train_acc

                train_errors[e] = train_error
                if not val_set:
                    progression = e / epochs

                    # ----- printing progress bar ------------
                    length = self._progress_bar(
                        progression,
                        train_error=train_error,
                        train_acc=train_acc,
                        val_acc=val_acc,
                    )

                if val_set:
                    val_errors[e] = val_error
                    progression = e / epochs

                    # ----- printing progress bar ------------
                    length = self._progress_bar(
                        progression,
                        train_error=train_error,
                        val_error=val_error,
                        train_acc=train_acc,
                        val_acc=val_acc,
                    )
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        print(" " * length, end="\r")
        if not val_set:
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
                val_error=val_error,
                train_acc=train_acc,
                val_acc=val_acc,
            )
            print()

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors
        scores["best_train_error"] = best_train_error

        if val_set:
            scores["val_errors"] = val_errors
            scores["best_val_error"] = best_val_error

        if classification:
            scores["train_accs"] = train_accs
            scores["best_train_acc"] = best_train_acc

            if val_set:
                scores["val_accs"] = val_accs
                scores["best_val_acc"] = best_test_acc

        return scores


    def predict(self, X: np.ndarray, *, regression=False, threshold=0.5):
        """
        Description: 
        ------------
            Performs prediction after training of the network has been finished.

        Parameters:
       ------------
            I.  X (np.ndarray): The design matrix, with n rows of p features each

        Optional Parameters:
        ------------
            II. regression (boolean) : if set to True, performs prediction for regression problems
            III.threshold (float) : sets minimal value for a prediction to be predicted as the positive class
                in classification problems

        Returns:
        ------------
            I.  z (np.ndarray): A prediction vector (row) for each row in our design matrix
                This vector is thresholded if regression=False, meaning that classification results
                in a vector of 1s and 0s, while regressions in an array of decimal numbers

        """

        predict = self._feedforward(X)
        # boolean == True is equivalent with the model performing regression 
        # (in other words, having no activation function in the output layer)
        if regression:
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
        Description: 
        ------------
            Resets/Reinitializes the weights in order to train the network for a new problem. 
        
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
        Description: 
        ------------
            Calculates the activation of each layer starting at the input and ending at the output. 
            Each following activation is calculated from a weighted sum of each of the preceeding 
            activations (except in the case of the input layer). 
        
        Parameters:
        ------------
            I.  X (np.ndarray): The design matrix, with n rows of p features each
        
        Returns:
        ------------    
            I.  z (np.ndarray): A prediction vector (row) for each row in our design matrix
        """

        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # if X is just a vector, make it into a design matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Add a coloumn of zeros as the first coloumn of the design matrix, in order 
        # to add bias to our data 
        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        # a^0, the nodes in the input layer (one a^0 for each row in X - where the 
        # exponent indicates layer number).
        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        # The feed forward algorithm
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = a @ self.weights[i]
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                # bias column again added to the data here
                bias = np.ones((a.shape[0], 1)) * 0.01
                a = np.hstack([bias, a]) 
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
        Description: 
        ------------
            Performs the backpropagation algorithm. In other words, this method
            calculates the gradient of all the layers starting at the 
            output layer, and moving from right to left accumulates the gradient until
            the input layer is reached. Each layers respective weights are updated while 
            the algorithm propagates backwards from the output layer (auto-differentation in reverse mode).
        
        Parameters:
        ------------
            I   X (np.ndarray): The design matrix, with n rows of p features each.
            II  t (np.ndarray): The target vector, with n rows of p targets.
            III lam (float32): regularization parameter used to punish the weights in case of overfitting

        Returns:
        ------------
            No return value. 

        """
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)
        update_list = list()

        # creating the delta terms
        for i in range(len(self.weights) - 1, -1, -1):
            # delta terms for output
            if i == len(self.weights) - 1:
                if (
                    self.output_func.__name__ == "softmax"
                    and self.cost_func.__name__ == "CostCrossEntropy"
                ):
                    delta_matrix = self.a_matrices[i + 1] - t
                else:
                    cost_func_derivative = grad(self.cost_func(t))
                    delta_matrix = out_derivative(
                        self.z_matrices[i + 1]
                    ) * cost_func_derivative(self.a_matrices[i + 1])

            # delta terms for hidden layer
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

        # update weights and biases
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
