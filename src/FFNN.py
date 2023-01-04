import numpy as np
from Schedulers import *
from utils import *
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler


class FFNN:
    """
    Feed Forward Neural Network

    Attributes:
        dimensions (list[int]): A list of positive integers, which defines our layers. The first number
        is the input layer, and how many nodes it has. The last number is our output layer. The numbers
        in between define how many hidden layers we have, and how many nodes they have.

        hidden_func (Callable): The activation function for the hidden layers

        output_func (Callable): The activation function for the output layer

        cost_func (Callable): Our cost function

        checkpoint_file (string): A file path where our weights will be saved

        weights (list): A list of numpy arrays, containing our weights
    """

    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = lambda x: x,
        cost_func: Callable = CostOLS,
        checkpoint_file: str = None,
        seed: int = None,
    ):
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.checkpoint_file = checkpoint_file
        self.seed = seed

        # weight initialization
        for i in range(len(self.dimensions) - 1):
            if self.seed is not None:
                np.random.seed(self.seed)
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.1

            self.weights.append(weight_array)

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler_class,
        *args,  # arguments for the scheduler
        batches: int = 1,
        epochs: int = 1000,
        lam: float = 0,
        X_test: np.ndarray = None,
        t_test: np.ndarray = None,
        use_best_weights: bool = False,
    ):
        """
        Trains the neural network via feedforward and backpropagation

        :param X: training data
        :param t: target data
        :param scheduler_class: specified scheduler
        :param args: scheduler args
        :param batches: batches to split data into
        :param epochs: how many epochs for training
        :param lam: regularization parameter
        :param use_best_weights: save the best weights and only use them. This runs slower because
        it has to check every epoch if it is better than the best. Only use with X_test set

        :return: scores dictionary containing test and train error amongst other things
        """

        # --------- setup ---------

        classification = False
        if (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            classification = True

        test_set = False
        if X_test is not None and t_test is not None:
            test_set = True

        # --- arrays of scores over epochs ----
        train_errors = np.empty(epochs)
        train_errors.fill(
            np.nan
        )  # nan makes for better plots if we cancel early (they are not plotted)
        test_errors = np.empty(epochs)
        test_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        test_accs = np.empty(epochs)
        test_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches

        if self.seed is not None:
            np.random.seed(self.seed)

        # logic for getting the best weights
        best_weights = deepcopy(self.weights)
        best_test_error = 10e20
        best_test_acc = 0
        best_train_error = 10e20
        best_train_acc = 0

        X, t = resample(X, t)

        checkpoint_length = epochs // 10
        checkpoint_num = 0

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(t)  # used for performance metrics
        if test_set:
            cost_function_test = self.cost_func(t_test)

        # create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(scheduler_class(*args))
            self.schedulers_bias.append(scheduler_class(*args))

        print(f"{scheduler_class.__name__}: Eta={args[0]}, Lambda={lam}")
        # this try is only so that we may cancel early by hitting ctrl+c
        try:
            for e in range(epochs):
                for i in range(batches):
                    # -------- minibatch gradient descent ---------
                    if i == batches - 1:
                        # if we are on the last, take all thats left
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(X_batch, t_batch, lam)

                # reset schedulers every epoch (some schedulers pass in this call)
                for scheduler in self.schedulers_weight:
                    scheduler.reset()

                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # --------- performance metrics -------

                prediction = self.predict(X, raw=True)
                train_error = cost_function_train(prediction)
                if train_error > 10e20:
                    # if this happens, we have a problem
                    length = 10
                    train_error = None
                    test_error = None
                    train_acc = None
                    test_acc = None
                    break
                if test_set:
                    prediction_test = self.predict(X_test, raw=True)
                    test_error = cost_function_test(prediction_test)

                    if use_best_weights:
                        if test_error < best_test_error:
                            best_test_error = test_error
                            best_train_error = train_error
                            best_weights = deepcopy(self.weights)
                    else:
                        best_test_error = test_error
                        best_train_error = train_error

                else:
                    test_errors = np.nan  # a

                train_acc = None
                test_acc = None
                if classification:
                    train_acc = accuracy(self.predict(X, raw=False), t)
                    train_accs[e] = train_acc
                    if test_set:
                        test_acc = accuracy(self.predict(X_test, raw=False), t_test)
                        test_accs[e] = test_acc
                        if use_best_weights:
                            if test_acc > best_test_acc:
                                best_test_acc = test_acc
                                best_train_acc = train_acc
                                best_weights = deepcopy(self.weights)
                        else:
                            best_test_acc = test_acc
                            best_train_acc = train_acc

                train_errors[e] = train_error
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

                # save to file every 10% if checkpoint file given
                if (e % checkpoint_length == 0 and self.checkpoint_file and e) or (
                    e == epochs - 1 and self.checkpoint_file
                ):
                    print(" " * length, end="\r")
                    checkpoint_num += 1
                    print()
                    print(f"{checkpoint_num}/10: Checkpoint reached")
                    self.write(self.checkpoint_file)

        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # overwrite last print so that we dont get 99.9 %
        print(" " * length, end="\r")
        self._progress_bar(
            1,
            train_error=train_error,
            test_error=test_error,
            train_acc=train_acc,
            test_acc=test_acc,
        )
        print()

        # update weights if specified
        if use_best_weights:
            self.weights = best_weights

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
        Return a prediction vector for each row in X

        Parameters:
            X (np.ndarray): The design matrix, with n rows of p features each

        Returns:
            z (np.ndarray): A prediction vector (row) for each row in our design matrix
            This vector is thresholded if we are dealing with classification and raw is not True
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
        Resets weights in order to reuse FFNN object for training
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
                if self.output_func.__name__ == "softmax":
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

    def _progress_bar(self, progression, **kwargs):
        length = 40
        num_equals = int(progression * length)
        num_not = length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = fmt(progression * 100, N=5)
        line = f"  {bar} {perc_print}% "

        for key in kwargs:
            if kwargs[key]:
                value = fmt(kwargs[key], N=4)
                line += f"| {key}: {value} "
        print(line, end="\r")
        return len(line)
