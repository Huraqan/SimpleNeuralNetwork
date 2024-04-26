import pickle
from random import seed

import numpy as np
import pandas as pd

from Layer import Layer
from data_handler import split_batches, get_digit
from itertools import chain

line_ending = "============================================================================"

class SimpleNeuralNetwork:
    def __init__(self, sizes=[28 * 28, 16, 16, 10]) -> None:
        seed(42) # for reproducibility
        self.sizes = sizes
        self.layer_count = len(sizes)
        self.layers = [Layer(i, sizes) for i in range(len(self.sizes))]
        self.weights = [lyr.weights for lyr in self.layers if lyr.weights is not None]
        self.biases = [lyr.biases for lyr in self.layers if lyr.biases is not None]
        self.epochs = 0
    
    
    def train_network(self, training_data, batch_size=100, epochs=5, filepath="neural_network.pickle"):
        print("\nTraining network of size:", self.sizes)
        print("Number of finished epochs on this instance:", self.epochs)
        
        for epoch in range(epochs):
            print("\n=== EPOCH:", epoch + 1, "/", epochs, line_ending)
            
            batches = split_batches(training_data, size=batch_size)
            self.stochastic_gradient_descent(batches)
            
            self.save_to_file(filepath)
            self.epochs += 1
    
    
    def stochastic_gradient_descent(self, batches: list, re_score: bool = False):
        print("")
        
        scores = []
        for i, batch in enumerate(batches):
            info_string = f"Processed {i+1:03d} / {len(batches)} batches "
            
            win_fails = self.update_batch(batch, info_string)
            
            scores += win_fails
        
        print("Done! Average score during training:", ratio_string(scores))
        
        if re_score:
            tr_data = list(chain(*batches))
            self.validate(tr_data)
            
    
    def update_batch(self, batch: list, info_string = ""):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        win_fails = []
        
        for training_example in batch:
            input, expected_output = training_example
            
            delta_nw, delta_nb, win_fail = self.backpropagation(input, expected_output)
            
            for i in range(len(nabla_w)):
                nabla_w[i] += delta_nw[i]
                nabla_b[i] += delta_nb[i]
            
            win_fails.append(win_fail)
        
        # Apply gradients to all weights and biases to train the network
        
        learning_rate = 1.0 / len(batch)
        
        for i in range(len(self.weights)):
            self.weights[i] -= nabla_w[i] * learning_rate
            self.biases[i] -= nabla_b[i] * learning_rate
        
        print(info_string + ratio_string(win_fails), end="\r")
        
        return win_fails

    def backpropagation(self, input, expected_output):
        self.layers[0].activations = input
        
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        ########################### Feedforward ###########################
        # This is different to a simple prediction feedforward.
        # Here we must keep track of what our activations were before
        # the sigmoid transformation (or any other activation function).
        # We'll keep them in a list called 'z_vectors' and use them during backprop
        
        z_vectors = []
        
        for i in range(self.layer_count - 1):
            next_activations = self.next_activations(i)

            z_vectors.append(next_activations)
            
            self.layers[i + 1].activations = sigmoid(next_activations)
        
        error, win_fail = self.compute_error(self.layers[-1].activations, expected_output)
        
        delta_nabla_w[-1] += np.dot(error, self.layers[-2].activations.transpose())
        delta_nabla_b[-1] += error
        
        ########################### Backpropagation ###########################
        # Previous values for error are reused at each layer, going backwards
        # Direction naming:
        # [input layer] ... prev <- current -> next ... [output layer]
        # Backwards iteration from output to input using a negative index
        
        for layer_i in range(2, self.layer_count):
            prev_activations = self.layers[-layer_i - 1].activations.transpose()
            next_weights = self.weights[-layer_i + 1].transpose()
            error = np.dot(next_weights, error)
            error *= sigmoid_prime(z_vectors[-layer_i])
            
            delta_nabla_w[-layer_i] += np.dot(error, prev_activations)
            delta_nabla_b[-layer_i] += error
        
        return delta_nabla_w, delta_nabla_b, win_fail
    
    
    def compute_error(self, output, expected_output):
        error = output - expected_output
        
        if np.argmax(output) == np.argmax(expected_output):
            return error, "win"
        return error, "fail"
    
    
    def predict(self, input, verbose=False):
        output = np.round(self.feedforward(input), 3)
        
        if verbose:
            simple_scale = np.round(output / sum(output), 3)
            softmax_output = np.round(softmax(self.layers[-1].activations), 3)
            dic = {
                'OUTPUT': output.flatten(),
                'SIMPLE': simple_scale.flatten(),
                'SOFTMAX': softmax_output.flatten()
            }
            print(pd.DataFrame(dic), "\n")
        
        return output
    
    
    def feedforward(self, input):
        self.layers[0].activations = np.copy(input)

        for i in range(self.layer_count - 1):
            self.layers[i + 1].activations = sigmoid(self.next_activations(i))
        
        return self.layers[-1].activations
    
    
    def next_activations(self, i):
        a = self.layers[i].activations
        w = self.layers[i].weights
        b = self.layers[i].biases

        return np.dot(w, a) + b
    
    
    def calculate_scores(self, output, expected_output, verbose=False):
        output = self.layers[-1].activations
        error, win_fail = self.compute_error(output, expected_output)
        
        expected = pd.Series(expected_output.flatten())
        out = pd.Series(np.round(output, 2).flatten())
        rounded_error = pd.Series(np.round(error, 2).flatten())
        cost_float = np.sum(np.square(rounded_error))
        
        if verbose:
            df = pd.concat([expected, out, rounded_error, pd.Series([""]*10)], axis=1)
            df.columns = ["Expected output", "   Output", "  Error", f"   {win_fail}"]
            print("")
            print(df)
            print("\n    TOTAL COST:        ", round(cost_float, 4))
        
        return out, cost_float, win_fail
    
    
    def validate(self, validation_data, csv_path=""):
        results = [[], [], []]
        for i, validation_example in enumerate(validation_data):
            input, expected_output = validation_example
            output = self.feedforward(input)
            out, cost_float, win_fail = self.calculate_scores(output, expected_output)
            digit = get_digit(expected_output)
            results[0].append(digit if digit != 10 else "noise")
            results[1].append(cost_float)
            results[2].append(win_fail)
            
            print(f"{i+1}/{len(validation_data)} samples, current score:", ratio_string(results[2]), end="                    \r")
        
        print(f"{len(validation_data)} samples, final score:", ratio_string(results[2]), "                    ")
        
        # Save to CSV
        if not csv_path:
            return
        
        df = pd.concat([pd.Series(results[0]), pd.Series(results[1]), pd.Series(results[2])], axis=1)
        df.columns = ["Digit", "Cost", "Win or Fail"]
        df.to_csv("reports/" + csv_path, index=False)
    
    
    def save_to_file(self, fp="neural_network.pickle"):
        with open("models/" + fp, "wb") as f:
            pickle.dump(self, f)
        print("\nFile saved as", fp, "\n")
    
    @staticmethod
    def load_from_file(fp="neural_network.pickle"):
        with open("models/" + fp, "rb") as f:
            return pickle.load(f)


def sigmoid(array):
    return 1 / (1 + np.exp(-array))

def sigmoid_prime(z):
    """Derivative of the sigmoid function. I copy-pasted, don't question it."""
    sz = sigmoid(z)
    return sz * (1 - sz)

def reLU(self, array):
    return np.maximum(array, 0)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Improve numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    return np.round(probabilities, 3)


def ratio_string(win_fails:list, length=50):
    df = pd.DataFrame(win_fails).value_counts()
    
    if df.get("fail") is None:
        wins = length
        fails = 0
        score = 1
    elif df.get("win") is None:
        wins = 0
        fails = length
        score = 0
    else:
        score = df["win"] / (df["fail"] + df["win"])
        wins = int(length * score)
        fails = length - wins
    
    return f"{'\033[92m'}" + "|" * wins + f"{'\033[0m'}{'\033[91m'}" + "|" * fails + f"{'\033[0m'}" + f"  {score:.3f}"
    