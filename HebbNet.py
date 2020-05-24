import math
import numpy as np

gamma = .5
threshold = -1 * np.log((1 / gamma) - 1)


def sigmoid(x):  # Avoid overflow by printing asymptotic values for sufficiently large |x|
    if x < -20:
        return 0
    elif x > 20:
        return 1
    else:
        return 1 / (1 + math.exp(-x))


def supersigmoid(x):
    return max(sigmoid(x - threshold) - gamma, 0) / gamma
    # return max(sigmoid(x) - .5, 0) / .5


alpha = .1  # theta_d in O'Reilly


def xcal(z, theta):
    global alpha
    if z > alpha * theta:
        return z - theta
    else:
        return z * (alpha - 1) / alpha
    # TODO level off for large z instead of ---> infinity


def contrast(w):
    if w == 0 or w == 1:
        return w
    gamma = 6
    theta = 1.25
    return 1 / (1 + (w / (theta * (1 - w))) ** (-1 * gamma))


# TODO Some kind of sparse mechanism so it doesn't waste time on inactive neurons

class Neuron:
    next_id = 1
    beta = .7  # Determines how much of the new weight learning threshold comes from the old
    learn_rate = .1

    def __init__(self, scale):
        self.inputs = {}  # id to input dict
        self.outputs = []  # actual neuron list
        self.weights = {}  # id to weight dict
        self.bias = 0  # TODO How does this work for more biologically realistic neurons?
        self.firing = 0
        self.weight_learning_threshold = np.random.normal(loc=.2, scale=.1)
        self.scale = scale
        self.neuron_id = Neuron.next_id
        Neuron.next_id += 1

    def compute_firing(self, memory):
        activation = self.bias
        for neuron_id in self.inputs:
            # activation += self.inputs[neuron_id] * self.weights[neuron_id]
            activation += self.inputs[neuron_id] * contrast(self.weights[neuron_id])  # Debug
        self.firing = memory * self.firing + (1 - memory) * supersigmoid(activation / self.scale)

    def learn(self):
        self.weight_learning_threshold *= Neuron.beta
        self.weight_learning_threshold += (1 - Neuron.beta) * self.firing ** 2
        for neuron_id in self.inputs:
            weight_change = Neuron.learn_rate * xcal(self.inputs[neuron_id] * self.firing,
                                                     self.weight_learning_threshold)
            # self.weights[neuron_id] += weight_change
            if weight_change > 0:
                self.weights[neuron_id] = self.weights[neuron_id] + (1 - self.weights[neuron_id]) * weight_change
            else:
                self.weights[neuron_id] = self.weights[neuron_id] + self.weights[neuron_id] * weight_change

    def fire(self):
        for output in self.outputs:
            output.receive(self.firing, self.neuron_id)

    def receive(self, signal, input_neuron):
        self.inputs[input_neuron] = signal
        
    def clear(self):
        self.firing = 0
        for neuron_id in self.inputs:
            self.inputs[neuron_id] = 0


# TODO Fix the issue with neuron IDs occurring if multiple nets are created in a runtime.
class HebbNet:
    def __init__(self, scale=30, weight_init_exp=.25, weight_init_std=.25):
        self.neurons = {}  # id to neuron
        self.scale = scale
        self.weight_init_exp = weight_init_exp
        self.weight_init_std = weight_init_std

    def connect_neurons(self, source_id, target_id):
        source = self.neurons[source_id]
        target = self.neurons[target_id]
        target.inputs[source_id] = 0  # source
        target.weights[source_id] = np.random.normal(loc=self.weight_init_exp, scale=self.weight_init_std)
        source.outputs.append(target)

    def add_neuron(self, sources=[], targets=[]):
        neuron = Neuron(scale=self.scale)
        self.neurons[neuron.neuron_id] = neuron
        for source in sources:
            self.connect_neurons(source, neuron)
        for target in targets:
            self.connect_neurons(neuron, target)

    def new_inputs(self, values, memory):
        if len(values) != len(self.neurons):
            print("Error:  The number of values given must match the number of inputs!")
            print(f"# values: {len(values)} \n # inputs:  {len(self.neurons)}")
            return
        else:
            for index in range(1, len(values) + 1):
                neuron = self.neurons[index]
                neuron.firing = memory * neuron.firing + (1 - memory) * values[index - 1]
                neuron.fire()

    def fire_neurons(self, memory):
        for neuron in self.neurons.values():
            neuron.compute_firing(memory)
            neuron.learn()
        for neuron in self.neurons.values():
            neuron.fire()
            
    def get_encoding(self):
        return np.array([neuron.firing for neuron in self.neurons.values()])
    
    def clear_neurons(self):
        for neuron in self.neurons.values():
            neuron.clear()
