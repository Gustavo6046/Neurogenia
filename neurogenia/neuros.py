import numpy as np # Numeric Processing
import math        # Basic Pythonic Maths


class Layer(object):
    inherit_init = False

    """
    The Layer class is a generic abstraction for every neural
    network layer in Neuros.
    
    This class is also used as an input class for neural networks.
    
    The reason we didn't use Keras or another high- or low-level
    tensor library is simplicity and performance; this library
    implements a genetic algorithm that needs to work with a list
    of numbers, so it must be simple, and Keras is overstructured,
    with many excesses like tensors.
    """
    
    def __init__(self, shape, id=None, **kwargs):
        self.id = (id if id is not None else ''.join(np.random.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), (20,))))
        self.shape = shape
        self.kwargs = kwargs

    def copy(self, copy_genes=True, copy_id=False):
        """
        Returns a copy of this layer.
        """
        res = type(self)(self.shape, (self.id if copy_id else None))
        
        if hasattr(self, 'input'):
            res(self.input)
            
        if copy_genes:
            print("genecopy")
            res.set_genes(self.get_genes())
        
        return res
        
    def set_input(self, data):
        """
        Sets per-layer input data, like the model's input in
        a Layer class, or other special attributes that aren't
        constant (unlike the activations of an Activation layer),
        and aren't particular to the previous (local input) layer.
        """
        self.input_data = data.reshape(self.shape)
        
    def __call__(self, input, genes=None):
        """
        Initializes the network's input layer.
        """
        self.input = input
        
        if (
            (type(self).inherit_init and hasattr(self, 'layer_init'))
            or (not type(self).inherit_init and hasattr(type(self), 'layer_init'))
        ):
            self.layer_init(genes, **self.kwargs)
        
    def retrieve(self):
        """
        Returns a transformation of the previous layer
        (aka local input)'s data, usually a multiplication,
        activation or concatenation.
        """
        return np.array(self.input_data)
        
    def get_genes(self):
        """
        Returns an one-dimensional array with the genetic representation
        of this layer's important attributes, e.g. weights, for Dense
        layers, or alpha, for Activation layers,
        """
        return []
        
    def set_genes(self, genes):
        """
        Sets the state of this layer based on the given 1D genetic representation,
        e.g. the weights of Dense layers.
        """
        pass
        
class Dense(Layer):
    """
    The Dense object is a basic keras.layers.Dense-esque layer, where
    every neuron's value is a weighted sum of the input's values.
    """
    inherit_init = True
    
    def layer_init(self, genes=None, random_range=1):
        if genes is not None:
            self.weights = np.array(genes).reshape((self.shape, self.input.shape))
        
        else:
            self.weights = np.random.uniform(-random_range, random_range, (self.shape, self.input.shape))
            
    def retrieve(self):
        """
        Returns a transformation of the previous layer
        (aka local input)'s data; in this case, a multiplication
        of the Dense layer's weights with the previous output.
        """
        prev = self.input.retrieve()
        res = []
        
        for wl in list(self.weights):
            r = 0
            
            for i, w in enumerate(wl):
                r += w * prev[i]
                
            res.append(r)
            
        return np.array(res)
        
    def get_genes(self):
        """
        Returns an one-dimensional array with the genetic representation
        of this layer's important attributes, i.e., weights, for this Dense
        layer.
        """
        return self.weights.flatten()
        
    def set_genes(self, genes):
        """
        Sets the state of this layer based on the given 1D genetic representation,
        i.e.. the weights of this Dense layer.
        """
        self.weights = genes.reshape((self.shape, self.input.shape))
        
class Activation(Layer):
    """
    The Activation object is a neural network layer, that will
    apply a single function (usually the activator or squasher) to
    the "tensor" retrieved from the input layer.
    """
    
    activations = {
        "SIGMOID": lambda x: 1 / (1 + math.exp(-x)),
        "RELU": lambda x: max(0, x),
        "-RELU": lambda x: min(0, x)
    }
    
    def __init__(self, activation, id=None, **kwargs):
        self.id = (id if id is not None else ''.join(np.random.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), (20,))))
        
        if type(activation) is str:
            if activation.upper() not in type(self).activations:
                raise ValueError("Unknown activation: {}".format(activation))
        
            self.activation = type(self).activations[activation.upper()]
            
        else:
            self.activation = activation
            
        self.kwargs = kwargs
        
    def copy(self, copy_genes=True, copy_id=False):
        """
        Returns a copy of this layer.
        """
        res = type(self)(self.activation, (self.id if copy_id else None))
        
        if hasattr(self, 'input'):
            res(self.input)
            
        if copy_genes:
            res.set_genes(self.get_genes())
            
        res(self.input)
        
        return res
        
    def __call__(self, input, genes=None):
        """
        Initializes the network's input layer and activation shape
        """
        self.shape = input.shape
        super().__call__(input, genes)
        
    def retrieve(self):
        """
        Returns a transformation of the previous layer
        (aka local input)'s data; in this case, an activation
        (or squash).
        """
        prev = self.input.retrieve()
        
        return np.array(list(map(self.activation, list(prev))))
        
class Softmax(Layer):
    """
    The Softmax object is an Activation-like neural network layer,
    that will apply a softmax function to the local input (previous
    layer)'s output.
    """
    
    def __init__(self, id=None, **kwargs):
        self.id = (id if id is not None else ''.join(np.random.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), (20,))))
        self.kwargs = kwargs
        
    def copy(self, copy_genes=True, copy_id=False):
        """
        Returns a copy of this layer.
        """
        res = type(self)((self.id if copy_id else None))
        
        if hasattr(self, 'input'):
            res(self.input)
            
        if copy_genes:
            res.set_genes(self.get_genes())
            
        res(self.input)
        
        return res
        
    def __call__(self, input, genes=None):
        """
        Initializes the network's input layer and activation shape
        """
        self.shape = input.shape
        super().__call__(input, genes)
        
    def retrieve(self):
        """
        Returns a transformation of the previous layer
        (aka local input)'s data; in this case, an activation
        (or squash).
        """
        prev = self.input.retrieve()
        
        return np.exp(prev) / np.sum(np.exp(prev))
        
class Operation(Layer):
    """
    This class exists to serve as an abstract class,
    mostly to implement multi-input copy.
    """
    def __init__(self, id=None, **kwargs):
        self.id = (id if id is not None else ''.join(np.random.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), (20,))))
        self.shape = None # Operations take their shapes from their inputs. 
        self.kwargs = kwargs
    
    def copy(self, copy_genes=True, copy_id=False):
        """
        Returns a copy of this layer.
        """
        res = type(self)((self.id if copy_id else None))
        
        if hasattr(self, 'input'):
            res(self.input)
            
        if copy_genes:
            res.set_genes(self.get_genes())
            
        res(self.inputs)
        
        return res

class Concatenation(Operation):
    def __call__(self, inputs):
        """
        Initializes the network's input layer.
        """
        self.shape = sum([x.shape for x in inputs])
        self.inputs = inputs
        
    def retrieve(self):
        """
        Returns a transformation of the previous layers
        (aka local inputs)' data; in this case, a
        concatienation.
        """
        return np.concatenate([x.retrieve() for x in self.inputs])
        
class Addition(Operation):
    def __call__(self, inputs):
        """
        Initializes the network's input layer.
        """
        for ind, i in enumerate(inputs[1:]):
            if i.shape != inputs[0].shape:
                raise ValueError("Input #{} of an Addition has shape {}, but {} was expected!".format(ind, i.shape, inputs[0].shape))
        
        self.shape = inputs[0].shape
        self.inputs = inputs
        
    def retrieve(self):
        """
        Returns a transformation of the previous layers
        (aka local inputs)' data; in this case, an
        addition.
        """
        return sum([x.retrieve() for x in self.inputs])
        
class Multiplication(Operation):
    def __call__(self, inputs):
        """
        Initializes the network's input layer.
        """
        for ind, i in enumerate(inputs[1:]):
            if i.shape != inputs[0].shape:
                raise ValueError("Input #{} of an Addition has shape {}, but {} was expected!".format(ind, i.shape, inputs[0].shape))
        
        self.shape = inputs[0].shape
        self.inputs = inputs
        
    def retrieve(self):
        """
        Returns a transformation of the previous layers
        (aka local inputs)' data; in this case, a
        multiplication.
        """
        prev = [x.retrieve() for x in self.inputs]
        res = prev[0]
        
        for a in prev[1:]:
            res *= a
            
        return res
        
class Subtraction(Operation):
    def __call__(self, inputs):
        """
        Initializes the network's input layer.
        """
        for ind, i in enumerate(inputs[1:]):
            if i.shape != inputs[0].shape:
                raise ValueError("Input #{} of an Addition has shape {}, but {} was expected!".format(ind, i.shape, inputs[0].shape))
        
        self.shape = inputs[0].shape
        self.inputs = inputs
        
    def retrieve(self):
        """
        Returns a transformation of the previous layers
        (aka local inputs)' data; in this case, a
        subtraction.
        """
        prev = [x.retrieve() for x in self.inputs]
        res = prev[0]
        
        for a in prev[1:]:
            res -= a
            
        return res
        
class Division(Operation):
    def __call__(self, inputs):
        """
        Initializes the network's input layer.
        """
        for ind, i in enumerate(inputs[1:]):
            if i.shape != inputs[0].shape:
                raise ValueError("Input #{} of an Addition has shape {}, but {} was expected!".format(ind, i.shape, inputs[0].shape))
        
        self.shape = inputs[0].shape
        self.inputs = inputs
        
    def retrieve(self):
        """
        Returns a transformation of the previous layers
        (aka local inputs)' data; in this case, a
        division.
        """
        prev = [x.retrieve() for x in self.inputs]
        res = prev[0]
        
        for a in prev[1:]:
            res /= a
            
        return res
        
class Model(object):
    """
    A Keras-esque Model object, for neural networks
    and similar objects.
    """
    
    def __init__(self, input, output, layer_set=None):
        self.input = input
        self.output = output
        
        if layer_set is None:
            layer_set = set()
            
            def find_layers(c):
                while c is not None and c not in layer_set:
                    layer_set.add(c)
                    
                    if hasattr(c, 'inputs'): # Multiple Inputs (e.g. Operator layers)
                        for a in c.inputs:
                            find_layers(a)
                    
                    else:
                        c = getattr(c, "input", None)
        
            find_layers(output)
            
        self.layer_set = set(layer_set)
        
    def copy(self, copy_genes=True, copy_ids=False):
        """
        Returns a deep copy of this Model.
        """
        layer_copies = set()
        lays = {}
        inp = None
        out = None
        
        for l in self.layer_set:
            copy = l.copy(copy_genes, copy_ids)
            lays[l.id] = copy
        
            if l.id == self.output.id:
                out = copy
                
            elif l.id == self.input.id:
                inp = copy
                
            layer_copies.add(copy)
            
        for l in self.layer_set:
            if hasattr(l, 'input'):
                lays[l.id](lays[l.input.id])
                
            elif hasattr(l, 'inputs'):
                lays[l.id]([lays[lx.id] for lx in l.inputs])
            
        return Model(inp, out, layer_copies)
        
    def find_layer(self, id, default=None):
        """
        Returns the layer of the given ID in this model.
        Rarely needs to be used externally.
        """
        for l in self.layer_set:
            if l.id == id:
                return l
                
        return default
        
    def get_genes(self):
        """
        Returns a dictionary with the genes of all
        the layers in this Model.
        """
        res = {}
        
        for l in self.layer_set:
            res[l.id] = l.get_genes()
            
        return res
        
    def set_genes(self, genes):
        """
        Applies a dictionary with the new genes of all
        the layers in this Model.
        
        Returns the amount of gene arrays ("chromossomes")
        which corresponding layers were not found (by ID).
        """
        i = 0
        
        for k, v in genes.items():
            l = self.find_layer(k)
        
            if l is not None:
                l.set_genes(v)
                
            else:
                i += 1
            
        return i
        
    def target_fitness(self, inputs, outputs):
        """
        Returns how well this model fits to
        convert the input to the output.
        
        Since we use loss as a measure in neural
        networks, fitness will be negative loss;
        so those which are closest to 0 are the
        best ones, while maintaining the usual
        crescent order of fitness.
        """
        loss = 0
        
        for i, input in enumerate(inputs):
            output = outputs[i]
        
            predicted = self.predict(input)
            l = 0
            
            for i, pred in enumerate(predicted):    
                l += (output[i] - pred) ** 2
                
            loss += l / len(output)
            
        return -loss / len(inputs)
        
    def predict(self, input):
        """
        Returns the result value of this model, after giving it
        the input values.
        """
        self.input.set_input(np.array(input))
        
        return self.output.retrieve() # This simple! :)
        
def sequence(layers, genes=None, restrict_layer_set=False):
    prev = layers[0]

    for i, l in enumerate(layers[1:]):
        if genes is not None:
            l(prev, genes[i])
            
        else:
            l(prev)
    
        prev = l
        
    return Model(layers[0], layers[-1], (set(layers) if restrict_layer_set else None))