# use numpy for array operations
import numpy as np 

class Mapping:
    def __init__(self, num_hidden, num_labels):
        self.num_hidden = num_hidden;
        self.num_labels = num_labels;

        # create array to hold hidden mean activations (array size of (num_labels, num_hidden))
        self.mean_activations = np.zeros([num_labels, num_hidden])
        # create array to store total counts for each label (array size of (num_labels, 1))
        self.total_counts = np.zeros([num_labels, 1])
    
    def add_hidden(self, hidden, label):
        # compute new mean activation based on provided hidden array 
        self.mean_activations[label] = (self.total_counts[label] * self.mean_activations[label] + hidden) / (self.total_counts[label] + 1);

        self.total_counts[label] = self.total_counts[label] + 1;
    
    def get_hidden(self, label):
        # return the mean activation array based on the label
        return self.mean_activations[label]

""" TESTING
mapping = Mapping(5, 2);

mapping.add_hidden([0,0,0,0,0], 0);
mapping.add_hidden([.5,.5,.5,.5,.5], 0);

mapping.add_hidden([1,1,1,1,1], 1);
mapping.add_hidden([.5,.5,.5,.5,.5], 1);

print(mapping.get_hidden(0)) # should return [.25,.25,.25,.25,.25]
print(mapping.get_hidden(1)) # should return [.75,.75,.75,.75,.75]

"""