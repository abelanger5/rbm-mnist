# use numpy for array operations
import numpy as np 

class Mapping:
    def __init__(self, num_hidden, num_labels):
        self.num_hidden = num_hidden;
        self.num_labels = num_labels;

        # create array to hold hidden mean activations (array size of (num_labels, num_hidden))
        # self.mean_activations = (HERE)
        # create array to store total counts for each label (array size of (num_labels, 1))
        # self.total_counts = (HERE)
    
    def add_hidden(self, hidden):
        # compute new mean activation based on provided hidden array 
    
    def get_hidden(self, label):
        # return the mean activation array based on the label