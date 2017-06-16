import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class NeuralNetwork(object):
	'''Basic Neural Network Object to be trained on a given dataset.'''
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# Set number of nodes in input, hidden and output layers
		# Define architecture of the network
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# Set learning rate
		self.learning_rate = learning_rate

		# Initialize weights
		self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
		self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))

		# Define sigmoid activation function
		self.activation_function = lambda x : 1 / (1 + np.exp(-x))

	def train(self, features, targets):
		# No. of examples
		n_records = features.shape[0]

		# Initialize weights gradient to 0
		delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
		delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

		# Iterate over all training examples
		for X, y in zip(features, targets):
			#Forward pass
			hidden_inputs = np.dot(X, self.weights_input_to_hidden)
			hidden_outputs = self.activation_function(hidden_inputs)
        
			final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
			final_outputs = final_inputs

			# Backward pass
			# Calculate network error
			error = y - final_outputs
			# Output error term
			output_error_term = error * 1
			

        	
			hidden_error = np.dot(self.weights_hidden_to_output, error)
			# Hidden error term
			hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        	# Update gradient (input to hidden)
			delta_weights_i_h += X[:, None] * hidden_error_term.T
        	# Update gradient (hidden to output)
			delta_weights_h_o += hidden_outputs[:, None] * output_error_term

		self.weights_input_to_hidden += self.learning_rate * delta_weights_i_h / n_records
		self.weights_hidden_to_output += self.learning_rate * delta_weights_h_o / n_records

	def run(self, features):
    	#Feed forward propagation
		hidden_inputs = np.dot(features, self.weights_input_to_hidden)
		hidden_outputs = self.activation_function(hidden_inputs)
        
		final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
		final_outputs = final_inputs
        
		return final_outputs



