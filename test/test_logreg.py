from regression import LogisticRegression
from regression import loadDataset
import numpy as np
import pandas as pd

"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""


def test_updates():
	"""

	"""
	# Check that your gradient is being calculated correctly
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training

	pass
def test_loss():
	"""
	Because the loss is being minimized, via gradient descent, using the training data, it is expected that the loss (with respect to the training data)
	will approach 0 as number of gradient descent itertions increases. Test that is the case
	"""
	X_train, X_test, y_train, y_test = loadDataset(split_percent = 0.7)
	model = LogisticRegression(num_feats=X.shape[1], max_iter=1000, tol=0.001, learning_rate=0.001, batch_size=10)
	model.train(X_train, X_test, y_train, y_test)
	num_training_iters  = length(model.loss_history_train)
	early_losses = model.loss_history_train[0:(num_training_iters // 3)] #loss history from first 1/3 of data
	middle_losses = model.loss_history_train[(num_training_iters // 3): 2 * (num_training_iters // 3)] # loss history from second 1/3 of data 
	late_losses = model.loss_history_train[ 2 * (num_training_iters // 3):] #loss history from last 1/3 of data

	assert that np.mean(early_losses) < np.mean(middle_losses) < np.mean(late_losses), "Your loss is not approaching 0!"

def test_loss_calculation():
	"""
	Given a small set of data and parameters, test the loss is calculated correctly by comparing the result to my hand-calculated result
	"""
	num_points = 5

	# Chose values for params (w vector) where w[-1] is bias term
	w = [2, 2.6, 1]


	

def test_grad_calc():
	pass

def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

	pass