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
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training


	pass
def test_expected_loss():
    """
    Because a random seed is used, the weight initialization, batch shuffling, and all other pseudo-random events should be reproducible. This test tests
    that the final loss at the end of training is what is expected, provided the hyperparameters are fixed and pseudo-random events are controlled through the random seed
    """

    X_train, X_test, y_train, y_test = loadDataset(split_percent = 0.7)
    lin_model = LogisticRegression(num_feats=6, max_iter=1000, tol=0.000001, learning_rate=0.001, batch_size=400,random_seed = 42) #set random seed
    lin_model.train_model(X_train, y_train, X_test, y_test)
    assert lin_model.loss_history_val[-1] == 0.6808438002254648, "Your final loss value is different than expected!"

def test_loss():
	"""
	Because the loss is being minimized, via gradient descent, using the training data, it is expected that the loss (with respect to the training data)
	will approach 0 as number of gradient descent itertions increases. Test that is the case
	"""
	X_train, X_test, y_train, y_test = loadDataset(split_percent = 0.7)
    model = LogisticRegression(num_feats=6, max_iter=1000, tol=0.000001, learning_rate=0.001, batch_size=400)
    model.train_model(X_train, y_train, X_test, y_test)
    num_training_iters  = len(model.loss_history_train)
    early_losses = model.loss_history_train[0:(num_training_iters // 3)] #loss history from first 1/3 of data
    middle_losses = model.loss_history_train[(num_training_iters // 3): 2 * (num_training_iters // 3)] # loss history from second 1/3 of data 
    late_losses = model.loss_history_train[ 2 * (num_training_iters // 3):] #loss history from last 1/3 of data

	assert that np.mean(early_losses) < np.mean(middle_losses) < np.mean(late_losses), "Your loss is not approaching 0!"

def test_loss_calculation():
	"""
	Given a small set of data and parameters, test the loss is calculated correctly by comparing the loss computed by the Class Object (in a vectorized way)
    to the loss computed in this function via a for loop, to the loss I calculated by hand. All 3 values should be identical. Assert that is the case
	"""
    

    X = np.array([[0.4,0.3,1],
                [0.1,0.9,1],
                [0.2,0.8,1]]) #last column is for bias term
    lin_model = LogisticRegression(num_feats=3)
    y = np.array([1,1,0])
    m = X.shape[0]
    lin_model.W = np.array([2.94,1.03,0.65])

    y_pred = lin_model.make_prediction(X)
    model_loss_val = lin_model.loss_function(X,y)

    ### calculate binary cross entropy loss by looping through each observation, appending the loss for that observation to a list, and taking the negative mean
    loss_vals = []
    for obs in range(X.shape[0]):
        single_loss = (y[obs]*np.log(y_pred[obs])) + ((1-y[obs])*np.log(1-y_pred[obs]))
        loss_vals.append(single_loss)
    loss = -np.mean(loss_vals)

    manually_calculated_loss = 0.8122346881829742

    assert np.allclose(model_loss_val, loss, manually_calculated_loss), "Loss function is not being calculated correctly!"


	

def test_grad_calc():
    """
    Test the self.calculate_gradient method performs correct calculations by comparing it's output to my manual calculations for a simple dataset with
    manually initialized parameters. Also compare it to the output from a non-vectorized calculation. 
    """

	X = np.array([[0.4,0.3,1],
                [0.1,0.9,1],
                [0.2,0.8,1]]) #last column is for bias term
    lin_model = LogisticRegression(num_feats=2)
    y = np.array([1,1,0])
    m = X.shape[0]
    lin_model.W = np.array([2.94,1.03,0.65])
    model_grad = lin_model.calculate_gradient(X,y)
    y_pred = lin_model.make_prediction(X)

    #calculate gradient in non-vectorized way
    grad_component = []
    for obs in range(X.shape[0]):
        grad_component.append((y[obs] - y_pred[obs]) * X[obs]) #y-y_pred is a scalar here
    grad = -np.sum(grad_component,axis = 0)/m #


    manual_grad = np.array([0.04059727, 0.1859726 , 0.21599574])

    assert np.allclose(model_grad,grad,manual_grad), "Your gradient is not being calculated properly!"

def test_update_W():
    """
    This test tests that the parameters in self.W get updated properly throughout gradient descent. Because the initialization of self.W will be
    controlled by the random seed, the hyperparameters are fixed, and the calculations of the gradiens, predictions,  and loss functions have been tested, 
    the self.W updates should be predictable/expected. Here, I assert W is being updated properly throughout gradient descent by checking the final W parameters after
    training are as expected by checking the final self.W object is as expected 
    """
    pass

def test_predict_calculation():
    """
    Test the self.make_predictions method performs correct calculations by comparing it's output to my manual calculations for a simple dataset with
    manually initialized parameters.
    """
    X = np.array([[0.4,0.3,1],
                [0.1,0.9,1],
                [0.2,0.8,1]]) #last column is for bias term
    lin_model = LogisticRegression(num_feats=3)
    y = np.array([1,1,0])
    m = X.shape[0]
    lin_model.W = np.array([2.94,1.03,0.65])

    model_y_pred = lin_model.make_prediction(X)
    manually_calc_pred = np.array([[0.8942587414290022,0.8665739434012857,0.8871545477278288]])

    assert np.allclose(model_y_pred,manually_calc_pred) "Class Predictions are not being calculated as expected!"

def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

def test_classification_ability():


	pass