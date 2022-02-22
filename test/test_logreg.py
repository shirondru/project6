from regression import LogisticRegression,loadDataset
import numpy as np
import pandas as pd
import pytest

"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""


def test_expected_loss():
    """
    Because a random seed is used, the weight initialization, batch shuffling, and all other pseudo-random events should be reproducible. Therefore, the precise final training loss can be expected. This test tests
    that the final loss at the end of training is what is expected, provided the hyperparameters are fixed and pseudo-random events are controlled through the random seed
    """

    X_train, X_test, y_train, y_test = loadDataset(split_percent = 0.7)
    lin_model = LogisticRegression(num_feats=6, max_iter=1000, tol=0.000001, learning_rate=0.001, batch_size=400,random_state = 42) #set random seed
    lin_model.train_model(X_train, y_train, X_test, y_test)

    #test final training loss is approx identical to the expected value under these hyperparameters and random seed
    assert np.allclose(lin_model.loss_history_val[-1],0.5745952593988821), "Your final loss value is different than expected!"

def test_loss():
    """
    Because the loss is being minimized via gradient descent it is expected that the loss (with respect to the training data)
    will approach 0 as number of gradient descent iterations increases. Test that is the case by asserting the loss from the 1st third of the data 
    is greater than the loss from the middle third of the data, which is greater than the loss from the last third of the data.
    """
    X_train, X_test, y_train, y_test = loadDataset(split_percent = 0.7)
    model = LogisticRegression(num_feats=6, max_iter=10000, tol=0.000001, learning_rate=0.001, batch_size=400)
    model.train_model(X_train, y_train, X_test, y_test)
    num_training_iters  = len(model.loss_history_train)
    early_losses = model.loss_history_train[0:(num_training_iters // 3)] #loss history from first 1/3 of data
    middle_losses = model.loss_history_train[(num_training_iters // 3): 2 * (num_training_iters // 3)] # loss history from second 1/3 of data 
    late_losses = model.loss_history_train[ 2 * (num_training_iters // 3):] #loss history from last 1/3 of data

    #assert the training loss approaches 0 
    assert np.mean(early_losses) > np.mean(middle_losses) > np.mean(late_losses), "Your loss is not approaching 0!"

def test_loss_calculation():
    """
    Given a small set of data and parameters, test the loss is calculated correctly by comparing the loss computed by the Class Object (in a vectorized way)
    to the loss computed in this function via a for loop, to the loss I calculated by hand. All 3 values should be identical. Assert that is the case
    """


    X = np.array([[0.4,0.3,1],
                [0.1,0.9,1],
                [0.2,0.8,1]]) #last column is for bias term
    lin_model = LogisticRegression(num_feats=2)
    y = np.array([1,1,0])
    m = X.shape[0]
    lin_model.W = np.array([2.94,1.03,0.65]) #manually initialize W

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
    manually initialized parameters. Also compare it to the output from a non-vectorized calculation. Assert all 3 values are identical to ensure gradient calculation is correct
    Here, gradient is calculated on a manually defined set of parameters (W)
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
    grad = -np.sum(grad_component,axis = 0)/m #gradient result via calculation in a non vectorized way


    manual_grad = np.array([0.04059727, 0.1859726 , 0.21599574]) #gradient results by manual calculation

    assert np.allclose(model_grad,grad,manual_grad), "Your gradient is not being calculated properly!"

def test_update_W():
    """
    This test tests that the parameters in self.W get updated properly throughout gradient descent. Because the initialization of self.W will be
    controlled by the random seed, the hyperparameters are fixed, and the calculations of the gradients, predictions,  and loss functions have all been tested to work properly, 
     self.W should be updated in a reproducible/expected manner throughout gradient descent. Here, I assert self.W is being updated properly throughout gradient descent by testing that the final values of
    self.W are different from the initial set of self.W values, and that the final values are the same as what is expected. As long as nothing has changed in the calculation of the gradient, predictions, or loss functions, or 
    that nothing has changed in the gradient descent code or any of the hyperparameters or random seed, this should be true.
    """
    X_train, X_test, y_train, y_test = loadDataset(split_percent = 0.8)
    lin_model = LogisticRegression(num_feats=X_train.shape[1], max_iter=10000, tol=0.000001, learning_rate=0.001, batch_size=400,random_state = 42)
    init_w = lin_model.W
    lin_model.train_model(X_train, y_train, X_test, y_test)
    final_w = lin_model.W
    expected_final_w = np.array([ 0.89807518,  0.66200367,  1.06751873,  1.52302986, -0.23415337, -0.06909126,  1.36745666]) #expected final set of W parameters given the random seed and hyperparamters being fixed
    
    #test final set of parameters are different than the initial set of parameters, and test that final set of parameters are same as what is expected, given the random seed and hyperparameters being fixed
    assert (not np.allclose(init_w, final_w)) and np.allclose(final_w,expected_final_w), "Your final W values are not as expected"

def test_predict_calculation():
    """
    Test the self.make_predictions method performs correct calculations by comparing it's output to my manual calculations for a simple dataset with
    manually initialized parameters.
    """
    X = np.array([[0.4,0.3,1],
                [0.1,0.9,1],
                [0.2,0.8,1]]) #last column is for bias term
    lin_model = LogisticRegression(num_feats=2)
    y = np.array([1,1,0])
    m = X.shape[0]
    lin_model.W = np.array([2.94,1.03,0.65])

    model_y_pred = lin_model.make_prediction(X)
    manually_calc_pred = np.array([[0.8942587414290022,0.8665739434012857,0.8871545477278288]])

    assert np.allclose(model_y_pred,manually_calc_pred), "Class Predictions are not being calculated as expected!"

def test_predict():

    """
    Because a seed is set, hyperparameters fixed, and the initialization of W is also set, all predictions for a given dataset should be reproducible under the same conditions.
    Additionally, because the gradient, loss, and prediction functions are tested to be calculated properly, the predictions for a given dataset should also have an expected value.
    Here, I test the classification accuracy of the logistic regression model on the NSCLC dataset is the expected value.

    ** This also tests that the algorithm is able to make good predictions, as the model is expected to generate somewhat high classification accuracy on this dataset**
    """

    X_train, X_test, y_train, y_test = loadDataset(split_percent = 0.8)
    lin_model = LogisticRegression(num_feats=X_train.shape[1], max_iter=10000, tol=0.000001, learning_rate=0.001, batch_size=400,random_state = 42)
    lin_model.train_model(X_train, y_train, X_test, y_test)
    y_pred = lin_model.make_prediction(X_test) #form predictions on held out test data
    #convert probabilities into binary classifications
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    accuracy = sum(y_pred == y_test)/len(y_test)

    #test predictions are as expected (and good)
    assert accuracy == 0.795, "Accuracy Score is Different than Expeted!"



