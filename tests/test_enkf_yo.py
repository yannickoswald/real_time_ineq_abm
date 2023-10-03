import copy

import pytest


import rtabm.run_enkf as run_enkf


@pytest.fixture()
def test_enkf_yo():
    '''This fixture prepares an EnsembleKalmanFilter object that will be
    used in later tests'''
    # There is some handy code in `run_enkf.py` that I use, but you
    # probably want to create a simpler/smaller EnKF for testing.
    enkf = run_enkf.prepare_enkf()
    yield enkf

def test_predict(test_enkf_yo):
    '''Test the predict function of EnsembleKalmanFilter

    PARAMETERS
      - test_enkf_yo: an initialised EnsembleKalmanFilter object
    '''

    # Make a copy of the EnsembleKalmanFilter object so that if this test makes
    # changes to it, the changes wont disrupt other tests
    enkf1 = copy.deepcopy(test_enkf_yo)
    enkf2 = copy.deepcopy(test_enkf_yo)

    # XXXX Now test the 'predict' function!
    # The function does the following:
    #for i in range(enkf.ensemble_size):
    #    enkf.models[i].step()
    # so in this test you could step enkf1 a few times manually and then
    # use 'predict' to step enkf2 a few times and check they're the same.
    # Or, probably better, use a small, manageable enkf and check, by hands,
    # that it steps as expected when the predict function is called.

    # For now lets just pass this test
    assert True

def test_nonsense():
    # This test will fail, just so you can see what it looks like
    assert False
