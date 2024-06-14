import numpy as np 


def uniform_sample(sample_size):

    """This function generates a sample of size sample_size from a uniform distribution. 
    Can be used to represent agents later on in the model."""

    u = np.random.uniform(0, 1, sample_size)
    # sort u in ascending order
    u = np.sort(u)

    return u 


def map_percentiles_weights(percentiles_given, lower_bound, upper_bound):

    """This function create the weights for the weighted average probability distribution as a mix of 
    exponential and Pareto distributions according to eq. 9 in Vallejos et al. (2018) """

    # check percentiles_given type, if not numpy array convert to numpy array
    if type(percentiles_given) != np.ndarray:
        percentiles_given = np.array(percentiles_given)

    #print("percentiles_given", percentiles_given)

    # create weights according to eq. 9 in Vallejos et al. (2018)
    normalization_constant = upper_bound - lower_bound
    weights = (percentiles_given - lower_bound) / normalization_constant

    # for weights smaller than lower bound of percentiles at the correct index of percentiles, set to zero
    weights = np.where(percentiles_given < lower_bound, 0, weights)
    # for weights larger than upper bound of percentiles given, set to one
    weights = np.where(percentiles_given > upper_bound, 1, weights)

    # Ensure arrays are at least 1-dimensional
    weights = np.atleast_1d(weights)
    percentiles_given = np.atleast_1d(percentiles_given)

    # make a dictionary mapping percentiles to weights
    percentiles_weights_dict = dict(zip(percentiles_given, weights))

    return percentiles_weights_dict


def weighted_avg_exp_pareto_distr(percentiles_given, lower_bound, upper_bound, alpha = 1.3, Temperature = 5):

    """This function calculates the weighted average of the exponential and Pareto distributions as
    new wealth distribution according to Vallejos et al. (2018)"""

    #based on Vallejos et al. (2018) and compare the
    # results to the actual data. J Econ Interact Coord (2018) 13:641–656 https://doi.org/10.1007/s11403-017-0200-9

    # calculate the weights
    lower_bound = lower_bound # pass from fct. to fct. to make it more general
    upper_bound = upper_bound
    percentiles_weights_dict = map_percentiles_weights(percentiles_given, lower_bound, upper_bound)
    #print("percentiles_weights_dict", percentiles_weights_dict)
    # extract the weights as a numpy array
    weights = np.array(list(percentiles_weights_dict.values()))
    #print("weights", weights)

    # other fixed parameters
    omega = 10
    c = 1

    # wealth of percentiles given
    exponential_part = -Temperature*(1-weights)*np.log((1-percentiles_given)/c)
    #print("this is exp", exponential_part)
    pareto_part = weights * omega * (1-percentiles_given)**(-1/alpha)
    #print("this is pareto", pareto_part)
    wealth_of_percentiles = exponential_part + pareto_part

    return wealth_of_percentiles
