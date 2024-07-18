# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:47:43 2023

@author: earyo
"""
# Imports
import os
import warnings as warns
import numpy as np
import pandas as pd
from plot_bivariate_distr import *  
#from scipy.stats import powerlognorm
import seaborn as sns
import matplotlib.pyplot as plt
#from matplotlib.ticker import LogFormatterSciNotation
import matplotlib.ticker as mticker


class EnsembleKalmanFilter:
    """
    A class to represent a EnKF for application with a wealth 
    agent-based model for the United States.
    """
    def __init__(self, model, filter_params, model_params, constant_a, filter_freq):
        """
        Initialise the Ensemble Kalman Filter.
        
        PARAMETERS:
        - ensemble_size:                      Number of models used in EnKf
        - macro_state_vector_length:          Corresponds to Number of aggregate wealth groups
        - micro_state_vector_length:          Corresponds to Number of agents
        - population_size:                    Number of agents
        - macro_history:                      Records of wealth group evolution
        - models:                             The actual agent-based model (objects)
        - macro_state_ensemble                Stores all wealth group data over all
                                              distinct ensemble runs
        - micro_state_ensemble                Stores all agents over all ensemble runs
        - update_decision                     Binary var recording whether EnKF update
                                              happens or not
        - H                                   EnKF observation operator 
                                              (transforms obs. shape into model prediction shape)
        - time                                 /
        - other Kalman params.                 /                                                 
        """
        
        self.modelclass = str(model)
        # Add a new instance attribute for constant 'a'
        self.constant_a = constant_a
        self.ensemble_size = None
        self.macro_state_vector_length = None
        self.micro_state_vector_length = None
        self.population_size = None
        self.filter_frequency = filter_freq
        # Get filter attributes from params, warn if unexpected attribute
        for k, v in filter_params.items():
             if not hasattr(self, k):
                 w = 'EnKF received unexpected {0} attribute.'.format(k) 
                 warns.warn(w, RuntimeWarning)
             setattr(self, k, v)

        self.macro_history_share = list()
        self.micro_history = list()
        self.error_history = list()
        
        #record history of diff eigenvalues
        self.eigenvalues_diff_history = list()
        
        # Set up ensemble of models and other global properties
        self.population_size = model_params["population_size"]  
        ### set up storage for data history. Macro-history consists of 4 groups
        ### so thus it is a list with four elements which will be arrays that 
        ### increase their size with time. 
        ### but control flow for if there are < 100 agents
        macro_hist_shape = np.zeros(shape=(self.ensemble_size,1))
        if self.population_size >= 100:
            self.macro_history = [macro_hist_shape,
                                  macro_hist_shape,
                                  macro_hist_shape,
                                  macro_hist_shape]
        else:
            ## there are only 3 wealth groups with < 100 agents
            self.macro_history = [macro_hist_shape,
                                  macro_hist_shape,
                                  macro_hist_shape]

        self.models = [model(**model_params) for _ in range(self.ensemble_size)]
        shape_macro = (self.macro_state_vector_length, self.ensemble_size)
        shape_micro = (self.micro_state_vector_length, self.ensemble_size)
        self.macro_state_ensemble = np.zeros(shape=shape_macro) ## per adult wealth per wealth group
        self.micro_state_ensemble = np.zeros(shape=shape_micro)
        ## fill variable to record previous state estimate sin case of update
        self.micro_state_ensemble_old = None
        self.macro_state_ensemble_old = None
        ## var to pass on to other methods 
        ## for decision making
        self.update_decision = None
        #### Observation matrix = translation matrix between macro and micro
        #### states
        self.H = self.make_H(self.micro_state_vector_length, 
                             self.macro_state_vector_length).T
        self.ensemble_covariance = None
        self.data_ensemble = None 
        self.data_ensemble_history = list() ### need this to track and plot observation
        self.data_ensemble_history_average = list() ### need this to track and plot observation mean
        self.current_obs_history = list()
        self.current_obs_var_history = list()
        self.data_covariance = None
        self.Kalman_Gain = None
        self.state_mean = None
        self.time = 0 
        ### load observation data from desired start year (y)
        ### LOAD empirical monthly wealth Data sorted by group
        ### for state vector check
        path = ".."
        with open(os.path.join(path, 'data', 'wealth_data_for_import2.csv')) as f2:
            self.data = pd.read_csv(f2, encoding = 'unicode_escape')        
        y = model_params["start_year"]
        self.idx_begin = min((self.data[self.data["year"]==y].index.values))
        self.obs = self.data.iloc[self.idx_begin::][["year",
                                                     "month",
                                                     "date_short",
                                                     "group",
                                                     "real_wealth_share",
                                                     "real_wealth_per_unit",
                                                     "variance_real_wealth"]]
        

        # Set negative values in 'real_wealth_per_unit' to 1
        #self.obs['real_wealth_per_unit'] = self.obs['real_wealth_per_unit'].apply(lambda x: 1 if x < 0 else x)

        # Set negative values in 'real_wealth_share' to 0.005
        #self.obs['real_wealth_share'] = self.obs['real_wealth_share'].apply(lambda x: 0.005 if x < 0 else x) 
        
        # Update the 'variance_real_wealth' column
        self.update_variance_real_wealth(self.constant_a)
        
    def update_variance_real_wealth(self, a):
       """
       Update the 'variance_real_wealth' column in the dataframe 'self.data'
       using the specified constant 'a' and the formula (a * real_wealth_per_unit)^2.
       """
       if 'real_wealth_per_unit' in self.data:
           self.obs['variance_real_wealth'] = (a * self.data['real_wealth_per_unit']) ** 2
       else:
           raise ValueError("The dataframe does not have the required 'real_wealth_per_unit' column.")
    
    def predict(self):
        """
        Step the model forward by one time-step to produce a prediction.
        Params:
        Returns:
            None
        """
        for i in range(self.ensemble_size):
            self.models[i].step()
            ## print("Model ensemble member", i, "has stepped")
            ## print("this is the macro state of this member", self.models[i].macro_state)
            ## print("ths is the micro state of this member", self.models[i].micro_state)
        
    def set_current_obs(self):
        """
        Here we set the current observation corresponding to the time
        in the models as well as the variance of the current observation.
        This is used in the method update_data_ensemble"""
        
        ## control for population size = number of agents
        if self.population_size >= 100:
            self.current_obs = self.obs.iloc[self.time*4-4:self.time*4, 5]
            self.current_obs_var = self.obs.iloc[self.time*4-4:self.time*4, 6]
            self.current_obs_history.append(self.current_obs)
            self.current_obs_var_history.append(self.current_obs_var)
        else:
            ### sub set only top 10% idx 4-3 excludes only the top1%
            self.current_obs = self.obs.iloc[self.time*4-3:self.time*4, 5]
            self.current_obs_var = self.obs.iloc[self.time*4-3:self.time*4, 6]
            self.current_obs_history.append(self.current_obs)
            self.current_obs_var_history.append(self.current_obs_var)
            
            
    def update_state_ensemble(self):
        """
        Update self.state_ensemble based on the states of the models.
        Which records the model state as average wealth per adult per wealth group.
        """
        for i in range(self.ensemble_size):
            self.macro_state_ensemble[:, i] = self.models[i].macro_state
            self.micro_state_ensemble[:, i] = self.models[i].micro_state
            ### print("this is the macro state of model ensemble", i,  self.macro_state_ensemble[:, i])
            ### print("this is the micro state of model ensemble", i,  self.micro_state_ensemble[:, i])
            
    def update_state_mean(self):
        """
        Update self.state_mean based on the current state ensemble.
        """
        self.state_mean = np.mean(self.macro_state_ensemble, axis=1)
        
    def make_ensemble_covariance(self):
        """
        Create ensemble covariance matrix.
        Rowvar determines that the rows are the variables which is true since 
        rows represent wealth groups and columns different observations thereof.
        """
        self.ensemble_covariance = np.cov(self.macro_state_ensemble, rowvar = True)
      
    def make_data_covariance(self):
        """
        Create data covariance matrix which assumes no direct correlation between 
        data time series.
        """
        self.data_covariance = np.diag(self.current_obs_var)

    def make_H(self, dim_micro_state, dim_data):
        
        '''This method creates the observation operator. It constructs a matrix that 
        transforms the microstate of a model into the macrostate. The micro state is just
        the ordered (from top to down) wealth of agents and the macrostate the 
        average wealth per top 1%, top10%, next40%, bottom50%. Therefore it is 
        a matrix that sums the normalized agent wealth values so that they yield
        the average per group. This means the normalization constant is just 1/k
        where k is the number of agents per group.'''
        
        ## check whether there are at least 10 agents    
        assert dim_micro_state >= 10, "agent quantity cannot be less than 10" # denominator can't be 0
        ## set overall dimensions
        
        H = np.zeros((dim_micro_state, dim_data))
        L = self.micro_state_vector_length
        ## check whether there are at least 100 agents
        #print("this is dim micro", dim_micro_state)
        if dim_micro_state >= 100:
            H[:int(round(L*0.01)),0] = 1/(L*0.01) ### H entries are normalized
            H[int(round(L*0.01)):int(round(L*0.1)),1] = 1/(L*0.09) #next 9%
            H[int(round((L*0.1))):int(round(L*0.5)),2] = 1/(L*0.4) # next 40%
            H[int(round(L*0.5)):,3] = 1/(L*0.5) #bottom 50%
        else:
            H[:int(round(L*0.1)),0] = 1/(L*0.1) #top 10%
            H[int(round(L*0.1)):int(round(L*0.5)),1] = 1/(L*0.4) # next 40%
            H[int(round(L*0.5)):,2] = 1/(L*0.5) # bottom 50%
            
        return H
    
    def update_data_ensemble(self):
        """
        Create perturbed data vector.
        I.e. a replicate of the data vector plus normal random n-d vector.
        R - data (co?)variance; this should be either a number or a vector with
        same length as the data. Second parameter in np.random-normal 
        is standard deviation. Be careful to take square root when
        passing the obs variance.
        
        Also, as 2nd task, track history of data ensemble mean i.e.
        the history of wealth shares given by the 
        data ensemble/uncertainty (which is different from the actual given data)
        due to stochasticity.
        """
        
        
        ''' original Keiran
        x = np.zeros(shape=(len(data), self.ensemble_size))
        for i in range(self.ensemble_size):
            x[:, i] = data + np.random.normal(0, self.R_vector, len(data))
        self.data_ensemble = x
        '''
        
        ### 1ST TASK
        x = np.zeros(shape=(len(self.current_obs), self.ensemble_size)) 
        for i in range(self.ensemble_size):
            err = np.random.normal(0, np.sqrt(self.current_obs_var), len(self.current_obs))
            x[:, i] = self.current_obs + err
        self.data_ensemble = x   
       
        
        ### 2ND TASK
        ### track history of computed data ensemble average and of entire data_ensemble
        
        ''' TO DO track entire data ensemble !!! '''
        
        r = np.mean(self.data_ensemble, 1)[:,None] ## r for intermediate result
        
        #print("this is data_ensemble in enkf", self.data_ensemble)
        
        p = self.population_size
        if p >= 100:
            pop = np.array([0.01*p,0.09*p,0.4*p,0.5*p])[:, None]
            r2 = np.multiply(r,pop)
            d = np.where(r2 > 0)
            
            #print("this is r2 in enkf", r2)
            
            r3 = r2 / np.sum(r2[d])
            self.data_ensemble_history_average.append(r3)
            self.data_ensemble_history.append(self.data_ensemble)
        else:
            pop = np.array([0.1*p,0.4*p,0.5*p])[:, None]
            r2 = np.multiply(r,pop)
            d = np.where(r2 > 0)
            r3 = r2 / np.sum(r2[d])
            self.data_ensemble_history_average.append(r3)
            self.data_ensemble_history.append(self.data_ensemble)

        
    def make_gain_matrix(self):
        """
        Create kalman gain matrix.
        Should be a (n x 4) matrix since in the state update equation we have
        the n-dim vector (because of n-agents) + the update term which 4 dimensional
        so the Kalman Gain needs to make it (n x 1)
        micro_state_ensemble should be num_agents x ensemble_size 
        """
        #### here the control sequence is implemented to control for ensemble size = 1
        #### the np.cov fct. then does not correctly interpret the micro_state_vector
        #### hence in tht case the ensemble size covariance matrix will be 0
        #### as it should be because there is no sample covariance with a sample of 1
        if self.ensemble_size == 1:
            ### squeeze array to get rid of the added column dimension
            help_array = np.concatenate((self.micro_state_ensemble, self.micro_state_ensemble),1)
            C = np.cov(help_array)    
        else:
            C = np.cov(self.micro_state_ensemble)
        state_covariance = self.H @ C @ self.H.T
        eigenvalues_state_covariance = np.linalg.eigvals(state_covariance)
        eigenvalues_data_covariance = np.linalg.eigvals(self.data_covariance)
        #if self.update_decision == True:
           #print("this is state_covariance eigenvalues", eigenvalues_state_covariance)
           #print("this is data_covariance eigenvalues", eigenvalues_data_covariance)
       
        #max_eigenvalue = np.max(eigenvalues)
        #scaled_covariance = state_covariance / max_eigenvalue
        diff = state_covariance + self.data_covariance
        if self.update_decision == True:
            #print("this is diff eigenvalues", np.linalg.eigvals(diff))
            self.eigenvalues_diff_history.append(np.linalg.eigvals(diff))
        self.Kalman_Gain = C @ self.H.T @ np.linalg.inv(diff)

        
    
        '''
        Keiran version original
        C = np.cov(self.state_ensemble)
        state_covariance = self.H @ C @ self.H_transpose
        diff = state_covariance + self.data_covariance
        return C @ self.H_transpose @ np.linalg.inv(diff)
        '''
        
        
    def state_update(self):
        """
        Update system state of model. This is the state update equation of the 
        Kalman Filter. Here the decisive step then is that the difference is 
        calculated between self.data_ensemble (4-dimensional) and self.micro_state_ensemble
        (n-dimensional) so there the observation matrix H
        does a translation between micro and macro state. And 
        we update the micro-level state vector which is n-
        dimensional based on n-agents.
        """
        ## save previous system state estimate before updating
        self.micro_state_ensemble_old = self.micro_state_ensemble
        self.macro_state_ensemble_old = self.macro_state_ensemble
        
        ### start update
        #X = np.zeros(shape=(self.micro_state_vector_length, self.ensemble_size))
        #Y = np.zeros(shape=(self.macro_state_vector_length, self.ensemble_size))
        #for i in range(self.ensemble_size):
         #   diff = self.data_ensemble[:, i] - self.H @ self.micro_state_ensemble[:, i] 
          #  X[:, i] = self.micro_state_ensemble[:, i] + self.Kalman_Gain @ diff
           # Y[:, i] =  self.H @ X[:, i]
        
        diff = self.data_ensemble - self.H @ self.micro_state_ensemble
        X = self.micro_state_ensemble + self.Kalman_Gain @ diff
        #print("this is X", X)
        ### error in model 2 definitely stems from update here and is about values being smaller than 0 because it 
        #### disappear if this is introduced
        #### not sure that is a good solutions though
       
        Y = self.H @ X

        # print("this is unaltered X", X)
        
        #X[X < 0] = 0

        self.micro_state_ensemble = X
        self.macro_state_ensemble = Y
       
    def update_models(self):
        """
        Update individual model states based on state ensemble.
        """
        for i in range(self.ensemble_size):
            self.models[i].micro_state = self.micro_state_ensemble[:, i]
            self.models[i].macro_state = self.macro_state_ensemble[:, i]
            self.models[i].update_agent_states()
            assert self.models[i].agents[0].wealth == self.micro_state_ensemble[:, i][0], (self.models[i].agents[0].wealth, self.micro_state_ensemble[:, i][0]) 

        ##### HERE EACH MODEL ECONOMY NEEDS TO UPDATE ITS OWN INTERNAL AGENT
        ##### STATES 

    def plot_macro_state(self, log_var: bool):
        
        '''This method plots the macro state of economy. That is the system state
        estimate at a macro/aggregate level plus the observation for the current time step.
        If the EnKF update step is actually conducted, this function plots three
        different bivariate probability distributions in one plot: 1) previous
        system estimate 2) observation including their uncertainty 3) new estimate'''
        
        if not isinstance(log_var, bool):
            raise TypeError
        
        ## set data dimensions 
        dim = len(self.macro_state_ensemble)-1
        ####################################
        ### prepare system macro-state estimate
        ####################################
        
        if log_var == True:
            x = np.log(self.macro_state_ensemble[0,:])
            y = np.log(self.macro_state_ensemble[dim,:])
            # Set up grid points for plotting later fed into meshgrid
            x_grid = np.linspace(min(x)*0.9, max(x)*1.1, 100)
            y_grid = np.linspace(min(y)*0.9, max(y)*1.1, 100)
        elif log_var == False:
            x = self.macro_state_ensemble[0,:]
            y = self.macro_state_ensemble[dim,:]
            x_grid = np.linspace(0, max(x)*1.1, 100)
            y_grid = np.linspace(0, max(y)*1.1, 100)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_hat = np.matrix([x_mean, y_mean]).T
        C = np.cov(x,y)
        # Set up grid for plotting
        X, Y = np.meshgrid(x_grid, y_grid)
        varname1 = "ln($ Wealth per adult Top 1%)"
        varname2 = "ln($ Weatlh per adult Bottom 50%)"
        
        ##########################################
        ### prepare observation also on the same plot
        ##########################################
        if log_var == True:
            x2 = np.log(self.data_ensemble[0,:])
            y2 = np.log(self.data_ensemble[dim,:])
            x_grid2 = np.linspace(min(x2)*0.9, max(x2)*1.1, 100)
            y_grid2 = np.linspace(min(y2)*0.9, max(y2)*1.1, 100)
        elif log_var == False: 
            x2 = self.data_ensemble[0,:]
            y2 = self.data_ensemble[dim,:]
            x_grid2 = np.linspace(0, max(x2)*1.1, 100)
            y_grid2 = np.linspace(0, max(y2)*1.1, 100)
        x_mean2 = np.mean(x2)
        y_mean2 = np.mean(y2)
        x_hat2 = np.matrix([x_mean2, y_mean2]).T
        C2 = np.cov(x2,y2)
        X2, Y2 = np.meshgrid(x_grid2, y_grid2)
        varname1 = "ln($ Wealth per adult Top 1%)"
        varname2 = "ln($ Weatlh per adult Bottom 50%)"
        
        ##########################################
        ### prepare old state estimate on the same plot
        ### in case the time step included an EnKF update step
        ##########################################
        if self.update_decision == True: 
            if log_var == True:
                x3 = np.log(self.macro_state_ensemble_old[0,:])
                y3 = np.log(self.macro_state_ensemble_old[dim,:])
                x_grid3 = np.linspace(min(x3)*0.9, max(x3)*1.1, 100)
                y_grid3 = np.linspace(min(y3)*0.9, max(y3)*1.1, 100)
            elif log_var == False:
                x3 = self.macro_state_ensemble_old[0,:]
                y3 = self.macro_state_ensemble_old[dim,:]
                x_grid3= np.linspace(0, max(x3)*1.1, 100)
                y_grid3 = np.linspace(0, max(y3)*1.1, 100)

            x3_mean = np.mean(x3)
            y3_mean = np.mean(y3)
            x3_hat = np.matrix([x3_mean, y3_mean]).T
            C3 = np.cov(x3,y3)
            # Set up grid for plotting
            X3, Y3 = np.meshgrid(x_grid3, y_grid3)
            #varname1 = "ln($ Wealth per adult Top 1%)"
            #varname2 = "ln($ Weatlh per adult Bottom 50%)"
    
        ###########################
        ####### ACTUAL PLOT #######
        ###########################
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid()
        
        
        ########### NECESSARY CONTROL FLOW FOR different plot during update step
        ########## MAKE NICER

        #### SYSTEM ESTIMATE
        if self.update_decision == False: 
            Z = gen_gaussian_plot_vals(x_hat, C, X, Y)
            ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.coolwarm)
            cs = ax.contour(X, Y, Z, 6, colors="black")
        else: 
            Z = gen_gaussian_plot_vals(x_hat, C, X3, Y3)
            ax.contourf(X3, Y3, Z, 6, alpha=0.6, cmap=cm.coolwarm)
            cs = ax.contour(X3, Y3, Z, 6, colors="black")
            
            
        #### OBSERVATION 
        ### obs needs to be plotted on same grid for comparison, hence X and Y
        Z2 = gen_gaussian_plot_vals(x_hat2, C2, X, Y)
        #if log_var == "yes":
        Z2[Z2<np.mean(Z2)/100] = 0
        ax.contour(X, Y, Z2, 4, colors="black", linestyles = '--')
        
        ### **PREVIOUS** System estimate
        #### old estimate plot if included
        u = self.update_decision
        v = self.macro_state_ensemble_old
        if u == True and not v is None:
            Z3 = gen_gaussian_plot_vals(x3_hat, C3, X3, Y3)
            #if log_var == "yes":
            Z3[Z3<np.mean(Z3)/100] = 0
            ax.contour(X3, Y3, Z3, 6, colors="black", linestyles = 'dotted')
            
                
        #ax.contourf(X, Y, Z2, 6, alpha=0.6)
        #Contour levels are a probability density
        ax.clabel(cs,levels = cs.levels, inline=1, fontsize=10)
        ax.set_xlabel(varname1, fontsize = 14)
        ax.set_ylabel(varname2, fontsize = 14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, pos: f'{x/10**np.floor(np.log10(x)):.1f} $\\times 10^{int(np.floor(np.log10(x)))}$' if x > 0 else '0'
            ))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, pos: f'{x/10**np.floor(np.log10(x)):.1f} $\\times 10^{int(np.floor(np.log10(x)))}$' if x > 0 else '0'
            ))
   
        plt.show()
        
    def plot_micro_state(self):
        
        """
        Function estimates current micro state of models and the ensemble of model
        estimates is an estimate for the EnKF micro state. This function estimates
        the probability of wealth across agents. For this purpose it applies some 
        Kernel Density estimation techniques.It has to do so on the log-transform of wealth
        (base e, so natural log) because wealth is per definition in model 
        power-lognormally distributed.The Gaussian, or any other, Kernel does not work well
        for visualization purpose on this kind of distribution if untreated.
        https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/ 
        """
        
        fig, ax = plt.subplots()
        ax.set_xlabel("ln(wealth) per agent")
        colors = sns.color_palette("viridis", n_colors=self.ensemble_size)
        for i in range(self.ensemble_size):
            distr = np.log(self.micro_state_ensemble[:,i])
            sns.kdeplot(data=distr, fill=False, alpha = 0.5, color = colors[i], ax = ax)
        mean_log = np.log(np.mean(self.micro_state_ensemble, axis = 1))
        sns.kdeplot(data=mean_log, fill=False, alpha = 1, color = "black", lw = 3, 
                    linestyle = "--", ax = ax, label = "Mean of microstates")
        ax.legend(frameon = False)
        plt.show()
        
    def record_history(self):
        ''' saves data over time '''
        
        for count, value in enumerate(self.macro_history):
            x = np.expand_dims(self.macro_state_ensemble[count,:],1)
            self.macro_history[count] = np.concatenate((value, x), axis = 1)
        self.micro_history.append(self.micro_state_ensemble)
        
        
    def make_macro_history_share(self):
        
        A = None ## placeholder for macro_history as wealth shares
        ### PLOT empirical monthly wealth Data vs model output for chosen time-frame
        colors = ["tab:red", "tab:blue", "grey", "y"]
        
        ## control for population size
    
        if self.population_size >= 100:
            wealth_groups = ["Top 1%", "Top 10%-1%", "Middle 40%", "Bottom 50%"]
            
            multipliers = [int(0.01*self.population_size),
                            int(0.09*self.population_size),
                            int(0.4*self.population_size),
                            int(0.5*self.population_size)]
        else:
            wealth_groups = ["Top 10%", "Middle 40%", "Bottom 50%"]
            multipliers = [int(0.1*self.population_size),
                                int(0.4*self.population_size),
                                int(0.5*self.population_size)]
            
        #### compute total_wealth time series
        total_wealth_ts = np.zeros(shape=(self.ensemble_size, self.time))
        
    
        for i in range(len(multipliers)):
            ### here we make the total wealth calculation. ## needs to be flexible
            ### for different size populations
            m = self.macro_history[i][:,1:]
            n = multipliers[i]
            total_wealth_ts += np.multiply(m,n)  
            #print("this is total_wealth_ts", total_wealth_ts)
        
        for i in range(len(multipliers)):
            m = self.macro_history[i][:,1:]
            n = multipliers[i]
            q = np.multiply(m, n)
            p = total_wealth_ts
            A = np.divide(q, p) 
            #print('this is A', A)
            self.macro_history_share.append(A)
        
    def plot_fanchart(self, ax): 
        
        '''make fanchart of model runs over wealth share by group
        until up to time point where the filter/model is applied to.'''
        
        A = None ## placeholder for macro_history as wealth shares
        ### PLOT empirical monthly wealth Data vs model output for chosen time-frame
        colors = ["tab:red", "tab:blue", "grey", "y"]
        
        ## control for population size
    
        if self.population_size >= 100:
            wealth_groups = ["Top 1%", "Top 10%-1%", "Middle 40%", "Bottom 50%"]
            
            multipliers = [int(0.01*self.population_size),
                            int(0.09*self.population_size),
                            int(0.4*self.population_size),
                            int(0.5*self.population_size)]
        else:
            wealth_groups = ["Top 10%", "Middle 40%", "Bottom 50%"]
            multipliers = [int(0.1*self.population_size),
                                int(0.4*self.population_size),
                                int(0.5*self.population_size)]
            
        #### compute total_wealth time series
        total_wealth_ts = np.zeros(shape=(self.ensemble_size, self.time))
        
    
        for i in range(len(multipliers)):
            ### here we make the total wealth calculation. ## needs to be flexible
            ### for different size populations
            m = self.macro_history[i][:,1:]
            n = multipliers[i]
            total_wealth_ts += np.multiply(m,n)  
            #print("this is total_wealth_ts", total_wealth_ts)
        
        for i in range(len(multipliers)):
            m = self.macro_history[i][:,1:]
            n = multipliers[i]
            q = np.multiply(m, n)
            p = total_wealth_ts
            A = np.divide(q, p) 
            #print('this is A', A)
            self.macro_history_share.append(A)
        
        #### Make 4 arrays for all time steps so far for all of the 4 wealth
        #### groups 
        ######## NEED TO RECORD MICROHISTORY AS WELL ###
        #fig, ax = plt.subplots(figsize = (8,6))
        ### x needs to be set just once
        ### to the correct length
        L = np.shape(self.macro_history_share[0][:,1:])[1]
        x = self.obs["date_short"][::4].reset_index(drop = True).iloc[:L]
        for i in range(len(multipliers)):
            arr = self.macro_history_share[i][:,1:]  ## without the first column
            # for the median use `np.median` and change the legend below
            mean = np.mean(arr, axis=0)
            offsets = (25,67/2,47.5)
            ax.plot(x, mean, color=colors[i], lw=3)
            for offset in offsets:
                low = np.percentile(arr, 50-offset, axis=0)
                high = np.percentile(arr, 50+offset, axis=0)
                # since `offset` will never be bigger than 50, do 55-offset so that
                # even for the whole range of the graph the fanchart is visible
                alpha = (55 - offset) / 100
                ax.fill_between(x, low, high, color=colors[i], alpha=alpha)
            
        #### Plot data ensemble history and actual observations
        h = np.array(self.data_ensemble_history_average)
        for i, g in enumerate(wealth_groups):
            ### y variables
            y = h[:,i]
            S = self.obs["real_wealth_share"][self.obs["group"] == g].reset_index(drop = True)
            l = S.iloc[:L]
            ### x variable
            T = self.obs["date_short"][self.obs["group"] == g].reset_index(drop = True)
            x = T.iloc[:L]
            ### plot data ensemble history
            #ax.plot(x,y[1:], label = g, color = colors[i], linestyle = '--')
            ### plot actual observations
            ax.plot(x,l, label = g, color = colors[i], linestyle = 'dotted')
        '''
               for i in range(len(self.data_ensemble_history_average)):
                   if i % self.filter_frequency != 0 or i == 0:
                       pass
                   else:
                       for data_point in self.data_ensemble_history_average[i]:
                           ax.scatter(i, data_point, marker='x',  color='black', s=100)
                  
          ''' 
        #ax.set_xlabel("time")
        ax.set_ylabel("wealth share")
        ax.set_ylim((-0.05,1))
        ax.set_xticks(x.iloc[0::20].index)
        ax.set_xticklabels(x.iloc[0::20], rotation = 90)
        legend_items = ["Top 1%","__ci1","__ci2","__ci3", 
                        "Top 10%-1%","__ci4","__ci5","__ci6",
                        "Middle 40%", "__ci7","__ci8","__ci9",
                        "Bottom 50%", "__ci10","__ci11","__ci12"]
        #ax.legend([f'{o}' for o in legend_items],
         #         frameon = False, bbox_to_anchor = (1.25, 0.6), loc='center right')
        ax.margins(x=0)
        
        
    def post_update_difference(self):
        
        ### set up data collection which collects all differences between model state
        ### and the data points of the four wwealth groups post up date step
        ### we need this analysis/data in order to check whether the enkf works correctly
        ### in the sense that if variance in model is high the, the enkf should emphasize the data 
        ### and if variance in data is high it should emphasize the model
        
        sum_diff_model_data_list = []
        
        for i in range(len(self.data_ensemble_history_average)):
            
            ### if not filter step do not do anything
            if i % self.filter_frequency != 0 or i == 0:
                pass
            else: ### if filter step start looping over four data ensembles wealth groups for and measure their difference to model ensemble 
                sum_diff_model_data = 0
                for idx, data_point in enumerate(self.data_ensemble_history_average[i]):
                    #print("this is np.mean(self.macro_history_share[idx][:,i]", np.mean(self.macro_history_share[idx][:,i]))
                    #print("this data_point", data_point)
                    diff_model_data_point = abs(np.mean(self.macro_history_share[idx][:,i]) - data_point)
                    sum_diff_model_data += diff_model_data_point
                sum_diff_model_data_list.append(sum_diff_model_data)
        
        ### now sum all sums of differences (first sum across the four wealth groups) across time
        #print("sum_diff_model_data_list", sum_diff_model_data_list)
        total_sum = sum(sum_diff_model_data_list)
        #print("this is total sum", total_sum)
        return total_sum
                 

    def quantify_error(self, model_output, data_vector):
            
            """
            Compute the error metric as the average absolute distance between 
            the model output and the data vector, averaged across all ensemble members 
            and the four wealth groups.
    
            :param model_output: 2D list or numpy array of shape [n, 4]
                where n is the number of ensemble members.
            :param data_vector: 1D list or numpy array of shape [4]
            :return: float, the error metric
            """
            
            # Convert to numpy arrays for easier calculations
            model_output = np.array(model_output)
            data_vector = np.array(data_vector)
            
            # Ensure dimensions are correct
            assert model_output.shape[1] == len(self.macro_history), f"Model output should have shape [n, {len(self.macro_history)}]"
            assert len(data_vector) == len(self.macro_history), "Data vector should have shape [{len(self.macro_history)}]"
            
            # Calculate absolute differences between the model output and data vector
            abs_diffs = np.abs(model_output - data_vector)
            
            
            # sum differences across four wealth groups as in equation 6 of the paper first summation sign
            # second summation sign and average is over ensemble runs and done in compute error 
            # sum differences across four wealth groups as in equation 6 of the paper
            abs_diffs_sum = np.sum(abs_diffs, axis = 1)
        
            # Return the average absolute difference as well as the error per group
            return abs_diffs, abs_diffs.mean(), abs_diffs_sum
    
    def record_error(self):
        """
        Record the error metric properly applying def quantify_error.
        Some data transformation has to be conducted beforehand.
        """
        ### population numbers per wealth group
        if self.population_size >= 100:
            multipliers = np.array([
                            int(0.01*self.population_size),
                            int(0.09*self.population_size),
                            int(0.4*self.population_size),
                            int(0.5*self.population_size)],)
        else:
            multipliers = np.array([
                            int(0.1*self.population_size),
                            int(0.4*self.population_size),
                            int(0.5*self.population_size)],)
            
        
        ### OBSERVATION TRANSFORMATION DATA FORMAT
        y = np.sum(np.multiply(self.current_obs,multipliers))
        x =  np.multiply(self.current_obs,multipliers)
        share_obs = x/y
    
        ### MODEL TRANSFORMATION DATA 
        ### calculate total wealth over population numbers
        a1 = (self.macro_state_ensemble.T * multipliers).T
        ### calculate sum of wealth across wealth groups for all ensemble members
        a2 = np.sum(a1, 0)
        assert a1.shape[0] == multipliers.shape[0]
        ### expand array 2
        a3 = np.tile(a2, (multipliers.shape[0], 1))
        ### calculate wealth shares
        a4 = np.divide(a1, a3)
        current_error = self.quantify_error(a4.T, share_obs)
        self.error_history.append(current_error[2])
        
        
        
    def plot_error(self, ax):
        
        ''' this function plots the error over time which is defined as the 
        difference between "ground truth" and model outputs on average 
        across all 4 wealth groups and per single wealth group'''
        
        
        #fig, ax = plt.subplots(figsize=(10,4))
        L = np.shape(self.macro_history_share[0][:,1:])[1]
        x = self.obs["date_short"][::4].reset_index(drop = True).iloc[:L]
        ### here the np.mean is again the second summation in eq. 6 and the averaging
        print("this is model class", self.modelclass)
        if self.modelclass == "<class 'model1_class.Model1'>":
            ax.plot(x, np.mean(np.array(self.error_history),axis=1)[1:], label = "Model 1")
        else:
            ax.plot(x, np.mean(np.array(self.error_history),axis=1)[1:], label = "Model 2")
        ax.set_xticks(x.iloc[0::20].index)
        ax.set_xticklabels(x.iloc[0::20], rotation = 90)
        ax.set_ylabel("error")

        #my_array = np.concatenate((x, np.array(self.error_history)[1:]), axis = 1)
        
        # Create a new DataFrame
        #df = pd.DataFrame({
         #   'Date': x,
          #  'Error': self.error_history[1:]
        #})
        
        #df = pd.DataFrame(my_array)
        
        #df.to_csv('error_model1.csv', index=False)
        ### save error history data
        
        
    def integral_error(self):
        
        
        #if you look at figure 2 or 3 here we sum the integral under curves in panel e 
    
        return np.sum(np.mean(np.array(self.error_history), axis = 1))
    

    def step(self, update: bool):
        
        if not isinstance(update, bool):
            raise TypeError
        
        self.update_decision = update ## var to pass on to other methods 
        
                             ##>>>??? for decision making
        self.predict()
        self.time = self.time + 1
        self.set_current_obs()
        self.update_state_ensemble()
        self.update_state_mean()
        self.update_data_ensemble()
        self.make_ensemble_covariance()
        self.make_data_covariance()
        self.make_gain_matrix()
        self.record_history()
        self.record_error()
        #self.record_error()
        if update == True: 
            self.state_update()
        #### plot of the macro_state needs to come after state_update() so that
        #### it either includes all 3 (previous, new, observation) state estimates   
        #### or only the non-updated plus obsverational one
        #self.plot_macro_state(log_var = False)
        #self.plot_fanchart()
        #if update == True: 
            self.update_models()
            