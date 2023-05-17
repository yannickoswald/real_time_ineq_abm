# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:47:43 2023

@author: earyo
"""
# Imports
import warnings as warns
import numpy as np
import pandas as pd
from plot_bivariate_distr import *  
from scipy.stats import powerlognorm
import seaborn as sns
import matplotlib.pyplot as plt

# Classes
class EnsembleKalmanFilter:
    """
    A class to represent a EnKF for application with a wealth 
    agent-based model for the United States.
    """
    def __init__(self, model, filter_params, model_params):
        """
        Initialise the Ensemble Kalman Filter.
        Params:
            model
            filter_params
            model_params
        """
        
        self.ensemble_size = None
        self.macro_state_vector_length = None
        self.micro_state_vector_length = None
        
        # Get filter attributes from params, warn if unexpected attribute
        for k, v in filter_params.items():
             if not hasattr(self, k):
                 w = 'EnKF received unexpected {0} attribute.'.format(k) 
                 warns.warn(w, RuntimeWarning)
             setattr(self, k, v)
        
        #print(model, model_params)    
        # Set up ensemble of models and other global properties
        self.models = [model(**model_params) for _ in range(self.ensemble_size)]
        shape_macro = (self.macro_state_vector_length, self.ensemble_size)
        shape_micro = (self.micro_state_vector_length, self.ensemble_size)
        
        self.macro_state_ensemble = np.zeros(shape=shape_macro)
        self.micro_state_ensemble = np.zeros(shape=shape_micro)
        ## fill variable to record previous state estimate sin case of update
        self.micro_state_ensemble_old = None
        self.macro_state_ensemble_old = None
        ## var to pass on to other methods 
        ## for decision making
        self.update_decision = None
        #### Observation matrix = translation matrix between macro and micro
        #### states
        self.H = self.make_H(self.micro_state_vector_length, 4).T
        
        self.ensemble_covariance = None
        self.data_ensemble = None 
        self.data_covariance = None
        self.Kalman_Gain = None
        self.state_mean = None
        self.time = 0 
        
        ### load observation data
        ### LOAD empirical monthly wealth Data sorted by group
        ### for state vector check
        with open('./data/wealth_data_for_import2.csv') as f2:
            self.data = pd.read_csv(f2, encoding = 'unicode_escape')    
            
        y = model_params["start_year"]
        idx_begin = min((self.data[self.data["year"]==y].index.values))
        
        self.obs = self.data.iloc[idx_begin::][["year","month",
                                    "real_wealth_per_unit",
                                    "variance_real_wealth"]]
    
    def predict(self):
        """
        Step the model forward by one time-step to produce a prediction.
        Params:
        Returns:
            None
        """
        for i in range(self.ensemble_size):
            self.models[i].step()
        self.time = self.models[0].time 
        
    def set_current_obs(self):
        """
        Here we set the current observation corresponding to the time
        in the models as well as the variance of the current observation.
        This is used in the method update_data_ensemble"""
        
        self.current_obs = self.obs.iloc[self.time*4-4:self.time*4, 2]
        self.current_obs_var = self.obs.iloc[self.time*4-4:self.time*4, 3]
        
    def update_state_ensemble(self):
        """
        Update self.state_ensemble based on the states of the models.
        """
        
        for i in range(self.ensemble_size):
            self.macro_state_ensemble[:, i] = self.models[i].macro_state
            self.micro_state_ensemble[:, i] = self.models[i].micro_state
            
    def update_state_mean(self):
        """
        Update self.state_mean based on the current state ensemble.
        """
        self.state_mean = np.mean(self.macro_state_ensemble, axis=1)
        
    def make_ensemble_covariance(self):
        """
        Create ensemble covariance matrix.
        """
        self.ensemble_covariance = np.cov(self.macro_state_ensemble)
      
    def make_data_covariance(self):
        """
        Create data covariance matrix.
        """
        self.data_covariance = np.diag(self.current_obs_var)

    def make_H(self, dim_micro_state, dim_data):
        
        ''' This method creates the observation vector. It makes a matrix that 
        transform the microstate of a model into the macrostate. The micro state is just
        the ordered (from top to down) wealth of agents and the macrostate the 
        average wealth per top 1%, top10%, next40%, bottom50%. Therefore it is 
        a matrix that sums the normalized agent wealth values so that they yield
        the average per group. This means the normalization constant is just 1/k
        where k is the number of agents per group.'''
        
        H = np.zeros((dim_micro_state, dim_data))
        L = self.micro_state_vector_length
        H[:int(L*0.01),0] = 1/(L*0.01) ### H entries are normalized
        H[:int(L*0.1),1] = 1/(L*0.1)
        H[int(L*0.1):int(L*0.5),2] = 1/(L*0.4)
        H[int(L*0.5):,3] = 1/(L*0.5)
        return H
    
    def update_data_ensemble(self):
        """
        Create perturbed data vector.
        I.e. a replicate of the data vector plus normal random n-d vector.
        R - data (co?)variance; this should be either a number or a vector with
        same length as the data. Second parameter in np.random-normal 
        is standard deviation. Be careful to take square root when
        passing the obs variance.
        """
        x = np.zeros(shape=(len(self.current_obs), self.ensemble_size)) 
        for i in range(self.ensemble_size):
            err = np.random.normal(0, np.sqrt(self.current_obs_var), len(self.current_obs))
            x[:, i] = self.current_obs + err
        self.data_ensemble = x
        
    def make_gain_matrix(self):
        """
        Create kalman gain matrix.
        Should be a (n x 4) matrix since in the state update equation we have
        the n-dim vector (because of n-agents) + the update term which 4 dimensional
        so the Kalman Gain needs to make it (n x 1)
        """
        C = np.cov(self.micro_state_ensemble)
        state_covariance = self.H @ C @ self.H.T
        diff = state_covariance + self.data_covariance
        self.Kalman_Gain = C @ self.H.T @ np.linalg.inv(diff)
        
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
        X = np.zeros(shape=(self.micro_state_vector_length, self.ensemble_size))
        Y = np.zeros(shape=(self.macro_state_vector_length, self.ensemble_size))
        for i in range(self.ensemble_size):
            diff = self.data_ensemble[:, i] - self.H @ self.micro_state_ensemble[:, i]
            print('this is the diff ', diff)
            X[:, i] = self.micro_state_ensemble[:, i] + self.Kalman_Gain @ diff
            Y[:, i] =  self.H @ X[:, i]
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
            assert self.models[i].agents[0].wealth == self.micro_state_ensemble[:, i][0]
        ##### HERE EACH MODEL ECONOMY NEEDS TO UPDATE ITS OWN INTERNAL AGENT
        ##### STATES 

    def plot_macro_state(self, log_var: str):
        
        '''This method plots the macro state of economy. That is the system state
        estimate at a macro/aggregate level plus the observation for the current time step.
        If the EnKF update step is actually conducted, this function plots three
        different bivariate probability distributions in one plot: 1) previous
        system estimate 2) observation including their uncertainty 3) new estimate'''
        
        if not isinstance(log_var, str):
            raise TypeError
        
        ####################################
        ### prepare system macro-state estimate
        ####################################
        if log_var == "yes":
            x = np.log(self.macro_state_ensemble[0,:])
            y = np.log(self.macro_state_ensemble[3,:])
            # Set up grid points for plotting later fed into meshgrid
            x_grid = np.linspace(min(x)*0.9, max(x)*1.1, 100)
            y_grid = np.linspace(min(y)*0.9, max(y)*1.1, 100)
        elif log_var == "no":
            x = self.macro_state_ensemble[0,:]
            y = self.macro_state_ensemble[3,:]
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
        if log_var == "yes":
            x2 = np.log(self.data_ensemble[0,:])
            y2 = np.log(self.data_ensemble[3,:])
            x_grid2 = np.linspace(min(x2)*0.9, max(x2)*1.1, 100)
            y_grid2 = np.linspace(min(y2)*0.9, max(y2)*1.1, 100)
        elif log_var == "no": 
            x2 = self.data_ensemble[0,:]
            y2 = self.data_ensemble[3,:]
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
            if log_var == "yes":
                x3 = np.log(self.macro_state_ensemble_old[0,:])
                y3 = np.log(self.macro_state_ensemble_old[3,:])
                x_grid3 = np.linspace(min(x3)*0.9, max(x3)*1.1, 100)
                y_grid3 = np.linspace(min(y3)*0.9, max(y3)*1.1, 100)
            elif log_var == "no":
                x3 = self.macro_state_ensemble_old[0,:]
                y3 = self.macro_state_ensemble_old[3,:]
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
        #ax.set_xticklabels([0, 0.5, 1, 1.5,2,2.5,3,3.5])
        #ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        #ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        plt.show()
        #return X,Y,X2,Y2, Z2
        
    def plot_micro_state(self):
        
        """
        Function estimates current micro state of models and the ensemble of model
        estimates is an estimate for the EnKF micro state. This function estimates
        the probability of wealth across agents. For this purpose it applies some 
        Kernel Density estimation techniques.It has to do so on the log-transform of wealth
        (base e, so natural log) because wealth is per definition in model 
        power-lognormally distributed.The Gaussian, or any other, Kernel does not work well
        for visualization purpose on this kind of distribution.
        https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/ 
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("ln(wealth) per agent")
        colors = sns.color_palette("viridis", n_colors=self.ensemble_size)
        for i in range(self.ensemble_size):
            distr = np.log(self.micro_state_ensemble[:,i])
            sns.kdeplot(data=distr,
                        fill=False,
                        alpha = 0.5,
                        color = colors[i], 
                        ax = ax)
        mean_log = np.log(np.mean(self.micro_state_ensemble, axis = 1))
        sns.kdeplot(data=mean_log,
                    fill=False,
                    alpha = 1,
                    color = "black",
                    lw = 3,
                    linestyle = "--",
                    ax = ax,
                    label = "Mean of microstates")
        ax.legend(frameon = False)
        plt.show()
        
    def make_fanchart():
        
        '''make fancharts of model runs over growth rate, average wealth per adult,
        as well as wealth share by group until up to time point
        where the filter/model is applied to. Similar to current fig 2'''
        
        ### PLOT empirical monthly wealth Data vs model output for chosen time-frame
        colors = ["tab:red", "tab:blue", "grey", "y"]
        wealth_groups = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
        
        
        '''
        ### PLOT empirical monthly wealth Data (01/1990 to 12/2018) vs model output
        colors = ["tab:red", "tab:blue", "grey", "y"]
        wealth_groups = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
        fig, ax = plt.subplots(figsize=(6,4))
        for i, g in enumerate(wealth_groups): 
            x = d1["date_short"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
            y = d1["real_wealth_share"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
            x1 = np.linspace(1,time_horizon,time_horizon)
            y1 = wealth_groups_t_data[i]
            ax.plot(x,y, label = g, color = colors[i])
            ax.plot(x1, y1, label = g + ' model', linestyle = '--', color = colors[i])
            
        x = x.reset_index(drop=True)
        ax.set_xticks(x.iloc[0::20].index)
        ax.set_xticklabels(x.iloc[0::20], rotation = 90)
        ax.legend(frameon = False, bbox_to_anchor=(0.45, 0.7, 1., .102))
        ax.set_ylim((-0.05, 1))
        ax.set_ylabel("Share of wealth")
        ax.margins(0)
        '''
        pass

    def step(self, update: bool):
        
        if not isinstance(update, bool):
            raise TypeError
        
        self.update_decision = update ## var to pass on to other methods 
                                      ## for decision making
        self.predict()
        self.set_current_obs()
        self.update_state_ensemble()
        self.update_state_mean()
        self.update_data_ensemble()
        self.make_ensemble_covariance()
        self.make_data_covariance()
        self.make_gain_matrix()
        if update == True: 
            self.state_update()
        #### plot of the macro_state needs to come after state_update() so that
        #### it either includes all 3 (previous, new, observation) state estimates   
        #### or only the non-updated plus obsverational one
        self.plot_macro_state(log_var = "no")
        if update == True: 
            self.update_models()
            
        
