import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize

#perform a regression of the each asset's excess return and risk factors
def ols(excess,factors):
    ols_dict = []
    for port in excess.columns:
        ols_dict.append(sm.OLS(excess[port],sm.add_constant(factors),missing='drop').fit())
    return ols_dict

#generate random weights
def random_weights(number):
    return np.random.rand(number)

#Portfolio return
def portfolio_return(weights,returns_df):
    mean_returns = returns_df.mean().values#+ols_list[i].params['const'] 
    port_return = np.dot(weights,mean_returns)
    #print(port_return)
    return port_return

#normal portfolio variance
def portfolio_variance(weights,returns_df):
    port_cov = returns_df.cov().values
    port_variance = np.matmul(np.transpose(weights),np.matmul(port_cov,weights))
    return np.sqrt(port_variance)

#ols_list = ols(excess_returns,FF_factors)
#ols_list_index = ols(excess_returns_index,FF_factors)

#function to calculate portfolio variance considering risk factors
def factor_portfolio_variance(weights,risk_factors_df,ols_list):
    factor_cov_matrix = risk_factors_df.cov() #If K factors in Multifactor model KxK matrix
    factor_loadings = np.array([fit.params[1:] for fit in ols_list])
    port_cov = np.matmul(np.matmul(factor_loadings,factor_cov_matrix.values),np.transpose(factor_loadings))
    #print('Portfolio Covariance\n')
    #print(port_cov)
    residual_matrix = [fit.mse_resid for fit in ols_list]
    diag = np.einsum("ii->i",port_cov)
    diag+=residual_matrix
    port_variance = np.dot(np.transpose(weights),np.dot(weights,port_cov))
    #print('Portfolio Variance')
    return port_variance

#function to calculate factor covariance
def factor_cov(weights,risk_factors_df,factor_idx,ols_list):
    factor_cov_matrix = risk_factors_df.cov() #If K factors in Multifactor model KxK matrix
    factor_loadings = np.array([fit.params[1:] for fit in ols_list])
    #print("Factor Loadings\n")
    #print(factor_loadings)
    port_factor_loadings = np.dot(weights,factor_loadings) #If N securities 1xK matrix
    #print(port_factor_loadings)
    factor_cov = np.dot(port_factor_loadings,factor_cov_matrix.values[factor_idx,])/factor_cov_matrix.values[factor_idx,factor_idx]
    return factor_cov

#function to display answer in required format
def display_answer(risk_factors_df,ols_list,excess_returns,opt_weights):
    factor_cov_matrix = risk_factors_df.cov() #If K factors in Multifactor model KxK matrix
    factor_loadings = np.array([fit.params[1:] for fit in ols_list])
    port_cov = np.matmul(np.matmul(factor_loadings,factor_cov_matrix.values),np.transpose(factor_loadings))
    residual_matrix = [fit.mse_resid for fit in ols_list]
    diag = np.einsum("ii->i",port_cov)
    diag+=residual_matrix
    #print('residual matrix\n')
    #print(residual_matrix)
    #print('\n')
    #print('Portfolio Covariance\n')
    #print(pd.DataFrame(port_cov,index=excess_returns.columns,columns=excess_returns.columns))
    output = {'Weights':opt_weights, 'Mean Returns':excess_returns.mean(),'Standard Deviation':excess_returns.std()}
    q1 = pd.concat([pd.DataFrame(output),pd.DataFrame({name:fit.params for name,fit in zip(excess_returns.columns,ols_list)}).transpose()],axis=1)#.to_csv('q1.csv')
    factor_sensetivity = np.matmul(q1['Weights'].values,q1[risk_factors_df.columns].values)
    #print('\n')
    #print('Optimized Weights')
    #print(q1['Weights'])
    #print('\n')
    #print('Portfolio Factor Loadings')
    #print(factor_sensetivity)
    #print('\n')
    #print('Portfolio Excess Return')
    #print(portfolio_return(q1['Weights'],excess_returns))
    #print('\n')
    #print('Portfolio Variance')
    #print(portfolio_variance(q1['Weights'],excess_returns))
    return tuple([q1['Weights'].values,
                 factor_sensetivity,
                 portfolio_return(q1['Weights'],excess_returns*12),
                 portfolio_variance(q1['Weights'],excess_returns)])