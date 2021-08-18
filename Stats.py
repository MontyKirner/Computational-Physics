"""
-------------------------------------------------------------------------------

Statistical functions - Linear fitting / chi squared

Monty Kirner - 10301768 - 01/05/20 - University of Manchester

-------------------------------------------------------------------------------

This code contains statistical tools such as least squares fitting and 
chi-squared analysis.

------------------------------------------------------------------------------- 
"""
#------------------------------------------------------------------------------
# Initialisation
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------    

def chi_squared(x_data, y_data, y_err, y_fit):
    ''' Returns chi squared for a polynomial fit of input data.
    '''
    return np.sum( ( (y_data - y_fit) / y_err)**2 )
    
#-----------------------------------------------------------------------------      

def value_rounder(value, error_in_value):
    ''' Rounds final value to 1 sig fig of error. Returns these values.
    '''
    error_sig_fig = float('{:.1g}'.format(error_in_value))
    
    # Turns error into a string with 6 d.p:
    error_sig_fig_str = str( '{:f}'.format(error_sig_fig))
    
    # If number is greater than zero:
    if '.000000' in error_sig_fig_str:
        
        # Gets rid of zeros after dp:
        after_decimal_point_removed = error_sig_fig_str.replace('.000000', '')
        
        # Finds number of sig figs:
        sig_fig_number = len(after_decimal_point_removed)
        
        # Rounds calculated value to same as error sig fig:
        value_sig_fig = round(value, -(sig_fig_number - 1) )
    
    else:
        # Rounds calculated value to same as error sig fig:
        
        # Finds number of decimal places:
        dp_number = len( str(error_sig_fig) ) - 2
        
        # Rounds calculated value to same as error sig fig:
        value_sig_fig = round(value, (dp_number) )
    
    return value_sig_fig, error_sig_fig

#------------------------------------------------------------------------------    

def linear_fit(x_data, y_data, y_error, data_name):
    ''' Performs linear LS fitting of data and finds fitting parameters.
    Returns fitting parameters and array of values for LOBF.
    '''
    weights = 1 / y_error
    
    fit_parameters, fit_errors = np.polyfit(x_data, y_data, 1, 
                                            cov = True, w = weights)
    # Create set of fit values for graph:
    y_fit = np.polyval(fit_parameters, x_data)

    # The fit parameters are returned in fit_parameters:
    fit_m = fit_parameters[0]
    fit_c = fit_parameters[1]
    
    # The errors are returned in fit_errors:
    fit_sigma_m = np.sqrt(fit_errors[0][0])
    fit_sigma_c = np.sqrt(fit_errors[1][1])
    
    #--------------------------------------------------------------------------
    
    # Computes chi squared:
    fit_chi_squared = chi_squared(x_data, y_data, y_error, y_fit)
    
    # Computes reduced chi squared where denom = number of data points - 2,
    # since two degrees of freedom, a & b for linear fit:
    fit_reduced_chi_squared = ( fit_chi_squared / ( len(y_data) - 2) )
    
    #--------------------------------------------------------------------------
    
    # Rounds all values to 1 sig fig of error:
    fit_m_rounded, fit_sigma_m_rounded = value_rounder(fit_m, fit_sigma_m)
    fit_c_rounded, fit_sigma_c_rounded = value_rounder(fit_c, fit_sigma_c)
    
    #--------------------------------------------------------------------------
    
    # Summarise and print the results for the fit:
    print('\n' + '-' * 70)
    print('\nResults for linear fit of {} data:'.format(data_name) )
    print('\nGradient,  m = {:g} ± {:g}'.format(fit_m_rounded, 
                                                fit_sigma_m_rounded) )
    print('Intercept, c = {:g} ± {:g}'.format(fit_c_rounded, 
                                              fit_sigma_c_rounded) )
    print('\nReduced χ² = {:3.2f}\n'.format(fit_reduced_chi_squared) )
    print('-' * 70)
    
    #--------------------------------------------------------------------------
    
    fit_results_df = pd.DataFrame({
        'Gradient' : [fit_m, fit_sigma_m],
        'Intercept' : [fit_c, fit_sigma_c],
        'Reduced chi-squared' : [fit_reduced_chi_squared, 0]},     
        index = ['Value', 'Error'] )
    
    return y_fit, fit_results_df

#------------------------------------------------------------------------------