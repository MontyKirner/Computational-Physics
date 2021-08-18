"""
-------------------------------------------------------------------------------

Plotting Data Functions

Monty Kirner - 10301768 - 01/05/20 - University of Manchester

-------------------------------------------------------------------------------

This code contains various functions required to plot differnt types of graphs.

-------------------------------------------------------------------------------
"""
#------------------------------------------------------------------------------
# Initialisation
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker

# Sets up scientific notation for 3D plots:
formatter = ticker.ScalarFormatter(useMathText = True) 
formatter.set_scientific(True) 
formatter.set_powerlimits((-4,4)) 

params = {'legend.fontsize': 12,
         'axes.labelsize': 14,
         'xtick.labelsize':12,
         'ytick.labelsize':12 }
plt.rcParams.update(params)

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------    
        
def histogram(x_data, bins, save_name, axes, fig_num):
    ''' Creates a histogram for x_data with desired bin size. 
    '''
    x_label = axes[0]
    y_label = axes[1]
    
    fig = plt.figure(num = fig_num, figsize = (6, 5) )
    
    # Makes 1 subplot of figure:
    ax = fig.add_subplot()
    
    #ax.grid(True, c = '#777B88', dashes=[4,2])
    
    ax.hist(x_data, bins, color = 'orange', edgecolor='black')
    
    #--------------------------------------------------------------------------
           
    # Formatting, saving and displaying plots:
    ax.set_xlabel(r'{}'.format(x_label) )
    ax.set_ylabel(r'{}'.format(y_label) )
    
    ax.set_xlim(0, None)
    
    # If ticks sizes are >= 1e4, ticks are changed to scientific:
    ax.ticklabel_format(style = "sci", scilimits = (-4,4), useMathText = True)
    
    plt.tight_layout(pad = 0.5)
    plt.savefig('{}-graph.png'.format(save_name), dpi = 300 )
    
    return plt.show()

#------------------------------------------------------------------------------

def scatter(x_data, y_data, y_error, y_fit, save_name, axes, fig_num):
    ''' Creates a scatter graph for multiple datasets with errorbars and lines 
    of best fit. y_data_df and y_fit_df must be in pandas dataframe format with 
    the same column names, where the column names are also the labels used in 
    the legend.
    '''
    x_label = axes[0]
    y_label = axes[1]
        
    fig = plt.figure(num = fig_num, figsize = (6, 5) )
    
    # Makes 1 subplot of figure:
    ax = fig.add_subplot()
    
    ax.grid(True, c = '#777B88', dashes=[4,2])

    # Assigns colours to each set of data:
    plot_colours = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    
    # Iterates through y_data and y_fit to plot each method:
    for column in y_data:
        
        # Gets the data of one column in y_data:
        column_y_values = y_data.loc[ : , column ]
        
        # Gets the errorbars of one column in y_error:
        column_y_error_values = y_error.loc[ : , column ]
        
        # Gets the fitted data of one column in y_fit:
        column_y_fit_values = y_fit.loc[ : , column ]
        
        # Finds the index of the column:
        column_index = y_data.columns.get_loc(column)
    
        # Gets correct label to use in legend:
        labels = r'{:}'.format(column)
  
        # Plots data with errorbars:
        ax.errorbar(x_data, column_y_values, yerr = column_y_error_values,
                     ecolor = plot_colours[column_index], fmt = 'none', 
                     capsize = 5, label = labels)
  
        # Add the linear fit line:
        ax.plot(x_data, column_y_fit_values, '-', 
                c = plot_colours[column_index])
            
    #--------------------------------------------------------------------------
           
    # Formatting, saving and displaying plots:
    ax.set_xlabel(r'{}'.format(x_label) )
    ax.set_ylabel(r'{}'.format(y_label) )
    
    # If ticks sizes are >= 1e4, ticks are changed to scientific:
    ax.ticklabel_format(style = "sci", scilimits = (-4,4), useMathText = True)

    ax.legend(loc = 'best', framealpha = 0.9)
    
    plt.tight_layout(pad = 0.5)
    plt.savefig('{}-graph.png'.format(save_name), dpi = 300 )
    
    return plt.show()

#------------------------------------------------------------------------------

def scatter_3d(x_data, y_data, z_data, view_angle, save_name, axes, fig_num):
    ''' Creates a 3d scatter plot of given data.
    '''
    x_label = axes[0]
    y_label = axes[1]
    z_label = axes[2]
    
    fig = plt.figure(num = fig_num, figsize = (6, 6) )
    
    # Makes 1 3D subplot of figure:
    ax = fig.add_subplot(projection = '3d')
    
    ax.scatter(x_data, y_data, z_data, c = '#1F77B4')
    
    #--------------------------------------------------------------------------
           
    # Formatting, saving and displaying plots:
    ax.set_xlabel(r'{}'.format(x_label) )
    ax.set_ylabel(r'{}'.format(y_label) )
    ax.set_zlabel(r'{}'.format(z_label) )
    
    # If ticks sizes are >= 1e4, ticks are changed to scientific:
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    
    # For unit vector plots:
    if np.amax(x_data) <= 1 or np.amax(y_data) <= 1 or np.amax(z_data) <= 1:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(0.5))
    
    ax.view_init(azim = view_angle)
    
    plt.tight_layout(pad = 0.5)
    plt.savefig('{}-graph.png'.format(save_name), dpi = 300 )
    
    return plt.show()

#------------------------------------------------------------------------------

def particle_walk_3d(x_data, y_data, z_data, number_of_neutrons, thickness,
                     view_angle, save_name, axes, fig_num):
    ''' Creates a 3d plot of particle walks where x/y/z_data is in df format.
    '''
    x_label = axes[0]
    y_label = axes[1]
    z_label = axes[2]
    
    fig = plt.figure(num = fig_num, figsize = (8, 8) )
    
    # Makes 1 3D subplot of figure:
    ax = fig.add_subplot(projection = '3d')
    
    #--------------------------------------------------------------------------      
  
    # Plots starting point to use in legend: 
    ax.scatter(0, 0, 0, c = 'green', label = 'Start point')
    
    # Adds end points for legend:
    ax.scatter([], [], [], c = 'red', label = 'End points (reflected)')
    ax.scatter([], [], [], c = 'blue', label = 'End points (transmitted)')
    ax.scatter([], [], [], c = 'orange', label = 'End points (absorbed)')

    # Add number of neutrons to legend:
    plt.scatter([], [], [], c = 'white', label = ('''Number of neutrons = '''
                                          + str(number_of_neutrons)) )

    min_y_array = []
    max_y_array = []
    min_z_array = []
    max_z_array = []
    
    # Iterates through x_data to plot each neutron:
    for neutron in list(range(0, number_of_neutrons)):
        
        # Gets the data of one column in each df:
        x_values = x_data[neutron]
        y_values = y_data[neutron]
        z_values = z_data[neutron]
        
        # Plots data with connector lines:
        ax.plot(x_values, y_values, z_values)
       
        # Finds the last value of arrays:
        x_last = x_values[-1]
        y_last = y_values[-1]
        z_last = z_values[-1]
        
        # Finds min/max y and z values:
        min_y_value = np.amin(y_values)
        max_y_value = np.amax(y_values)
        min_z_value = np.amin(z_values)
        max_z_value = np.amax(z_values)
        
        min_y_array.append(min_y_value)
        max_y_array.append(max_y_value)
        min_z_array.append(min_z_value)
        max_z_array.append(max_z_value)
            
        if x_last < 0:
            ax.scatter(x_last, y_last, z_last, c = 'red')
            
        elif x_last > thickness:
            ax.scatter(x_last, y_last, z_last, c = 'blue')
            
        else:
            ax.scatter(x_last, y_last, z_last, c = 'orange')
             
    #--------------------------------------------------------------------------
    
    # Plot slab:
    
    # Gets min and max values of all y / z values:    
    min_y = np.amin(min_y_array)
    max_y = np.amax(max_y_array)
    
    min_z = np.amin(min_z_array)
    max_z = np.amax(max_z_array)
    
    y = np.linspace(min_y, max_y, 2)
    z = np.linspace(min_z, max_z, 2)

    Y, Z = np.meshgrid(y, z)
    
    # Add planes for slabs:
    ax.plot_surface(0, Y, Z, color = 'lightgrey', alpha = 0.3)
    ax.plot_surface(thickness, Y, Z, color = 'lightgrey', alpha = 0.3)
    
    #--------------------------------------------------------------------------
           
    # Formatting, saving and displaying plots:
    ax.set_xlabel(r'{}'.format(x_label) )
    ax.set_ylabel(r'{}'.format(y_label) )
    ax.set_zlabel(r'{}'.format(z_label) )
    
    # If ticks sizes are >= 1e4, ticks are changed to scientific:
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    
    ax.view_init(azim = view_angle)
    
    ax.legend(loc = 'best', framealpha = 0.9)
    
    plt.tight_layout(pad = 0.5)
    plt.savefig('{}-graph.png'.format(save_name), dpi = 300 )
    
    return plt.show()

#------------------------------------------------------------------------------
