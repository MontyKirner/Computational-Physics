'''
-------------------------------------------------------------------------------

PHYS20762 Project 3 - Monte Carlo Techniques

Monty Kirner - 10301768 - 24/04/20 - University of Manchester

-------------------------------------------------------------------------------

This code simulates the penetration of thermal neutrons through different
shielding using Monte Carlo techniques.

-------------------------------------------------------------------------------
'''
#------------------------------------------------------------------------------
# Initialisation
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as pc
from numba import jit

# Imported files (must be in same directory as this .py file):
import Plot_Data as plot
import Stats as stats
from randssp import randssp

#------------------------------------------------------------------------------
# Constants and input data
#------------------------------------------------------------------------------

BARN = 10**-24

# Data formatted as absorption (σa/barn), scattering (σs/barn),
# density (gcm-3) and atomic mass (u): 
WATER_DATA = np.array([0.6652, 103.0, 1.0, 18.015])
LEAD_DATA = np.array([0.158, 11.221, 11.35, 207.2])
GRAPHITE_DATA = np.array([0.0045, 4.74, 1.67, 12.0107])

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

@jit(nopython = True)
def calculate_mean_free_paths(material_data):
    ''' Calculates the three mean free paths for a given data set and returns
    these values in a numpy array.
    '''
    # Gets number of moles where M = material_data[3]:
    n_moles = ( pc.Avogadro * material_data[2] ) / material_data[3]
    
    # Finds absorption and scattering cross section:
    sigma_a = material_data[0] * BARN
    sigma_s = material_data[1] * BARN
    
    # Calculates macroscopic cross sections:
    macro_sigma_a = n_moles * sigma_a 
    macro_sigma_s = n_moles * sigma_s 

    # Finds the mean free paths:
    lambda_a = 1 / macro_sigma_a
    lambda_s = 1 / macro_sigma_s
    lambda_t = 1 / ( macro_sigma_a +  macro_sigma_s )
    
    return np.array([lambda_a, lambda_s, lambda_t, 
                     macro_sigma_a, macro_sigma_s])
    
#------------------------------------------------------------------------------

def get_mean_free_paths_df():
    ''' Calculates the three mean free paths for the three different materials
    and returns these values in a df.
    '''
    water_lambda_array = calculate_mean_free_paths(WATER_DATA)
    lead_lambda_array = calculate_mean_free_paths(LEAD_DATA)
    graphite_lambda_array = calculate_mean_free_paths(GRAPHITE_DATA)

    # Finds the mean free paths and converts to df:    
    all_mean_free_paths_df = pd.DataFrame({
        'Water' : water_lambda_array,
        'Lead' : lead_lambda_array,
        'Graphite' : graphite_lambda_array }, 
        index = ['lambda_a / cm', 'lambda_s / cm', 'lambda_t / cm', 
                 'macro_sigma_a / cm', 'macro_sigma_s / cm'])
    
    return all_mean_free_paths_df

#------------------------------------------------------------------------------

@jit(nopython = True)
def random_uniform(array_length):
    ''' Required to use np.random.uniform function with numba, use this to call
    the function.
    '''        
    random_array = np.empty(array_length, dtype = np.float64)
    for i in range(array_length):
        random_array[i] = np.random.uniform(0, 1)
  
    return random_array

#------------------------------------------------------------------------------

@jit(nopython = True)
def get_rand_exp(mean_free_path, array_length):
    ''' Uses a random number generator to generate samples distributed 
    according to an exponential function exp(- u_data / path_length).
    '''
    u_data = random_uniform(array_length) 
    s_data = - mean_free_path * np.log(u_data)
  
    return s_data

#------------------------------------------------------------------------------

@jit(nopython = True)
def get_rand_vector(radius, array_length):
    ''' Generates isotropic vectors, r = (x, y, z), of specified radius using 
    spherical polar co-ordinates and returns these values.
    '''
    # Uses random_uniform since numba can't compile np.random.uniform:
    random_theta_array = random_uniform(array_length)
    random_phi_array = random_uniform(array_length)
    
    theta = np.arccos( 2 * random_theta_array - 1 )
    phi = 2 * np.pi * random_phi_array
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    return x, y, z

#------------------------------------------------------------------------------

@jit(nopython = True)
def get_rand_exp_vector(mean_free_path, array_length):
    ''' Generates uniform, isotropic, randomly distributed vectors with lengths
    distributed as exp(-x/l).
    '''
    # Sets up empty array:
    rand_exp_vectors = np.zeros( (array_length, 3) )
    
    # Creates set of s_data (lengths):
    s_data = get_rand_exp(mean_free_path, array_length)
    
    # Creates set of random unit vectors:
    x, y, z = get_rand_vector(1, array_length)
    
    for i in range(len(s_data)):
        
        # Multiplies each unit vector x, y, z by the random length:
        rand_exp_vectors[i][0] = x[i] * s_data[i]
        rand_exp_vectors[i][1] = y[i] * s_data[i]
        rand_exp_vectors[i][2] = z[i] * s_data[i]

    return rand_exp_vectors

#------------------------------------------------------------------------------

@jit(nopython = True)
def particle_walk(total_mean_free_path, macro_sigma_a, macro_sigma_s, 
                  thickness, track_positions):
    ''' Simulates a particle walk for one neutron through a slab of certain
    thickness. If track_positions = True, the output will consist of full array
    of the particles position history, else it will return the final position.
    '''
    particle_absorbed = False
    particle_killed = False
    
    # Sets up initial starting point:
    position = np.array([0., 0., 0.])
    
    step = 0
    
    # Adds first x-step to array and keeps track:
    if track_positions == True:
        
        # Use np.absolute to only take +ve direction:        
        new_position = np.absolute(get_rand_exp(total_mean_free_path, 1))
        position[0] += new_position[0]
        
        tracked_positions = np.vstack((np.array([0., 0., 0.]), position))
        step = 1
        
    while not particle_absorbed and not particle_killed:

        # First step without particle tracking:
        if step == 0:
            
            new_position = np.absolute(
                get_rand_exp(total_mean_free_path, 1))
            position[0] += new_position[0]
            step = 1
        
        # Following steps:
        else:
            # next_step has shape (1, 3):
            next_step = get_rand_exp_vector(total_mean_free_path, 1)
            
            # position has shape (3,):
            position[0] += next_step[0][0]
            position[1] += next_step[0][1]
            position[2] += next_step[0][2]
              
        reshaped_position = position.reshape((1, 3))
            
        # So that all positions are added to array (slower):
        if track_positions == True:
            
            tracked_positions = np.vstack((tracked_positions, 
                                           reshaped_position))
        # Used for absorbed elif statement:
        random_number = random_uniform(1)[0]
        
        if position[0] < 0:
            
            kill_type = 'Reflected'
            particle_killed = True

        elif position[0] > thickness:
            
            kill_type = 'Transmitted'
            particle_killed = True
            
        elif random_number < ( macro_sigma_a / (macro_sigma_a + 
                                                    macro_sigma_s) ):
            kill_type = 'Absorbed'
            particle_absorbed = True
           
        step += 1
       
    if track_positions == True:
        
        return tracked_positions, kill_type
    
    else: 
        return reshaped_position, kill_type

#------------------------------------------------------------------------------
    
@jit(nopython = True)
def get_particle_walk_arrays(number_of_neutrons, thickness, 
                             track_positions, total_mean_free_path, 
                             macro_sigma_a, macro_sigma_s):
    ''' Runs the particle_walk function for the number of neutrons chosen, and
    also counts the number of reflected / absorbed / transmitted neutrons to
    then return in console.
    '''
    reflected = 0
    absorbed = 0
    transmitted = 0
    
    # Numba cannot use np.array([]) so use lists:
    random_walk_x = []
    random_walk_y = []
    random_walk_z = []

    for neutron in list(range(0, number_of_neutrons)):
        
        positions, kill_type = particle_walk(
            total_mean_free_path, macro_sigma_a, macro_sigma_s, 
            thickness, track_positions)
        
        if kill_type == 'Reflected':
            reflected += 1
            
        elif kill_type == 'Absorbed':
            absorbed += 1  
            
        elif kill_type == 'Transmitted':
            transmitted += 1
            
        # Appends positions array, so that random_walk_x[0] is first neutron:
        random_walk_x.append(positions[:,0])
        random_walk_y.append(positions[:,1])
        random_walk_z.append(positions[:,2])
        
    return (random_walk_x, random_walk_y, random_walk_z, 
            reflected, absorbed, transmitted)    

#------------------------------------------------------------------------------

def roundup_max_to_10(data):
    ''' Finds the maximum value of an array and then rounds this up to the 
    next multiple of 10 (ie. 342 -> 350). Use this to choose even bin size.
    '''
    max_point = np.amax(data)
    rounded_max = ( np.ceil( max_point / 10.0) * 10 )
    
    return rounded_max

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Main code
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# For first few example plots:
small_samples = 2000
large_samples = 1000000

# Set to True to display initial setup plots / histogram & log plots:
plot_initial_graphs = True

plot_exp_hist_graph = True

# Finds the mean free paths for all materials:
all_mean_free_paths_df = get_mean_free_paths_df()

#------------------------------------------------------------------------------

# Change these values depending on what you want to test for random walks:

# How many neutrons simulated for each run:
number_of_neutrons = 100

# How many times you simulate the randoms walks:
number_of_simulations = 10

# Either 'Water' / 'Lead' / 'Graphite' for which material tested:
material_name = 'Water'

# If True, the simulation will keep a full array of the particles position
# history, if False it will only return the final position:
track_positions = True

# Returned values:
output = True

# To plot final simulations graph (also set thickness_range = False):
plot_random_walks = True

# Use this when only looking at 10cm:
thickness_values = np.array([10.0])

# If you want to choose a range, set to True:
thickness_range = False

if thickness_range == True:
        
    # In cm:
    thickness_values = np.arange(0.5, 40.5, 2)

    # Clutters console otherwise:
    output = False
    
#------------------------------------------------------------------------------

# Random generator that produces tables of points in 3 dimensions (x,y,z):
x_np = np.random.uniform(size = small_samples)
y_np = np.random.uniform(size = small_samples)
z_np = np.random.uniform(size = small_samples)

if plot_initial_graphs == True:
    
    # Diplays plot in 3D using Plot_Data.py:
    plot.scatter_3d(x_np, y_np, z_np, -60, 'numpy-uniform-3D-plot', 
                    axes = ['x', 'y', 'z'], fig_num = 1)

#------------------------------------------------------------------------------

# Confirming the spectral problem present using randssp:
# Creates 3D array of shape (3, samples):
randssp = randssp(3, small_samples)

x_randssp = randssp[0]
y_randssp = randssp[1]
z_randssp = randssp[2]   

if plot_initial_graphs == True:

    plot.scatter_3d(x_randssp, y_randssp, z_randssp, 60, 'randssp-3D-plot',
                    axes = ['x', 'y', 'z'], fig_num = 2)

#------------------------------------------------------------------------------

# Generates isotropic unit vectors:
x_unit, y_unit, z_unit = get_rand_vector(1, small_samples)

if plot_initial_graphs == True:
    
    plot.scatter_3d(x_unit, y_unit, z_unit, -60, 'rand-unit-vectors-3D-plot',
                    axes = ['x', 'y', 'z'], fig_num = 3)

#------------------------------------------------------------------------------

# Generates randomly distributed isotropic exponential vectors:
rand_exp_vectors = get_rand_exp_vector(
    all_mean_free_paths_df.loc['lambda_a / cm', 'Water'], small_samples)

if plot_initial_graphs == True:
    
    plot.scatter_3d(rand_exp_vectors[:,0], rand_exp_vectors[:,1], 
                    rand_exp_vectors[:,2], -60, 
                    'rand-exp-vectors-3D-plot', 
                    axes = ['x', 'y', 'z'], fig_num = 4)

#------------------------------------------------------------------------------

# Exponential function random number generator:
    
# Get exponential function for absorption:
water_absorp_s_data = get_rand_exp(all_mean_free_paths_df.loc[
    'lambda_a / cm', 'Water'], large_samples)

# Finds largest data point and round this up to next 10:
water_absorp_s_bin_max = roundup_max_to_10(water_absorp_s_data)

water_absorp_bins = np.arange(0, water_absorp_s_bin_max, 10)

if plot_exp_hist_graph == True:
    
    plot.histogram(water_absorp_s_data, water_absorp_bins, 
                   'water-absorption-exponential-hist', 
                   axes = ['Depths / cm', 'Number of neutrons, N'], fig_num =5)

#------------------------------------------------------------------------------

# Finds the number of neutrons per bin, N:
water_absorp_hist, water_absorp_bins = np.histogram(water_absorp_s_data, 
                                                    water_absorp_bins) 

# Finds the error for each bin, N_error:
water_absorp_errors = np.sqrt( water_absorp_hist *
                              ( 1 - (water_absorp_hist / large_samples)) )   

# Finds the bin midpoints (removes initial 0 since shifting by -5):
water_absorp_bin_midpoints = water_absorp_bins[1::] - 5

water_absorp_neutrons_df = pd.DataFrame({
    'Depths / cm' : water_absorp_bin_midpoints, 
    'N' : water_absorp_hist,
    'N_error' : water_absorp_errors })

# Removes rows where N <= 0 (ie. no neutrons in bin):
water_absorp_remove_zeroes_df = water_absorp_neutrons_df[
    water_absorp_neutrons_df['N'] > 0]

# Renames for plotting:
water_absorp_log_neutrons_df = pd.DataFrame({
    'log(N)' : np.log(
        water_absorp_remove_zeroes_df['N']) })

water_absorp_log_errors = (water_absorp_remove_zeroes_df['N_error']
                           / water_absorp_remove_zeroes_df['N'])

water_absorp_log_errors_df = pd.DataFrame({
    'log(N)' : water_absorp_log_errors })

#------------------------------------------------------------------------------

# Finds the fitted line and results:
water_absorp_fit, water_absorp_fit_results_df = stats.linear_fit(
    water_absorp_remove_zeroes_df['Depths / cm'],
    water_absorp_log_neutrons_df['log(N)'],
    water_absorp_log_errors_df['log(N)'], 
    'water absorption')
    
water_absorp_fit_df = pd.DataFrame({
    'log(N)' : water_absorp_fit })
                               
water_atten_length = -1 / water_absorp_fit_results_df.loc[
    'Value', 'Gradient']

water_atten_length_error = (
    water_absorp_fit_results_df.loc['Error', 'Gradient'] / 
    (water_absorp_fit_results_df.loc['Value', 'Gradient'])**2 )

# Use Stats.py's value rounder:
(water_atten_length_rounded, 
 water_atten_length_error_rounded) = stats.value_rounder(
     water_atten_length, water_atten_length_error)

print('''\nAttenuation length of water without \
scattering: ({:g} ± {:g})cm'''.format(
water_atten_length_rounded, water_atten_length_error_rounded)) 
print('\n' + '-' * 70)

if plot_exp_hist_graph == True:
    
    plot.scatter(water_absorp_remove_zeroes_df['Depths / cm'],
                 water_absorp_log_neutrons_df, 
                 water_absorp_log_errors_df,
                 water_absorp_fit_df, 
                 'water-absorption-log', 
                 axes = ['Depths / cm', 'log(N)'], fig_num =6)

#------------------------------------------------------------------------------

# Calculating random walks and plotting:

# To append mean value of fraction transmitted for each thickness:
fraction_trans_array = np.zeros((1, len(thickness_values)) )
fraction_trans_errors = np.zeros((1, len(thickness_values)) )

thickness_index = 0

for thickness in thickness_values:
    
    percent_reflected_array = np.zeros((1, number_of_simulations))
    percent_absorbed_array = np.zeros((1, number_of_simulations))
    percent_transmitted_array = np.zeros((1, number_of_simulations))
    
    # Runs required functions for each simulation:
    for simulation in range(0,number_of_simulations):
        
        (random_walk_x, random_walk_y, random_walk_z, 
         reflected, absorbed, transmitted) = get_particle_walk_arrays(
             number_of_neutrons, thickness, track_positions,
             all_mean_free_paths_df.loc['lambda_t / cm', material_name],
             all_mean_free_paths_df.loc['macro_sigma_a / cm', material_name],
             all_mean_free_paths_df.loc['macro_sigma_s / cm', material_name] )
        
        random_walk_x = np.asarray(random_walk_x)
        random_walk_y = np.asarray(random_walk_y)
        random_walk_z = np.asarray(random_walk_z)
        
        percent_reflected = (reflected / number_of_neutrons) * 100
        percent_absorbed = (absorbed / number_of_neutrons) * 100
        percent_transmitted = (transmitted / number_of_neutrons) * 100
        
        # Adds above values to array:
        percent_reflected_array[0][simulation] = percent_reflected
        percent_absorbed_array[0][simulation] = percent_absorbed
        percent_transmitted_array[0][simulation] = percent_transmitted
    
    #----------------------------------------------------------------------
    
    # Final results and plotting:
    
    # Calculates the mean and std for each type of process:
    percent_reflected_mean = np.average(percent_reflected_array)
    percent_absorbed_mean = np.average(percent_absorbed_array)
    percent_transmitted_mean = np.average(percent_transmitted_array)
    
    percent_reflected_std = np.std(percent_reflected_array)
    percent_absorbed_std = np.std(percent_absorbed_array)
    percent_transmitted_std = np.std(percent_transmitted_array)
    
    # Convert to a fraction:
    fraction_trans_array[0][thickness_index] = percent_transmitted_mean/100
    fraction_trans_errors[0][thickness_index] = percent_transmitted_std/100
    
    thickness_index += 1
    
    # Rounds all values to 1 sig fig of error:
    reflected_mean_rounded, reflected_std_rounded = stats.value_rounder(
        percent_reflected_mean, percent_reflected_std)
    absorbed_mean_rounded, absorbed_std_rounded = stats.value_rounder(
        percent_absorbed_mean, percent_absorbed_std)
    transmitted_mean_rounded, transmitted_std_rounded = stats.value_rounder(
        percent_transmitted_mean, percent_transmitted_std)
    
    if output == True:
        
        print('\n' + '-' * 70)
        print(('\nTransmission of neutrons through thickness = ' + 
               str(thickness) + 'cm for ' + str(number_of_simulations)
               + ' simulations'))
        print('\n' + '-' * 70)
        print('\nType of material = ' + str(material_name) )
        print('\nTotal neutrons for each simulation = ' 
              + str(number_of_neutrons) )
        print('\nPercentage reflected = ({:g} ± {:g})%'.format(
            reflected_mean_rounded, reflected_std_rounded) )
        print('\nPercentage absorbed = ({:g} ± {:g})%'.format(
            absorbed_mean_rounded, absorbed_std_rounded) )
        print('\nPercentage transmitted = ({:g} ± {:g})%'.format(
            transmitted_mean_rounded, transmitted_std_rounded) )
        print('\n' + '-' * 70)
        
#------------------------------------------------------------------------------           
         
if thickness_range == True:
    
    fraction_trans_df = pd.DataFrame({
        'Thickness of slab / cm' : thickness_values,
        'Fraction transmitted' : fraction_trans_array[0],
        'Fraction transmitted error' : fraction_trans_errors[0] })

    # Removes rows where N <= 0 (ie. no neutrons in bin):
    fraction_trans_remove_zeroes_df = fraction_trans_df[
        fraction_trans_df['Fraction transmitted'] > 0]

    # Rename for plotting:
    log_fraction_trans_df = pd.DataFrame({
        material_name : np.log(
            fraction_trans_remove_zeroes_df['Fraction transmitted']) })
    
    log_fraction_trans_errors = fraction_trans_remove_zeroes_df[
        'Fraction transmitted error'] / fraction_trans_remove_zeroes_df[
            'Fraction transmitted']

    log_fraction_trans_errors_df = pd.DataFrame({
        material_name : log_fraction_trans_errors})
    
    #----------------------------------------------------------------------

    # Finds the fitted line and results:
        
    fraction_trans_fit, fraction_trans_fit_results_df = stats.linear_fit(
    fraction_trans_remove_zeroes_df['Thickness of slab / cm'],
    log_fraction_trans_df[material_name],
    log_fraction_trans_errors_df[material_name], 
    'range of thicknesses')
    
    fraction_trans_fit_df = pd.DataFrame({
        material_name : fraction_trans_fit })
    
    material_atten_length = -1 / fraction_trans_fit_results_df.loc[
    'Value', 'Gradient']
  
    material_atten_length_error = (
        fraction_trans_fit_results_df.loc['Error', 'Gradient'] / 
        (fraction_trans_fit_results_df.loc['Value', 'Gradient'])**2 )
    
    # Use Stats.py's value rounder:
    (material_atten_length_rounded, 
     material_atten_length_error_rounded) = stats.value_rounder(
         material_atten_length, material_atten_length_error)
    
    print('\nAttenuation length of '+ str(material_name) + 
          ': ({:g} ± {:g})cm'.format(material_atten_length_rounded, 
                                     material_atten_length_error_rounded)) 
    print('\n' + '-' * 70)

#------------------------------------------------------------------------------    
    
if plot_random_walks == True and not thickness_range:
       
    if track_positions == True:
        save_name = (material_name + '-' + str(number_of_neutrons) +
                     '-neutrons-tracking-random-walk-3D-plot')
    else:
        save_name = (material_name + '-' + str(number_of_neutrons) +
                     '-neutrons-no-tracking-random-walk-3D-plot')

    plot.particle_walk_3d(random_walk_x, random_walk_y, random_walk_z, 
                          number_of_neutrons, thickness, -80, save_name, 
                          axes = ['x - cm', 'y - cm', 'z - cm'], fig_num = 8)

if thickness_range == True:
    
    plot.scatter(fraction_trans_remove_zeroes_df['Thickness of slab / cm'],
                 log_fraction_trans_df, 
                 log_fraction_trans_errors_df,
                 fraction_trans_fit_df, 
                 'thickness-range-log', 
                 axes = ['Thickness of slab / cm', 
                         'log( Fraction of transmitted neutrons )'], 
                 fig_num = 7)

#------------------------------------------------------------------------------