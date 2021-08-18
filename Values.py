#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:32:10 2020

@author: monty
"""

# Calculated for each material:
water_log_fraction_trans = np.array([-0.6714815837871708, -2.0826546983431733, -3.0892959793323063, -4.070842902437009, -5.0208371565808525, -6.007187507573218, -6.980325971816972, -7.806697372521679, -8.8738681353549, -9.790158867229126, -10.724468104605958, -11.330603908176274, -12.429216196844383, -12.429216196844383])
water_log_fraction_trans_errors = np.array([0.0031210312036515156, 0.003964012034368528, 0.010727611984410072, 0.012667784715120674, 0.02759034222828288,0.061179309009609446,0.08493943078766238,0.12578796792304875,0.15649215928719032,0.5601566836199331,0.808017674301417,0.7264831572567788,1.2247448713915892,1.6583123951777])
water_fraction_trans_fit = np.array([-0.741591,-1.92736,-3.11313,-4.29891,-5.48468,-6.67045,-7.85622,-9.04199,-10.2278,-11.4135,-12.5993,-13.7851,-14.9709,-16.1566])

water_chi_squared = 256.73

lead_log_fraction_trans = np.array([-0.09232035379182099,-0.41020080383501556,-0.6782822136659465,-0.9098117657764461,-1.1235455552133213,-1.3168839768035516,-1.4989077822892864,-1.6770745611499116,-1.8416348220017058,-2.0079994054254837,-2.1659323269281368,-2.325065903423196,-2.4931283555747474,-2.644568488189788,-2.7997708310560014,-2.951332686257307,-3.1064627371642137,-3.2588908534064367,-3.411338630739048,-3.5653175925269043])
lead_log_fraction_trans_errors = np.array([0.000951404221861108,0.000914663157489358,0.0030026128928759715,0.002838406149180341,0.004406264732345099,0.004409260678513495,0.006764325839043382,0.005937308924159012,0.009065885091593565,0.008346534281672284,0.007993948316682344,0.010545393994993654,0.009429272753335778,0.010559753598695027,0.009517323432921478,0.013710178053659072,0.009510634830241284,0.014497711651082565,0.018830626005921967,0.013271155177292158])
lead_fraction_trans_fit = np.array([-0.167921,-0.369674,-0.571427,-0.77318,-0.974933,-1.17669,-1.37844,-1.58019,-1.78195,-1.9837,-2.18545,-2.3872,-2.58896,-2.79071,-2.99246,-3.19422,-3.39597,-3.59772,-3.79948,-4.00123])

lead_chi_squared = 1037.82

graphite_log_fraction_trans = np.array([	-0.09482399800984914,-0.4166827917136727,-0.672836839035195,-0.8824909353753561,-1.0630955421222477,-1.2170141407518904,-1.3523142791944567,-1.4743040544433048,-1.5854528671836396,-1.6884540638635115,-1.7806415034942085,-1.8678071675887093,-1.9483290722926232,-2.0276615657409662,-2.100812277816281,-2.1685088488280857,-2.2366288273313564,-2.3085729846885297,-2.3665264591308266,-2.4271624075790044])
graphite_log_fraction_trans_errors = np.array([0.0012027874333308299,0.0023296671017512963,0.0026535782535151127,0.004980874531510242,0.004258566513404878,0.0019125806833357349,0.006739115122627656,0.003194661954085484,0.00668677518807369,0.009297515132981965,0.006358493675618763,0.00778963365497433,0.008597001696261528,0.005276128101762667,0.006706189317526907,0.010363880664215393,0.011027595249632932,0.005913577614428852,0.009730371417039144,0.006734204145510679])
graphite_fraction_trans_fit = np.array([-0.246848,-0.391579,-0.53631,-0.681042,-0.825773,-0.970504,-1.11524,-1.25997,-1.4047,-1.54943,-1.69416,-1.83889,-1.98362,-2.12835,-2.27309,-2.41782,-2.56255,-2.70728,-2.85201,-2.99674])

graphite_chi_squared = 2872.23





water_log_fraction_trans = np.array([-0.6714815837871708, -2.0826546983431733, -3.0892959793323063, -4.070842902437009, -5.0208371565808525, -6.007187507573218, -6.980325971816972, -7.806697372521679, -8.8738681353549, -9.790158867229126, -10.724468104605958, -11.330603908176274, -12.429216196844383, -12.429216196844383])
water_log_fraction_trans_errors = np.array([0.0031210312036515156, 0.003964012034368528, 0.010727611984410072, 0.012667784715120674, 0.02759034222828288,0.061179309009609446,0.08493943078766238,0.12578796792304875,0.15649215928719032,0.5601566836199331,0.808017674301417,0.7264831572567788,1.2247448713915892,1.6583123951777])
water_fraction_trans_fit = np.array([-0.741591,-1.92736,-3.11313,-4.29891,-5.48468,-6.67045,-7.85622,-9.04199,-10.2278,-11.4135,-12.5993,-13.7851,-14.9709,-16.1566])

water_log_fraction_trans_df = pd.DataFrame({'Water' : water_log_fraction_trans})
water_log_fraction_trans_errors_df = pd.DataFrame({'Water' : water_log_fraction_trans_errors})
water_fraction_trans_fit_df = pd.DataFrame({'Water' : water_fraction_trans_fit})


lead_log_fraction_trans = np.array([-0.09232035379182099,-0.41020080383501556,-0.6782822136659465,-0.9098117657764461,-1.1235455552133213,-1.3168839768035516,-1.4989077822892864,-1.6770745611499116,-1.8416348220017058,-2.0079994054254837,-2.1659323269281368,-2.325065903423196,-2.4931283555747474,-2.644568488189788,-2.7997708310560014,-2.951332686257307,-3.1064627371642137,-3.2588908534064367,-3.411338630739048,-3.5653175925269043])
lead_log_fraction_trans_errors = np.array([0.000951404221861108,0.000914663157489358,0.0030026128928759715,0.002838406149180341,0.004406264732345099,0.004409260678513495,0.006764325839043382,0.005937308924159012,0.009065885091593565,0.008346534281672284,0.007993948316682344,0.010545393994993654,0.009429272753335778,0.010559753598695027,0.009517323432921478,0.013710178053659072,0.009510634830241284,0.014497711651082565,0.018830626005921967,0.013271155177292158])
lead_fraction_trans_fit = np.array([-0.167921,-0.369674,-0.571427,-0.77318,-0.974933,-1.17669,-1.37844,-1.58019,-1.78195,-1.9837,-2.18545,-2.3872,-2.58896,-2.79071,-2.99246,-3.19422,-3.39597,-3.59772,-3.79948,-4.00123])

lead_log_fraction_trans_df = pd.DataFrame({'Lead' : lead_log_fraction_trans})
lead_log_fraction_trans_errors_df = pd.DataFrame({'Lead' : lead_log_fraction_trans_errors})
lead_fraction_trans_fit_df = pd.DataFrame({'Lead' : lead_fraction_trans_fit})


graphite_log_fraction_trans = np.array([-0.09482399800984914,-0.4166827917136727,-0.672836839035195,-0.8824909353753561,-1.0630955421222477,-1.2170141407518904,-1.3523142791944567,-1.4743040544433048,-1.5854528671836396,-1.6884540638635115,-1.7806415034942085,-1.8678071675887093,-1.9483290722926232,-2.0276615657409662,-2.100812277816281,-2.1685088488280857,-2.2366288273313564,-2.3085729846885297,-2.3665264591308266,-2.4271624075790044])
graphite_log_fraction_trans_errors = np.array([0.0012027874333308299,0.0023296671017512963,0.0026535782535151127,0.004980874531510242,0.004258566513404878,0.0019125806833357349,0.006739115122627656,0.003194661954085484,0.00668677518807369,0.009297515132981965,0.006358493675618763,0.00778963365497433,0.008597001696261528,0.005276128101762667,0.006706189317526907,0.010363880664215393,0.011027595249632932,0.005913577614428852,0.009730371417039144,0.006734204145510679])
graphite_fraction_trans_fit = np.array([-0.246848,-0.391579,-0.53631,-0.681042,-0.825773,-0.970504,-1.11524,-1.25997,-1.4047,-1.54943,-1.69416,-1.83889,-1.98362,-2.12835,-2.27309,-2.41782,-2.56255,-2.70728,-2.85201,-2.99674])

graphite_log_fraction_trans_df = pd.DataFrame({'Graphite' : graphite_log_fraction_trans})
graphite_log_fraction_trans_errors_df = pd.DataFrame({'Graphite' : graphite_log_fraction_trans_errors})
graphite_fraction_trans_fit_df = pd.DataFrame({'Graphite' : graphite_fraction_trans_fit})


combined_log_fraction_trans_df = pd.concat(
    [water_log_fraction_trans_df, 
     lead_log_fraction_trans_df, 
     graphite_log_fraction_trans_df], ignore_index =True, axis =1)

combined_log_fraction_trans_errors_df = pd.concat(
    [water_log_fraction_trans_errors_df,
     lead_log_fraction_trans_errors_df,
     graphite_log_fraction_trans_errors_df], ignore_index =True, axis =1)

combined_fraction_trans_fit_df = pd.concat(
    [water_fraction_trans_fit_df,
     lead_fraction_trans_fit_df,
     graphite_fraction_trans_fit_df], ignore_index =True, axis =1)

combined_log_fraction_trans_df.columns = ['Water','Lead','Graphite']

combined_log_fraction_trans_errors_df.columns = ['Water','Lead','Graphite']

combined_fraction_trans_fit_df.columns = ['Water','Lead','Graphite']