 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 12:10:00 2025

@author: mmarry12
"""
# This file was written and created by Melissa P. Marry in 2025.
# File created for publication of software used to numerically solve for the pressure within 
# the sediment bed using the Dedalus Project and forced with field observations

# Boundary value problem (BVP) initially presented in Tong and Liu (2022) paper
# Citation: Tong and Liu (2022) 'Transient wave-induced pore-water-pressure 
# and soil responses in a shallow water unsaturated poroelastic seabed'
# Specifically Equations (3.7)-(3.9) in Tong and Liu (2022)
# P = p - beta*p_b
# BVP: (alpha^2)*dt(P) = dZ(dZ(P))
# Bounds: 0 <= Z <= 1
# B.C.: dZ(P) = 0 at Z = 0, P = (1 - beta)*p_b at Z = 1
#
# To put this in the same frame of reference as the Dedalus Project, then we can say that:
# Z = x, P = u
#
# Force the boundary condition with field observations of pressure from instruments called Pressure Sticks
#
# Field observations originally used are from the University of New Hampshire January 2025 Wallis Sands Beach (located in Rye, New Hampshire)
# field experiment with 2 Pressure Sticks and the SediMeter (20250118)
# but this code should work with any nearbed pressure time series as the forcing for the boundary condition.
# 
# This code solves the boundary value problem for 20 minute segments of data throughout a tidal cycle.
# 
# Results from this software can be found in: Doctoral dissertation, "Resolving the Role of Wave-Induced Pressure Gradients on the Movement of 
# Sediments in Nearshore Environments" (M. Marry, University of New Hampshire, 2025) and the manuscript, "Vertical Propagation of Wave-Induced 
# Pressure Within Nearshore Sediment Beds: Field Observations and a Numerical Solution" (Marry & Foster, XXXX) which has been submitted to the 
# Journal of Geophysical Research: Oceans in 2026. 
# This software is published in - insert GitHub Zenodo publication for this 
# For more information about the instrumentation (Pressure Sticks) and the field experiment please see: the publication "Field Observations of 
# Hydrostatic Pressure Deviations in a Nearshore Sediment Bed" (Marry & Foster, 2024) in the Journal of Geophysical Research: Oceans, the 
# doctoral dissertation, "Resolving the Role of Wave-Induced Pressure Gradients on the Movement of Sediments in Nearshore Environments" 
# (M. Marry, University of New Hampshire, 2025) and the manuscript, "Vertical Propagation of Wave-Induced Pressure Within Nearshore Sediment 
# Beds: Field Observations and a Numerical Solution" (Marry & Foster, XXXX) which has been submitted to the Journal of Geophysical Research: 
# Oceans in 2026.
# For more information about the Dedalus Project: https://dedalus-project.org/
#
#%% Importing the necessary libraries
import pathlib
import subprocess
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dedalus.public as d3
from datetime import datetime as datetime_builtfunc

from nondim_flow_pressure_HTh import get_H_T_h_L #This function is also published within this same data repository.
from func_dedalus_InputSedProp import solve_pressure_dedalus_inputSedProp #This function is also published within this same data repository.

#%% Importing the pressure field observations
# For this case, Pressure Stick version 1 data will be used but the options for Pressure Stick version 2 will be included in the comments 

savevar = 0 #Save variable, if savevar = 0 then no files will be saved, if savevar = 1 then the results from the numerical solution will be saved as a '.h5' file

dn = np.loadtxt('UNH_PS1_dataJ25_dn.txt',dtype=float,delimiter=None) #datenumber for the entire dataset, for PSv2a then use 'UNH_PSv2a_dataJ25_dn.txt'
#Pd = np.loadtxt('UNH_PS1_dataJ25_Pd.txt',dtype=float,delimiter=',') #pressure converted to meters for the entire dataset and for all pressure sensors within the Pressure Stick, for PSv2a then use 'UNH_PSv2a_dataJ25_Pd.txt'
PS_data = pd.read_csv("UNH_PS1dataJ25.csv") #dataframe of pressure converted to meters for the entire dataset and for all pressure sensors within the Pressure Stick, also includes the datetime value of each sample. For PSv2a then use "UNH_PSv2adataJ25.csv"
PS_data["datetime"] = pd.to_datetime(PS_data["datetime"],format="%d/%m/%Y %H:%M:%S.%f")
datetimes = PS_data["datetime"]
PS_data=PS_data.set_index("datetime") #setting the datetime as the index for the dataframe

#%% Plotting the raw Pressure Stick data (full deployment)
newcolors = ['#00FFFF', '#4DBEEE', '#0072BD', '#0000FF','#FF0000','#A2142F','#D95319','#EDB120'] #hex color codes for usual Pressure Stick version 1 sensor plotting colors
# for PSv2a use: newcolors = ['#00FFFF', '#4DBEEE', '#0072BD','#0000FF','#2960B0','#52c060','#2f9439','#FF0000','#A2142F','#D95319','#FF6E00','#EDB120'];

#%% Calculating an array of the time in seconds of the full record and adding it to our dataframe.
# Confirm that we can plot the data with the time in seconds as the x-axis
time_sec = (dn - dn[0])*24*60*60 #log of the time in seconds of the record
PS_data.insert(8, "Seconds Log", time_sec, True)
PS_data.insert(9, "dn", dn, True)

#%% Inputing the SediMeter data which tells you the pressure sensor elevations in relevance to the sediment-water interface through the time of the experiment
dn_SM = np.loadtxt('UNH_PS1_dataJ25_sensorselev_wtime_dn.txt',dtype=float,delimiter=None) #datenumber for the SediMeter data, for PSv2a then use 'UNH_PSv2a_dataJ25_sensorselev_wtime_dn.txt'
#sensorselev_wtime = np.loadtxt('UNH_PS1_dataJ25_sensorselev_wtime.txt',dtype=float,delimiter=',') #pressure sensors elevations in cm in relevance to the location of the sediment-water interface as measured by the SediMeter for the full experiment, for PSv2a then use 'UNH_PSv2a_dataJ25_sensorselev_wtime.txt'

SM_data = pd.read_csv("UNH_PS1dataJ25_senelev.csv") #dataframe of pressure sensors elevations in cm in relevance to the location of the sediment-water interface as measured by the SediMeter for the full experiment, for PSv2a then use "UNH_PSv2adataJ25_senelev.csv"
SM_data["datetime"] = pd.to_datetime(SM_data["datetime"],format="%d/%m/%Y %H:%M:%S.%f")
datetimesSM = SM_data["datetime"]
SM_data=SM_data.set_index("datetime") #setting the datetime as the index for the dataframe

#%% Selecting only a portion of the full record to work with
# 
# If want to go through tidal cycle by tidal cycle for the full deployment then use the following sections one at a time
# To work with a specific tidal cycle then unncomment the corresponding 'startTime' and 'endTime' variables
# If saving the data for multiple 20 minute segments the be sure to change the output file name below 

# first tidal cycle in the January 2025 deployment is from 1/18/2025 10:25 - 1/18/2025 18:05
# 7 hours and 40 minutes = 460 minutes divided by 20 minutes would then be 23 segments - make range of ii from 0 to 23
# output file name: "DedalusOutput_PS_chunks_Jan25_PS1_" + str(ii) + ".h5"
##startTime = pd.Timestamp('2025-01-18 10:25:00') #start time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'  
##endTime = pd.Timestamp('2025-01-18 10:45:00') #end time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'  

# second tidal cycle in the January 2025 deployment is from 1/18/2025 23:05 - 1/19/2025 6:55
# 7 hours and 50 minutes = 470 minutes divided by 20 minutes would then be 23.5 segments - make range of ii from 0 to 24
# add 23 to the output file name: "DedalusOutput_PS_chunks_Jan25_PS1_" + str(ii+23) + ".h5"
##startTime = pd.Timestamp('2025-01-18 23:05:00') #start time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'
##endTime = pd.Timestamp('2025-01-18 23:25:00') #end time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'

# third tidal cycle in the January 2025 deployment is from 1/19/2025 11:05 - 1/19/2025 18:55 
# 7 hours and 40 minutes = 460 minutes divided by 20 minutes would then be 23 segments - make range of ii from 0 to 23
# add 45 (23+22) to the output file name: "DedalusOutput_PS_chunks_Jan25_PS1_" + str(ii+45) + ".h5"
##startTime = pd.Timestamp('2025-01-19 11:05:00') #start time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'
##endTime = pd.Timestamp('2025-01-19 11:25:00') #end time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'

# fourth tidal cycle in the January 2025 deployment is from 1/19/2025 23:15 - 1/20/2025 7:15
# 8 hours and 0 minutes = 480 minutes divided by 20 minutes would then be 24 segments - make range of ii from 0 to 24
# add 68 (45+25) to the output file name: "DedalusOutput_PS_chunks_Jan25_PS1_" + str(ii+73) + ".h5"
##startTime = pd.Timestamp('2025-01-19 23:15:00') #start time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'
##endTime = pd.Timestamp('2025-01-19 23:35:00') #end time of first 20 minute section of interest, format is 'yyyy-MM-DD HH:mm:ss'

# If don't want to work one tidal cycle at a time or want to work with a specific 20 minute time section then can input that time section here
# You can work with sections longer or shorter than 20 minutes but you may have to adjust this current code as anything longer than 20 minutes will be divided up into 20 minute sections.
startTime = pd.Timestamp('2025-01-19 15:00:00') #start time of section of interest, format is 'yyyy-MM-DD HH:mm:ss' 
endTime = pd.Timestamp('2025-01-19 15:20:00') #end time of section of interest, format is 'yyyy-MM-DD HH:mm:ss' 


#%% Loop that solves the boundary value problem for the specified time section using the Dedalus Project
# ii is the indexing variable for the number of continuous 20 minute segments your interested in
#for ii in range(0,23): (0,24): (0,23): (0,24):
for ii in range(0,1):
    PS_data_short = PS_data.copy()
    SM_data_short = SM_data.copy()
    startTime2 = startTime + pd.to_timedelta(20*ii,unit='minute') #used to adjust the start and end times after the first 20 minute segment
    endTime2 = endTime + pd.to_timedelta(20*ii,unit='minute')
    
    PS_data_short = PS_data_short.truncate(before=startTime2, after=endTime2) #selects the pressure and sensor elevations data for the 20 minute segment from the full record
    SM_data_short = SM_data_short.truncate(before=startTime2, after=endTime2)
    
    PS_data_short.iloc[:,8:9] = (PS_data_short.iloc[:,9:10] - PS_data_short.iloc[0,9:10])*24*60*60 #log of the time in seconds of the record
    dn = PS_data_short.iloc[:,-1] #selecting the datenumber information for the 20 minute segment
       
    avg_senselev = SM_data_short.mean() #average sensor elevations over the 20 minutes    
    avg_senselev.reset_index(drop=True, inplace=True)
    nearbedsen_ind = avg_senselev.abs().idxmin() #identifying the index of the near-bed sensor, which pressure sensor is closest to the sediment-water interface for this 20 minute segment
     
    
    #%% Defining some helpful variables for nondimensionalization
    time_sec_short = (PS_data_short.iloc[:,-1]-PS_data_short.iloc[0,-1])*24*60*60 #log of the time in seconds of the record
    senh = nearbedsen_ind+1; #identifying the near-bed sensor number in the context of the Pressure Sensor not in pythons indexing framework          
    delta_t = np.mean(np.diff(time_sec)) #sampling rate, in seconds
    ts4 = PS_data_short.iloc[:,senh-1] #selecting the pressure time series closest to the sediment-water interface, in meters
    dp = 1000 #seabed thickness, in meters
    
    #%% Function to calculate the hydrodynamic parameters need to nondimensionalize
    [l0p, Tp, Hp, a0p, hp, a0p_hp, H, T, hW, sensorselev, z, ts4, ind] = get_H_T_h_L(ts4,time_sec_short)

    # Geotechnical inputs needed to run the numerical model 
    G = np.nan #Shear modulus, in Pa, can select a value or leave as a NaN and the defualt value will be used 
    Sr = np.nan #degree of saturation, can select a value or leave as a NaN and the defualt value will be used 
    ks = np.nan #soil permability, in m^2, can select a value of leave as a NaN and the defualt value will be used 
    
    # Function that solves the BVP for pressure within the sediment bed uses the Dedalus Project
    [zp, Z, dt_time, pb, u_array, t_array, x, p, alph, beta, pb_new_short, G, Sr, ks, K, n] = solve_pressure_dedalus_inputSedProp(ts4, time_sec_short, hp, dp, z, Hp, Tp, l0p, a0p, delta_t, G, Sr, ks,nearbedsen_ind)
    
    #%% To save the data if desired, here saving as a '.h5' file
    if savevar==1:
        outputfilename = "DedalusOutput_PS_chunks_Jan25_PS1_" + str(ii) + ".h5" #select an output file name for the data to be saved, something descriptive, if calculating over multiple 20 minute segments then can include str(ii) to number the files
        with h5py.File(outputfilename, "w") as data_file:
            data_file.create_dataset("data/dn", data=dn) #saving the datenumber
            data_file.create_dataset("data/z", data=x) #saving the vertical dimension, in cm
            data_file.create_dataset("data/t", data=t_array) #saving the time vector, in seconds
            data_file.create_dataset("data/P_cap", data=u_array) #saving the nondimensional pressure, 'P', throughout the sediment bed 
            data_file.create_dataset("data/p", data=p) #saving the pressure throughout the sediment bed, in m
            data_file.create_dataset("data/pb_tarr",data=pb_new_short) #saving the near-bed pressure record, in m
            data_file.create_dataset("data/a0p", data=a0p) #saving the nondimensional wave amplitude
            data_file.create_dataset("data/alpha", data=alph) #saving the alpha parameter
            data_file.create_dataset("data/beta", data=beta) #saving the beta parameter
            data_file.create_dataset("data/dp", data=dp) #saving the seabed depth, in m
            data_file.create_dataset("data/hp", data=hp) #saving the water depth, in m
            data_file.create_dataset("data/Hp", data=Hp) #saving the wave height, in m
            data_file.create_dataset("data/Tp", data=Tp) #saving the wave period, in s
            data_file.create_dataset("data/pb2",data=pb)  #saving the near-bed pressure record, in m??
            data_file.create_dataset("data/t2",data=dt_time)  #saving the time record again, in s??
            data_file.create_dataset("data/G",data=G)  #saving the Shear modulus, in Pa 
            data_file.create_dataset("data/Sr",data=Sr) #saving the degree of saturation
            data_file.create_dataset("data/ks",data=ks)  #saving the soil permability, in m^2
            data_file.create_dataset("data/avg_senselev",data=avg_senselev) #saving the average sensor elevations, in cm 
            data_file.create_dataset("data/nearbed_sen",data=senh) #saving the near-bed sensor number 
            
