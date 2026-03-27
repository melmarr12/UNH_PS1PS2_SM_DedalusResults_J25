#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file was written and created by Melissa P. Marry in 2025.
"""
Created on Tue Jun 25 12:24:22 2024

@author: mmarry12
"""

def get_H_T_h_L(ts4,time_sec_short):
    #nondim_flow_pressure_HTh(ts4,time_sec_short):

    import numpy as np
    import pandas as pd
    
    # Removing the mean of the time series 
    ts4_nomean = ts4 - np.mean(ts4) #ts4.rolling(int(1//delta_t)).mean() #need to 
                                #figure out how to get the moving mean still for longer records
                                
    # #Plotting the pressure record closest to the bed (confirmed that this matches MatLab)
    # plt.figure(figsize=(5,4), dpi=250)
    # plt.plot(time_sec_short,ts4_nomean, color = newcolors[senh-1])
    # plt.xlim(time_sec_short[0],time_sec_short[-1]) #0,120)
    # plt.xlabel('t (seconds)')
    # plt.ylabel('p_d sensor 4 no mean(m)') #'p_b')
    # plt.title(startTime.strftime('%m/%d/%Y %X') + ' to ' + endTime.strftime('%m/%d/%Y %X'))

    k = np.arange(0,np.max(np.shape(ts4_nomean))-2)

    row = np.size(ts4_nomean,0)
    sgn = np.sign(ts4_nomean) #sign of data, 1 is a positive number and -1 is a negative number

    #%% Identifying the indices of the crossings

    cross = np.zeros((row-1,1))

    for ii in range(0,row-2):
        if sgn[ii]==+1 and sgn[ii+1]==-1:
            cross[ii]=1; # putting 1 for zero-cross locations

            
    #%% Counting the number of waves and filtering the data for any misidentified crossing points
    
    ind_trial=[]; # zero-crossing indices
    
    for iii in range(0,row-1):
        if cross[iii]==1:
            ind_trial.append(iii)
            
    n_waves_trial = np.size(ind_trial)-1; # number of waves
    
    for n in range(0,n_waves_trial):
        
        start_T = ind_trial[n]+1; # start indice for the current wave
        end_T = ind_trial[(n+1)]; # end indice for the current wave
        
        if end_T-start_T>1:
            flucs_n_T = ts4[start_T:end_T];
            
            if time_sec_short[end_T] - time_sec_short[start_T] < 7.0: #T[n] < 7.0:
                ind_trial[n] = np.nan
                
    ind_trial = np.array(ind_trial)
    ind_trial2 = ind_trial[~pd.isna(ind_trial)] 
    ind_trial23 = np.array(ind_trial2,dtype=int)
    
    n_waves_trial2 = np.size(ind_trial23)-1; # number of waves
    
    for n in range(0,n_waves_trial2):
            
        start_T2 = ind_trial23[n]+1; # start indice for the current wave
        end_T2 = ind_trial23[(n+1)]; # end indice for the current wave
        
        if end_T2-start_T2>1:
            flucs_n_T2 = ts4_nomean[start_T2:end_T2];
            a_crest_T2 = np.nanmax(flucs_n_T2); # max crest amplitude
            a_trough_T2 = np.nanmin(flucs_n_T2); # min trough amplitude      
            
            if a_crest_T2 < 0.1:
                ind_trial2[(n+1)] = np.nan
                
    ind = ind_trial2[~pd.isna(ind_trial2)] 
    ind = np.array(ind,dtype=int)
    n_waves = np.size(ind)-1; # number of waves
    
    #%% Calculating the water depths from each wave
    
    hW = np.empty((n_waves,1)); # water depth
    
    for n in range(0,n_waves):
        
        start = ind[n]+1; # start indice for the current wave
        end = ind[(n+1)]; # end indice for the current wave
        
        if end-start>1:
            hW[n] = np.nanmean(ts4[start:end]) #confirmed that this matches MatLab
            
        if abs(hW[n])>np.max(ts4):
            hW[n]=np.nan
            
            
    #%%
    # Coordinate system such that z = 0 is at the top of the PS and then
    # negative down from there into the water and the sand
    sensorselev = np.array([-56.4, -66.4, -72.65, -78.9, -85.15, -95.15, -110.47, -125.79]); #cm, first number corresponds to sensor #1 which should be closest to the brains for most data sets
    #sensorselev = np.array([-47.37, -57.37, -63.37, -68.37, -73.37, -78.37, -84.37, -90.37, -100.37, -110.37, -120.37, -130.37]);
    z = sensorselev[0:8]/100; #m, for vertical distribution of sensors
    #z = sensorselev[0:12]/100; #m, for vertical distribution of sensors
    ts4_hydrostatic = -z;
    ts4 = ts4 - (np.nanmean(hW)); #pressure with hydrostatic removed, Pe, meters ts4 = ts4 - (ts4_hydrostatic[3] + np.mean(hW)); #pressure with hydrostatic removed, Pe, meters
    
    # #Plotting the pressure record closest to the bed with hydrostatic removed (confirmed that this matches MatLab)
    # plt.figure(figsize=(5,4), dpi=250)
    # plt.plot(time_sec_short,ts4, color = newcolors[senh-1])
    # plt.xlim(time_sec_short[0],time_sec_short[-1]) #0,120)
    # plt.xlabel('t (seconds)')
    # plt.ylabel('p_d sensor 4 hydro removed (m)') #'p_b')
    # plt.title(startTime.strftime('%m/%d/%Y %X') + ' to ' + endTime.strftime('%m/%d/%Y %X'))
    # # print(np.nanmin(ts4), print(np.nanmax(ts4)) # values match MatLab!
    
    #%% Now recalculating the wave characteristics
        
    # Removing the mean of the time series 
    ts4_nomean = ts4 - np.mean(ts4) #ts4.rolling(int(1//delta_t)).mean() #need to 
                                    #figure out how to get the moving mean still for longer records
                                    
    k = np.arange(0,np.max(np.shape(ts4_nomean))-2)
    
    row = np.size(ts4_nomean,0)
    sgn = np.sign(ts4_nomean) #sign of data, 1 is a positive number and -1 is a negative number
    
    #%% Identifying the indices of the crossings
    
    cross = np.zeros((row-1,1))
    
    for ii in range(0,row-2):
        if sgn[ii]==+1 and sgn[ii+1]==-1:
            cross[ii]=1; # putting 1 for zero-cross locations
            
    
    #%% Counting the number of waves and filtering the data for any misidentified crossing points
    
    ind_trial=[]; # zero-crossing indices
    
    for iii in range(0,row-1):
        if cross[iii]==1:
            ind_trial.append(iii)
            
    n_waves_trial = np.size(ind_trial)-1; # number of waves
    
    for n in range(0,n_waves_trial):
        
        start_T = ind_trial[n]+1; # start indice for the current wave
        end_T = ind_trial[(n+1)]; # end indice for the current wave
        
        if end_T-start_T>1:
            flucs_n_T = ts4[start_T:end_T];
            
            if time_sec_short[end_T] - time_sec_short[start_T] < 7.0: #T[n] < 7.0:
                ind_trial[n] = np.nan
                
    ind_trial = np.array(ind_trial)
    ind_trial2 = ind_trial[~pd.isna(ind_trial)] 
    ind_trial23 = np.array(ind_trial2,dtype=int)
    
    n_waves_trial2 = np.size(ind_trial23)-1; # number of waves
    
    for n in range(0,n_waves_trial2):
                
        start_T2 = ind_trial23[n]+1; # start indice for the current wave
        end_T2 = ind_trial23[(n+1)]; # end indice for the current wave
        
        if end_T2-start_T2>1:
            flucs_n_T2 = ts4_nomean[start_T2:end_T2];
            a_crest_T2 = np.nanmax(flucs_n_T2); # max crest amplitude
            a_trough_T2 = np.nanmin(flucs_n_T2); # min trough amplitude      
            
            if a_crest_T2 < 0.1:
                ind_trial2[(n+1)] = np.nan
      
    ind = ind_trial2[~pd.isna(ind_trial2)] 
    ind = np.array(ind,dtype=int)
    n_waves = np.size(ind)-1; # number of waves
    
    #%% Calculating the wave heights and wave periods and water depth
    
    ind_n=[];
    H = np.empty((n_waves,1)); # wave heights
    T = np.empty((n_waves,1)); # wave periods
    hW2 = np.empty((n_waves,1)); # water depth
    
    for n in range(0,n_waves):
        
        start = ind[n]+1; # start indice for the current wave
        end = ind[(n+1)]; # end indice for the current wave
        
        if end-start>1:
            flucs_n = ts4_nomean[start:end];
            a_crest = np.nanmax(flucs_n); # max crest amplitude
            a_trough = np.nanmin(flucs_n); # min trough amplitude
            print(a_crest)
            H[n] = a_crest - a_trough #a_crest + np.absolute(a_trough); # wave heights
            T[n] = time_sec_short[end] - time_sec_short[start]; # wave periods
            hW2[n] = np.mean(ts4[start:end])
            
            if T[n] < 4.0: #4.0: #7.0:
                T[n] = np.nan
                H[n] = np.nan
                
            if T[n] < 0:
                T[n] = np.nan
                H[n] = np.nan
                
            if T[n] > time_sec_short[end]:
                T[n] = np.nan
                H[n] = np.nan
                
            if a_crest < 0.1:
                H[n] = np.nan
                T[n] = np.nan
                
                
    #out = np.column_stack((H,T,hW2));
    #np.savetxt('Column'+str(i)+'.txt',aa,header='(H,T)')
    
    
    #%%
    # #P = PS_data_short.iloc[:,0:8]
    # mid_pts = ind[0:len(ind)-1] + np.round(np.diff(ind)/2,0).astype(int) 
    
    # fig, axs = plt.subplots(4, 1, layout='constrained',figsize=(8,7), dpi=250)
    # axs[0].plot(time_sec_short, ts4, color = newcolors[senh-1]) #plot(range(0,n_waves),H, '.', markersize=20, color='#7E2F8E') #, label='z = 1.0')
    # axs[0].set_ylim(np.nanmin(ts4),np.nanmax(ts4)) #10,20) #0,10)
    # axs[0].set_xlim(time_sec_short[0],time_sec_short[-1]) #0, 120)
    # axs[0].set_xlabel('t (seconds)')
    # axs[0].set_ylabel('p_de 4 (m)') #'p_d 4 hydro removed (m)')
    # axs[0].vlines(time_sec_short[ind],np.nanmin(ts4),np.nanmax(ts4),colors='k',linestyles='dashed')
    # axs[0].title.set_text(startTime.strftime('%m/%d/%Y %X') + ' to ' + endTime.strftime('%m/%d/%Y %X'))
    
    # axs[1].plot(time_sec_short[mid_pts],hW, '.', markersize=20, color='#7E2F8E') #, label='z = 1.0')
    # axs[1].set_ylim(np.nanmean(hW)-1,np.nanmean(hW)+1) #10,20) #0,10)
    # axs[1].set_xlim(time_sec_short[0],time_sec_short[-1]) #0, 120)
    # axs[1].hlines(np.nanmean(hW),time_sec_short[0],time_sec_short[-1],colors='k',linestyles='dashed')
    # axs[1].set_xlabel('t (seconds)') #'Wave no.')
    # axs[1].set_ylabel('Water Depths (m)')
    # axs[1].title.set_text('h'' = ' + str(round(np.nanmean(hW),2)) + ' m')
    
    # axs[2].plot(time_sec_short[mid_pts],H, '.', markersize=20, color='#7E2F8E') #, label='z = 1.0')
    # axs[2].set_ylim(0,2) #10,20) #0,10)
    # axs[2].set_xlim(time_sec_short[0],time_sec_short[-1]) #0, 120)
    # axs[2].hlines(np.nanmean(H),time_sec_short[0],time_sec_short[-1],colors='k',linestyles='dashed')
    # axs[2].set_xlabel('t (seconds)') #'Wave no.')
    # axs[2].set_ylabel('Wave Heights (m)')
    # axs[2].title.set_text('H'' = ' + str(round(np.nanmean(H),2)) + ' m, a0 = ' + str(round(np.nanmean(H),2)/2) + ' m' )
    
    # axs[3].plot(time_sec_short[mid_pts],T, '.', markersize=20, color='#7E2F8E') #cbar.set_label('p',rotation=360) #plt.clabel('p')
    # axs[3].set_ylim(0,30)#8,20) #plt.xlim(0,10)
    # axs[3].set_xlim(time_sec_short[0],time_sec_short[-1]) #0, 120)
    # axs[3].hlines(np.nanmean(T),time_sec_short[0],time_sec_short[-1],colors='k',linestyles='dashed')
    # axs[3].set_ylabel('Wave Periods')
    # axs[3].set_xlabel('t (seconds)') #'Wave No.') #'x') #recall x=z
    # axs[3].title.set_text('T'' = ' + str(round(np.nanmean(T),2)) + ' s')


    #%% Other variables of interest for the problem
    Tp = np.nanmean(T); #s, wave period
    Hp = np.nanmean(H); # m, wave height
    a0p = Hp/2; #m, wave amplitude
    hp = np.nanmean(hW); #%m, water depth
    a0p_hp = a0p/hp; 

    #%% Calculating the dispersion relation to get the wavenumber/wavelength
    # found a NOAA toolbox for these functions
    # https://github.com/NOAA-ORR-ERD/wave_utils
    
    def dispersion(p, tol=1e-14, max_iter=100):
        """
        The linear dispersion relation in non-dimensional form:
    
        finds q, given p
    
        q = gk/omega^2     non-d wave number
        p = omega^2 h / g   non-d water depth
    
        Starts with the Fenton and McKee approximation, then iterates with Newton's
        method until accurate to within tol.
    
        :param p: non-dimensional water depth
        :param tol=1e-14: acceptable tolerance
        :param max_iter: maximum number of iterations to accept
                         (it SHOULD converge in well less than 100)
    
        """
        if p <= 0.0:
            raise ValueError("Non dimensional water depth d must be >= 0.0")
        # First guess (from Fenton and McKee):
        q = np.tanh(p ** 0.75) ** (-2.0 / 3.0)
    
        iter = 0
        f = q * np.tanh(q * p) - 1
        while abs(f) > tol:
            qp = q * p
            fp = qp / (np.cosh(qp) ** 2) + np.tanh(qp)
            q = q - f / fp
            f = q * np.tanh(q * p) - 1
            iter += 1
            if iter > max_iter:
                raise RuntimeError("Maximum number of iterations reached in dispersion()")
        return q

    omega = (2*np.pi)/Tp #wave frequency, Hz
    g = 9.81 # acceleration due to gravity, m/s^2
    disp_in = ((omega**2)*np.abs(hp))/g #input for the dispersion relation
    disp_out = dispersion(disp_in)
    k_wavenum = (disp_out*(omega**2))/g
    l0p = (2*np.pi)/k_wavenum #m, wavelength, should be about 46.9602, which it is
    
    return l0p, Tp, Hp, a0p, hp, a0p_hp, H, T, hW, sensorselev, z, ts4, ind