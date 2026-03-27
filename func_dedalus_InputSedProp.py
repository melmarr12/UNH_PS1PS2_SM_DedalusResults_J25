#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file was written and created by Melissa P. Marry in 2025.
"""
Created on Thu Nov 14 10:15:28 2024

@author: mmarry12
"""

def solve_pressure_dedalus_inputSedProp(ts4, time_sec_short, hp, dp, z, Hp, Tp, l0p, a0p, delta_t, G, Sr, ks, nearbedsen_ind):
    
    import numpy as np
    import dedalus.public as d3
    #import pandas as pd
    
    if np.isnan(nearbedsen_ind):
        nearbedsen_ind = 3 #z - (hp + z[2]) - OW and Oct 21 set up, 3 sen in water #z - (hp + z[3]) #error here I fixed
    
    zp = z - (hp + z[nearbedsen_ind]) 
    zp = zp[nearbedsen_ind:8] # zp[2:8] - OW and Oct 21 set up #zp[3:8]
    #zp = zp[nearbedsen_ind:12] # zp[2:8] - OW and Oct 21 set up #zp[3:8]
    
    if np.isnan(G): #np.isnan(G): #G == np.nan:
        G = 2*pow(10,7) #%Pa, =K in this case, Shear modulus
      
    if np.isnan(Sr):
        Sr = 0.99975528536502 #NaN; % degree of saturation

    if np.isnan(ks):
        ks = 1*pow(10,-10) #m^2, soil permability

    g = 9.81 #acceleration due to gravity, m/s^2
    rhop = 1025 #%1*10^(3); %kg m^-3, density of the fluid
    vf = 1*pow(10,-6) #m^2 s^-1, kinematic viscosity of the pore fluid
    nu = 0.33 #% Poisson's ratio of the soil skeleton
    n = (0.17+0.49)/2 #(0.12+0.46)/2 - Straws Pt, ODP Pt #(0.17+0.49)/2 - WS #% porosity, 0.17 - 0.49 from grain size
    Kw = 2.34*pow(10,9) ##np.nan #NaN; #% bulk modulus of elasticity of pure water
    pabsp = a0p*rhop*g ##np.nan #NaN; % absolute pore-water pressure
    K = 1/((1/Kw) + ((1 - Sr)/pabsp)) #% apparent bulk modulus of the elasticity of the pore fluid
    #K = 2*pow(10,7) #%Pa, =G in this case, ^ (if using this when K = G then Sr = 1 (Sr = (pabsp/Kw)+1-(1/(K*Kw))))
    
    nutil = (1 - 2*nu)/(2 - 2*nu) #% constant, nu with a tilde on top
    Gtil = n*(G/K) #% constant, G with a tilde on top
    beta = nutil/(nutil + Gtil) #% constant, beta = nut/(nut + Gt)
    #beta = beta/5; #beta*5; 
    
    #dp = 100 #10 #1000 #10 #1000 # %zp(end); %1000 %1, %10; %20; %5; %10; %m, seabed thickness
    
    #dx = 6 # %10; %6; %10; %linspace(-16,46,621); %(52,80,621); %(0,297.6,621); %(0,148.8,621); %(-71.8,77,621); %(-34.6,39.8,621); %(-16,21.2,621); %(-2,7.3,621);
    kapp1 = dp/l0p #% vertical to horizontal length ratio
    kapp12 = kapp1**2 #should these be negative?
    kapp2 = (1/l0p)*(np.sqrt((G*Tp*ks)/(rhop*vf))) #%dimensionless parameter? 1/m?
    kapp22 = kapp2**2 #should these be negative?
    alph = kapp1*(np.sqrt((nutil + Gtil))/kapp2) # % constant
    #alph = alph/5; #*5
    
    Z = (1/dp)*(zp + dp + hp) # %nondimensional vertical dimension, if want Z = 1 then zp = -hp
    #% % Z = [Z';0];
    #% % Z = Z';
    #% % % % Z = flip(0:0.1:1.0);
    
    tp = time_sec_short #%linspace(0,17*Tp,621); %linspace(0,255,621); %linspace(0,255,621); %(825,1000,621); %(0,1984,621); %(0,992,621); %(0,496,621); %(0,248,621); %(0,62,621);
    #t = np.max(tp)/Tp; #% nondimensional time, commenting this out for now since not used after this, does match matlab
    dt_time = tp/Tp; #%nondimensional time vector
    
    pb = ts4 #Pd(:,4); %m, dimensional pressure at the bed from the PS
    pb = pb/a0p #%nondimensionalize the pressure at the bed (already in meters so don't need to divide by rhof*g in this case
    #pb = pb1 - np.mean(pb1) #trying to center around zero to see if that helps
    
    #%% Interpolating the nondimensional pressure at the bed so that we can use it to force the boundary condition at any time
    
    from scipy.interpolate import interp1d, UnivariateSpline
    #pb_t = interp1d(dt_time, pb, kind='linear')
    pb_t = UnivariateSpline(dt_time, pb, k = 1, s=0, ext = 1.)
    #pb_t(1.)
    
    #%% Now from here can go in to Dedalus since everything is non dimensionalized?
    
    #%% Basis and Fields
    
    # Basis
    xcoord = d3.Coordinate('x') 
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.Chebyshev(xcoord, size=1024, bounds=(0,1), dealias=3/2) #could also try the RealFourier bases here maybe?
    
    # Fields
    u = dist.Field(name='u', bases=xbasis) #recall u = pressure!
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    
    #%% Forcing/Fields
    
    # Forcing/Fields
    x = dist.local_grids(xbasis)
    t = dist.Field()
    pb_temp = dist.Field() #pb_eq = dist.Field(bases=xbasis)
    
    def pb_function(*args):
        t = args[0].data
        pb_temp['g'] = (pb_t(t))*(1-beta) #pb[range(0,len(t))]*(1-beta) #pb[range(0,len(t))]*(1-beta) #pb[t]*(1-beta) #(np.cos(2*np.pi*(6 - t)))*(1-betaa) # pressure at the bed
        #print(t, pb_t(t))
        #pressure = pb_t(t)
        return pb_temp['g']
    
    def pb_Eq(*args, domain=pb_temp.domain, F=pb_function):
        return d3.GeneralFunction(dist, domain, layout='g', tensorsig=(), dtype=np.float64, func=F,args=args)
    
    #%% Substitutions (dx) and confirming that calculated P_b correctly
    # Substitution
    dx = lambda A: d3.Differentiate(A, xcoord)
    #_vec = np.linspace(0,10,1024) #0,50,1024)
    pb_P = pb*(1-beta) #(np.cos(2*np.pi*(6 - t_vec)))*(1-betaa) # pressure at the bed
    print('pb_P size = ', np.shape(pb_P))
    print('pb_P(0) =', pb_P[0])
    
    #%% Tau polynomials (copying this from Tutorial 4)
    
    tau_basis = xbasis.derivative_basis(2)
    p1 = dist.Field(bases=tau_basis)
    p2 = dist.Field(bases=tau_basis)
    p1['c'][-1] = 1
    p2['c'][-2] = 2
    
    #%% Defining the Boundary Value Problem we want to solve
    
    # Problem
    problem = d3.IVP([u, tau1, tau2], time=t, namespace=locals())
    #problem = d3.LBVP([u, tau1, tau2], namespace=locals())
    problem.add_equation("(alph**2)*dt(u) - dx(dx(u)) + tau1*p1 + tau2*p2 = 0")
    problem.add_equation("dx(u)(x='left')=0")
    
    problem.add_equation("u(x='right') = pb_Eq(t)") #"u(x='right') = pb_Eq(t)") #pb[0] #*np.cos(2*np.pi*(6 - np.linspace(0,50,1024)))") #*pb")
    
    #%% Solver and Initial Conditions
    
    # Solver
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = np.max(dt_time) # + (2*((delta_t/Tp)/10)) #np.max(dt_time) #20 #500, the simulation time variable goes to 500, are there are stopping criteria? 
    
    # Initial conditions
    x = dist.local_grid(xbasis)
    
    #%%
    # Setup storage
    u.change_scales(1)
    u_list = [np.copy(u['g'])]
    t_list = [solver.sim_time]
    
    #%%
    # Main loop
    timestep = (delta_t/Tp)/10 #(delta_t/Tp)/10 #delta_t #delta_t/Tp #0.005
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 10 == 0:
            u.change_scales(1)
            u_list.append(np.copy(u['g']))
            t_list.append(solver.sim_time)
        
    # Convert storage lists to arrays
    u_array = np.array(u_list)
    t_array = np.array(t_list)
    
       
    #%% Converting back to p from capital P
    
    # This pressure is the wrong pressure though!! It is capital P not p!!! And need to figure out units!
    u_array_copy = u_array.copy() #need to copy the array before working with it again. otherwise it changes the original array?
    p = u_array_copy #u_array.copy()
    
    pb_new = pb #this needs to be pb! not Pb, p[:,1023] #pb #np.cos(2*np.pi*(6 - t_array)) # pressure at the bed
    #pb_t_new = interp1d(dt_time, pb_new, kind='linear')
    pb_t_new = UnivariateSpline(dt_time, pb_new, k = 1, s=0, ext = 1.)
    pb_new_short = pb_t_new(t_array)
    print(np.shape(pb_new_short))
    # u_dim = np.shape(u_array) 
    # u_dim1 = u_dim[0]
    # u_dim2 = u_dim[1]
    
    #n_u = np.linspace(0,1023,1024,dtype=int) # don't edit the last one since that is the pressure at the surface which should not change #np.linspace(0,1023,1024,dtype=int)
    #in_u = int(in_u)
    #for ii in in_u:
    for ii in range(0,1024):
         p[:,ii] = u_array_copy[:,ii] + beta*pb_new_short
        
    return zp, Z, dt_time, pb, u_array, t_array, x, p, alph, beta, pb_t, pb_new_short, G, Sr, ks, K, n