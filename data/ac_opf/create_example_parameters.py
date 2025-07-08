import pandas as pd
import numpy as np
import os
import pandapower as pp
import pandapower.converter as pc
from pypower.makeYbus import makeYbus

# def create_example_parameters(n_buses: int):
#     """
#     creates a basic set of parameters that are used in the following processes:
#     * data creation if measurements are to be simulated
#     * setting up the neural network model
#     * training procedure

#     :param n_buses: integer number of buses in the system
#     :return: simulation_parameters: dictonary that holds all parameters
#     """

#     # -----------------------------------------------------------------------------------------------
#     # underlying parameters of the power system
#     # primarily for data creation when no measurements are provided
#     # -----------------------------------------------------------------------------------------------
    
#     # Get the absolute path to the folder containing this script
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
#     # Construct the data directory path
#     data_dir = os.path.join(base_dir, f'ac_opf_data/{n_buses}')
        
#     AC_OPF = True    
#     if AC_OPF == True:
#         Gen = pd.read_csv(os.path.join(data_dir, 'Gen.csv'), index_col=0)
#         g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
#         n_gbus=len(g_bus)
#         # C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
#         Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
#         Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
#         Qg_max=Gen['Qg_max'].to_numpy().reshape((1, n_gbus))
#         Qg_min=Gen['Qg_min'].to_numpy().reshape((1, n_gbus))
#         Gen_delta=np.concatenate((Pg_max-Pg_min,Qg_max-Qg_min),axis=1).reshape((2*n_gbus,1))
#         Gen_max=np.concatenate((Pg_max,Qg_max),axis=1).reshape((2*n_gbus,1))
#         Volt_max = 1.06
#         Volt_min = 0.94
#         Map_g = np.zeros((2*n_gbus,2*n_buses))
#         gen_no=0
#         for g in g_bus:
#             Map_g[gen_no][g-1]=1
#             Map_g[n_gbus + gen_no][n_buses + g-1]=1
#             gen_no+=1   
            
#         Bus = pd.read_csv(os.path.join(data_dir, 'Bus.csv'))
#         l_bus=Bus['Node'].to_numpy()
#         Pd_max = Bus['Pdmax'].to_numpy()
#         Qd_max = Bus['Qdmax'].to_numpy()
#         n_lbus=len(l_bus)
#         Dem_max=np.concatenate((Pd_max,Qd_max),axis=0).reshape((2*n_lbus,1))
        
#         Map_L = np.zeros((2*n_lbus,2*n_buses))
#         l_no=0
#         for l in l_bus:
#             Map_L[l_no][l-1]=1
#             Map_L[n_lbus+l_no][n_buses+l-1]=1
#             l_no+=1
#         Y= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/Y.csv', header=None).to_numpy()
#         Yconj= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/Yconj.csv', header=None).to_numpy()
#         Ybr= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/Ybr.csv', header=None).to_numpy()
#         IM= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/IM.csv', header=None).to_numpy()
#         n_line=int(np.size(IM,0)/2)
#         L_limit= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/L_limit.csv', header=None).to_numpy().reshape(1, n_line)
        
        
#     # -----------------------------------------------------------------------------------------------
#     # True system parameters of the power system that are assumed to be known in the identification process
#     # -----------------------------------------------------------------------------------------------
    
#         true_system_parameters = {'Gen_delta': Gen_delta,
#                                   'Gen_max': Gen_max,
#                                   'Dem_max': Dem_max,
#                                   'Pd_max':Pd_max,
#                                   'Volt_max': Volt_max,
#                                   'Volt_min':Volt_min,
#                                   'Y':Y,
#                                   'Yconj':Yconj,
#                                   'Ybr':Ybr,
#                                   'Map_g':Map_g,
#                                   'Map_L':Map_L,
#                                   'IM':IM,
#                                   'g_bus':g_bus,
#                                   'n_lbus':n_lbus,
#                                   'n_line' : n_line,
#                                   'L_limit':L_limit
#                                   }
#     # -----------------------------------------------------------------------------------------------
#     # general parameters of the power system that are assumed to be known in the identification process
#     # n_buses: integer number of buses in the system
#     # -----------------------------------------------------------------------------------------------
#     general_parameters = {'n_buses': n_buses,
#                           'g_bus': g_bus,
#                           'n_gbus':n_gbus
#                           }
#     # -----------------------------------------------------------------------------------------------
#     # parameters for the training data creation 
#     # n_data_points: number of data points where measurements are present
#     # n_test_data_points: number of test data points where measurements are present
#     # s_point: starting point in the dataset from which the data sets will be collected from
#     # -----------------------------------------------------------------------------------------------
#     n_data_points = 4000
#     n_test_data_points=1000

#     data_creation_parameters = {'n_data_points': n_data_points,
#                                 'n_test_data_points': n_test_data_points,
#                                 's_point': 0}

#     # -----------------------------------------------------------------------------------------------
#     # combining all parameters in a single dictionary
#     # -----------------------------------------------------------------------------------------------
#     simulation_parameters = {'true_system': true_system_parameters,
#                              'general': general_parameters,
#                              'data_creation': data_creation_parameters}

#     return simulation_parameters


def create_example_parameters(n_buses: int):
    """
    Creates a basic set of parameters that are used in the following processes:
    * data creation if measurements are to be simulated
    * setting up the neural network model
    * training procedure

    Parameters
    ----------
    n_buses : int
        Integer number of buses in the system.
    case_path : str
        Absolute path to the MATPOWER .m case file (or other format pandapower can load).

    Returns
    -------
    simulation_parameters : dict
        Dictionary that holds all parameters.
    """
    # Initialize net to None before the try block
    net = None 
    
    # ============= specify pglib-opf case based on n_buses ==================
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
    
    if n_buses == 118:
        case_name = 'pglib_opf_case118_ieee.m'
    elif n_buses == 300:
        case_name = 'pglib_opf_case300_ieee.m'
    elif n_buses == 793:
        case_name = 'pglib_opf_case793_goc.m'
    elif n_buses == 1354:
        case_name = 'pglib_opf_case1354_pegase.m'
    elif n_buses == 2869:
        case_name = 'pglib_opf_case2869_pegase.m'
        
    case_path = os.path.join(base_dir, 'pglib-opf', case_name)

    # -----------------------------------------------------------------------------------------------
    # Load the pandapower network from the PGLib case file
    # -----------------------------------------------------------------------------------------------
    try:
        net = pc.from_mpc(case_path, casename_mpc_file=True)
        # It's good practice to run an initial power flow or OPF to populate result fields
        pp.runopp(net, verbose=False) 
    except Exception as e:
        print(f"Error loading or processing case file {case_path}: {e}")
    
    # Get the internal PYPOWER case from the pandapower network
    ppc = pc.to_ppc(net, init = 'flat') # net._ppc
    
    # Get the mapping from external (Matpower-style) to internal (Pandapower) bus indices
    external_bus_numbers = net.bus.index.values # indices from .m file from pandapower
    internal_indices = ppc['bus'][:, 0].astype(int) #  internal inddices from pypower

    external_to_internal = dict(zip(external_bus_numbers, internal_indices))
    internal_to_external = dict(zip(internal_indices, external_bus_numbers))
    
    # -----------------------------------------------------------------------------------------------
    # Extract parameters from the pandapower network object
    # -----------------------------------------------------------------------------------------------

    # Ensure n_buses matches the actual number of buses in the loaded network
    if n_buses != len(net.bus):
        print(f"Warning: n_buses ({n_buses}) passed to function does not match actual buses in case ({len(net.bus)}). Using actual.")
        n_buses = len(net.bus)
        
    n_gbus = len(net.gen) # Number of generators
    g_bus = net.gen.index.to_numpy() # Generator indices (pandapower uses 0-indexed)

    n_lbus = len(net.load) # Number of loads
    l_bus = net.load.index.to_numpy() # Load indices (pandapower uses 0-indexed)

    # Generator limits
    Pg_max = net.gen['max_p_mw'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)
    Pg_min = net.gen['min_p_mw'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)
    Qg_max = net.gen['max_q_mvar'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)
    Qg_min = net.gen['min_q_mvar'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)

    # Calculate Gen_delta and Gen_max
    epsilon = 1e-8
    if n_gbus > 0:
        Sg_delta = np.concatenate((Pg_max - Pg_min + epsilon, Qg_max - Qg_min + epsilon), axis=1).reshape((2 * n_gbus, 1))
        Sg_max = np.concatenate((Pg_max, Qg_max), axis=1).reshape((2 * n_gbus, 1))
    else:
        Sg_delta = np.array([]).reshape(0,1)
        Sg_max = np.array([]).reshape(0,1)

    # Bus voltage limits
    if 'max_vm_pu' in net.bus.columns and not net.bus.empty:
        Volt_max = net.bus['max_vm_pu'].fillna(1.10).values  # fill NaNs with 1.10
    else:
        Volt_max = np.ones(len(net.bus)) * 1.10  # default 1.10 for all buses

    if 'min_vm_pu' in net.bus.columns and not net.bus.empty:
        Volt_min = net.bus['min_vm_pu'].fillna(0.90).values  # fill NaNs with 0.90
    else:
        Volt_min = np.ones(len(net.bus)) * 0.90  # default 0.90 for all buses


    # Demand limits (using nominal load values as 'max' if no explicit max is defined)
    Pd_max_loads = net.load['p_mw'].to_numpy().reshape(-1, 1)
    Qd_max_loads = net.load['q_mvar'].to_numpy().reshape(-1, 1)
    Pd_max = Pd_max_loads # Keeping the original variable name for consistency with your dict structure
    
    # Assume Pd_min = 0 for all loads (zero minimum demand)
    Pd_min = np.zeros_like(Pd_max_loads)
    Pd_delta = Pd_max_loads - Pd_min
    Pd_min = Pd_min.reshape((n_lbus, 1))
    Pd_delta = Pd_delta.reshape((n_lbus, 1))
    Qd_min = np.zeros_like(Qd_max_loads)  # or 0.9 * Qd_max_loads etc.
    Qd_delta = Qd_max_loads - Qd_min
    
    # Stack real and reactive parts if needed
    Sd_max = np.concatenate((Pd_max_loads, Qd_max_loads), axis=0).reshape((2 * n_lbus, 1)) + epsilon
    Sd_min = np.vstack([Pd_min, Qd_min])
    Sd_delta = np.vstack([Pd_delta, Qd_delta]) + epsilon

    # Mappings (Map_g, Map_L)
    Map_g = np.zeros((2 * n_gbus, 2 * n_buses)) # for both P and Q
    # net.gen['bus'].values should contain the correct 0-based internal bus indices
    for i, external_bus_id in enumerate(net.gen['bus'].values):
        internal_bus_idx = external_to_internal[external_bus_id]
        Map_g[i, internal_bus_idx] = 1
        Map_g[n_gbus + i, n_buses + internal_bus_idx] = 1


    Map_L = np.zeros((2 * n_lbus, 2 * n_buses))
    # net.load['bus'].values should contain the correct 0-based internal bus indices
    for i, external_bus_id in enumerate(net.load['bus'].values):
        internal_bus_idx = external_to_internal[external_bus_id]
        Map_L[i, internal_bus_idx] = 1
        Map_L[n_lbus + i, n_buses + internal_bus_idx] = 1


    # ===============================================================================================
    # Admittance Matrices (Y, Yconj, Ybr, IM) and Line Limits (L_limit) using PYPOWER
    # ===============================================================================================
    try:
        # Use pypower's makeYbus to get the bus admittance matrix (Ybus),
        # and branch admittance matrices for 'from' and 'to' ends (Yf, Yt).
        # ppc['branch'] and ppc['bus'] are the PYPOWER branch and bus matrices.
        Ybus, Yf, Yt = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

        # 1. Y (Bus Admittance Matrix)
        Y = Ybus.todense()

        # 2. Yconj (Imaginary part of Ybus, often referred to as B matrix in power flow)
        Yconj = Y.imag 

        # 3. Ybr (Branch Admittance Matrix for Flows)
        # Yf and Yt are crucial for calculating branch power/current flows in AC.
        Ybr = np.vstack([Yf.real.todense(), Yf.imag.todense(), Yt.real.todense(), Yt.imag.todense()])
        
        # Determine the number of lines/branches from the PYPOWER branch data
        n_line = len(ppc['branch']) 

        # 4. IM (Incidence Matrix for Flows/Currents)
        num_branches = n_line
        num_buses_ppc = len(ppc['bus']) # Use num_buses_ppc if different from n_buses parameter
        IM = np.zeros((2 * num_branches, 2 * num_buses_ppc))

        # Get from_bus and to_bus indices from PYPOWER branch data (0-indexed)
        # These are the original bus numbers before internal reordering, if any.
        br_from = ppc['branch'][:, 0].astype(int) 
        br_to = ppc['branch'][:, 1].astype(int)   

        for i in range(num_branches):
            f_bus = br_from[i]
            t_bus = br_to[i]
            
            # Map for real part of voltages (or angles for DC equivalent)
            IM[i, f_bus] = 1
            IM[i, t_bus] = -1

            # Map for imaginary part of voltages (or magnitudes for DC equivalent)
            IM[num_branches + i, num_buses_ppc + f_bus] = 1
            IM[num_branches + i, num_buses_ppc + t_bus] = -1
        
        # 5. L_limit (Line Limits)
        # For AC OPF, this usually refers to apparent power (MVA) or current limits.
        # In PYPOWER's branch matrix, column 5 (index 5) is RATE_A (MVA limit).
        L_limit = ppc['branch'][:, 5].reshape(1, n_line)
        # Handle lines with zero or NaN limits (often means no limit) by setting a large value
        L_limit[L_limit <= 0] = 99999.0 # Placeholder for effectively infinite capacity
        L_limit[np.isnan(L_limit)] = 99999.0 # Handle NaN limits too
        
        # 6. Identify Slack, PV, and PQ buses
        BUS_TYPE = ppc['bus'][:, 1].astype(int)
        slack_buses = np.where(BUS_TYPE == 3)[0]
        pv_buses = np.where(BUS_TYPE == 2)[0]
        pq_buses = np.where(BUS_TYPE == 1)[0]

        # Slack bus index (used earlier for Bp/Bpp construction)
        slack_bus_idx = slack_buses[0] if len(slack_buses) > 0 else None
        
        # Identify generator indices with Pg_max > 0 excluding the slack generator
        gen_bus_ids = net.gen['bus'].values  # external bus IDs for each generator
        gen_bus_internal = np.array([external_to_internal[bid] for bid in gen_bus_ids])
        pg_active_indices = [i for i, (pmax, bus_idx) in enumerate(zip(Pg_max.flatten(), gen_bus_internal)) if pmax > 1e-9 and bus_idx != slack_bus_idx]
        
        # 7. Bp and Bpp matrices for Fast Decoupled Load Flow
        # ---------------------------------------------------
        # Standard method: use the imaginary part of Ybus
        B_full = Ybus.imag  # already a scipy sparse matrix

        # Remove slack bus row and column (index assumed to be 0, or fetch from ppc)
        slack_bus_idx = np.where(ppc['bus'][:, 1] == 3)[0][0]  # type 3 = slack
        buses = list(range(len(ppc['bus'])))
        pv_pq_buses = [i for i in buses if i != slack_bus_idx]

        # Extract submatrices for Bp and Bpp
        Bp = -B_full[pv_pq_buses, :][:, pv_pq_buses].tocsc()
        Bpp = -B_full[pq_buses, :][:, pq_buses].tocsc()

        # Optional: convert to dense arrays if needed for torch
        Bp_dense = np.array(Bp.todense())
        Bpp_dense = np.array(Bpp.todense())
        
        # 8. some additional stuff...
        from_buses_np = ppc['branch'][:, 0].astype(int)
        to_buses_np = ppc['branch'][:, 1].astype(int)
        R_np = ppc['branch'][:, 2]
        X_np = ppc['branch'][:, 3]


    except Exception as e:
        print(f"Error extracting admittance matrices or line limits using pypower: {e}")
        # Set to None or empty arrays to indicate failure, or raise specific error
        Y, Yconj, Ybr, IM, L_limit, n_line = None, None, None, None, None, 0
        print("Continuing with default (None/empty) values for these matrices. Downstream code may fail.")


    # -----------------------------------------------------------------------------------------------
    # True system parameters dictionary
    # -----------------------------------------------------------------------------------------------
    true_system_parameters = {'Sg_delta': Sg_delta,
                              'Sg_max': Sg_max,
                              'Sd_max': Sd_max,
                              'Sd_min': Sd_min, # This refers to nominal P for loads
                              'Sd_delta': Sd_delta, # This refers to nominal P for loads
                              'Sd_max': Sd_max, # This refers to nominal P for loads
                              'Volt_max': Volt_max,
                              'Volt_min': Volt_min,
                              'Ybus': Y,
                              'Yconj': Yconj, # Placeholder/requires specific AC derivation
                              'Ybr': Ybr,     # Placeholder/requires specific AC derivation
                              'Map_g': Map_g,
                              'Map_L': Map_L,
                              'IM': IM,       # Placeholder/requires specific AC derivation
                              'g_bus': g_bus, # 0-indexed pandapower generator bus indices
                              'n_lbus': n_lbus,
                              'n_line': n_line, # Updated to actual branches
                              'L_limit': L_limit, # Placeholder/requires specific AC derivation
                              'Bp': Bp_dense,
                              'Bpp': Bpp_dense,
                              'pq_buses': pq_buses,
                              'pv_buses': pv_buses,
                              'pg_active': pg_active_indices,
                              'slack_bus': slack_bus_idx,
                              'from_bus': from_buses_np,
                              'to_bus': to_buses_np,
                              'R': R_np,
                              'X': X_np,

                              }

    # -----------------------------------------------------------------------------------------------
    # General parameters dictionary
    # -----------------------------------------------------------------------------------------------
    general_parameters = {'n_buses': n_buses,
                          'g_bus': g_bus,
                          'n_gbus': n_gbus
                          }

    # -----------------------------------------------------------------------------------------------
    # Parameters for data creation
    # -----------------------------------------------------------------------------------------------
    n_data_points = 5_000
    n_test_data_points = 2_000

    data_creation_parameters = {'n_data_points': n_data_points,
                                'n_test_data_points': n_test_data_points,
                                's_point': 0}

    # -----------------------------------------------------------------------------------------------
    # Combining all parameters in a single dictionary
    # -----------------------------------------------------------------------------------------------
    simulation_parameters = {'true_system': true_system_parameters,
                             'general': general_parameters,
                             'data_creation': data_creation_parameters,
                             'net_object': ppc, # Optionally, pass the net object itself if other functions need it
                             'nn_output': 'pg_qg'}

    return simulation_parameters
