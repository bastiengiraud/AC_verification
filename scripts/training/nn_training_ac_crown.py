import time
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data', 'models', 'scripts/utils']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

from ac_opf.create_example_parameters import create_example_parameters
from ac_opf.create_data import create_data, create_test_data
from EarlyStopping import EarlyStopping
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from neural_network.lightning_nn_crown import NeuralNetwork
# from LiRPANet import LiRPANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_np(x):
    return x.detach().numpy()

def train(config):
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)

    # Training Data
    Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)
    Dem_train = torch.tensor(Dem_train).float().to(device)
    Gen_train = torch.tensor(Gen_train).float().to(device)
    Gen_train_typ = torch.ones(Gen_train.shape[0], 1).to(device)

    num_classes = Gen_train.shape[1]
    Gen_delta = simulation_parameters['true_system']['Sg_delta']
    Dem_min = simulation_parameters['true_system']['Sd_min']
    Dem_delta = simulation_parameters['true_system']['Sd_delta']

    Data_stat = {
        'Gen_delta': Gen_delta,
        'Dem_min': Dem_min,
        'Dem_delta': Dem_delta,
    }

    # Test Data
    Dem_test, Gen_test = create_test_data(simulation_parameters=simulation_parameters)
    Dem_test = torch.tensor(Dem_test).float().to(device)
    Gen_test = torch.tensor(Gen_test).float().to(device)

    network_gen = build_network(Dem_train.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed)
    # network_gen = normalise_network(network_gen, Dem_train, Data_stat) # data is already normalized

    # Convert NN to lirpa_model that can calculate bounds on output
    lirpa_model = BoundedModule(network_gen, torch.empty_like(Dem_train), device=device) 
    print('Running on', device)

    x = Dem_min.reshape(1, -1) + Dem_delta.reshape(1, -1) / 2
    x = torch.tensor(x).float().to(device)
    x_min = torch.tensor(Dem_min.reshape(1, -1)).float().to(device)
    x_max = torch.tensor(Dem_min.reshape(1, -1) + Dem_delta.reshape(1, -1)).float().to(device)
    
    # set up input specificiation. Define upper and lower bound. Boundedtensor wraps nominal input(x) and associates it with defined perturbation ptb.
    ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
    image = BoundedTensor(x, ptb).to(device)

    optimizer = torch.optim.Adam(network_gen.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) ** -config.lr_decay)

    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_save_directory = os.path.join(project_root_dir, 'models', 'best_model')
    path = f'checkpoint_{n_buses}_{config.hidden_layer_size}_{config.Algo}.pt'
    path_dir = os.path.join(model_save_directory, path)
    early_stopping = EarlyStopping(patience=100, verbose=False, NN_input=Dem_train, path=path_dir)

    train_losses = []
    test_losses = []

    for epoch in range(config.epochs):
        # after every 100 epochs, enrich dataset with worst-case data
        if epoch % 100 == 0 and epoch != 0 and config.Enrich:
            X, Y, typ = wc_enriching(network_gen, config, Dem_train, Data_stat)
            InputNN = torch.cat((Dem_train, X), 0).to(device)
            OutputNN = torch.cat((Gen_train, Y), 0).to(device)
            typNN = torch.cat((Gen_train_typ, typ), 0).to(device)
            idx = torch.randperm(InputNN.shape[0])
            InputNN, OutputNN, typNN = InputNN[idx], OutputNN[idx], typNN[idx]
        else:
            InputNN = Dem_train
            OutputNN = Gen_train
            typNN = Gen_train_typ

        start_time = time.time()
        mse_criterion, training_loss = train_epoch(network_gen, InputNN, OutputNN, typNN, optimizer, config, simulation_parameters, epoch)
        validation_loss = validate_epoch(network_gen, Dem_test, Gen_test)
        training_time = time.time() - start_time
        
        if epoch % 20 == 0 and epoch != 0:
            # Print MSE losses for both train and validation
            print(f"Epoch {epoch+1}/{config.epochs} — Train total loss: {training_loss:.6f}, Train MSE: {mse_criterion:.6f}, Validation MSE: {validation_loss:.6f}")

        train_losses.append(training_loss.item())
        test_losses.append(validation_loss.item())

        early_stopping(validation_loss, network_gen)

        # loss_weight = config.LPF_weight / (1 + epoch * 0.01)
        # lb, ub = lirpa_model.compute_bounds(x=(image,), method=config.abc_method) 
        # PF_violation = torch.abs(ub) + torch.abs(lb)

        # # after 100 epochs, start adding wc PF violation penalty
        # if config.Algo and epoch >= 100:
        #     optimizer.zero_grad()
        #     pf_loss = loss_weight * PF_violation 
        #     pf_loss.backward()
        #     optimizer.step()

        scheduler.step()

        if early_stopping.early_stop:
            break
    
    # store as .h5 model
    best_model = torch.load(path_dir)
    early_stopping.export_to_h5(best_model)
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("Early stopping")


def build_network(n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed):
    hidden_layer_size = [hidden_layer_size] * n_hidden_layers
    model = NeuralNetwork(n_input_neurons, hidden_layer_size=hidden_layer_size,
                          num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed)
    return model.to(device)


def normalise_network(model, Dem_train, Data_stat):
    pd_min = Data_stat['Dem_min']
    pd_delta = Data_stat['Dem_delta']
    pg_delta = Data_stat['Gen_delta']

    input_stats = (torch.from_numpy(pd_min.reshape(-1,).astype(np.float64)),
                   torch.from_numpy(pd_delta.reshape(-1,).astype(np.float64)))
    output_stats = torch.from_numpy(pg_delta.reshape(-1,).astype(np.float64))

    model.normalise_input(input_stats)
    model.normalise_output(output_stats)
    return model.to(device)


def validate_epoch(network_gen, Dem_test, Gen_test):
    criterion = nn.MSELoss()
    output = network_gen.forward_train(Dem_test)
    return criterion(output, Gen_test)


def train_epoch(network_gen, Dem_train, Gen_train, typ, optimizer, config, simulation_parameters, epoch):
    torch.autograd.set_detect_anomaly(True)

    network_gen.train()
    criterion = nn.MSELoss()
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    num_samples = Dem_train.shape[0]
    num_batches = num_samples // config.batch_size
    Gen_delta = simulation_parameters['true_system']['Sg_delta']
    n_gens = simulation_parameters['general']['n_gbus']
    n_loads = Dem_train.shape[1]
    n_bus = simulation_parameters['general']['n_buses']
    RELU = nn.ReLU()
    loss_sum = 0
    
    # get limits
    pg_max_values = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float64, device=device, requires_grad=False)[:n_gens] / 100 
    pg_min_values = torch.zeros_like(pg_max_values, requires_grad=False)
    vm_min_values = torch.tensor(simulation_parameters['true_system']['Volt_min'], dtype=torch.float64, device=device).unsqueeze(1)
    vm_max_values = torch.tensor(simulation_parameters['true_system']['Volt_max'], dtype=torch.float64, device=device).unsqueeze(1)
    
    # placeholder for Pg, and fill in with NN prediction
    Pg_place = torch.zeros((config.batch_size, n_gens), dtype=torch.float64, device=device, requires_grad=True)
    Vm_nn_place = torch.zeros((config.batch_size, n_bus), dtype=torch.float64, device=device, requires_grad=True)
    
    # Add a small epsilon to denominators if you want (optional, but you had it for stability)
    eps = 1e-6
    pg_max_values = pg_max_values + eps
    pg_min_values = pg_min_values - eps
    pg_denominator = pg_max_values - pg_min_values

    vm_max_values = vm_max_values + eps
    vm_min_values = vm_min_values - eps
    vm_denominator = vm_max_values - vm_min_values
    
    # get indices of active gens and pv buses
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices) # we're only predicting gens with pg_max > 0
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
 
    preds = []
    targets = []

    for i in range(num_batches):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        Gen_output = network_gen.forward_train(Dem_train[slce])
        Gen_target = Gen_train[slce]
        
        # compute actual pg and vm
        Pg_active = Gen_output[:, :n_act_gens] 
        Pg = Pg_place.clone()
        Pg[:, act_gen_indices] = Pg_active.to(dtype=torch.float64)
        
        Vm_nn_g = Gen_output[:, n_act_gens:] 
        Vm_nn = Vm_nn_place.clone()
        Vm_nn[:, pv_indices] = Vm_nn_g.to(dtype=torch.float64)
        
        # Inverse scaling of all states
        Pg = Pg * pg_denominator.T + pg_min_values.T  # shape (batch_size, n_gens)
        Vm_nn = Vm_nn * vm_denominator.T + vm_min_values.T  # shape (batch_size, n_bus)
        
        # # true labeled gen output
        # Pg_active_true = Gen_target[:, :n_act_gens] 
        # Pg_true = Pg_place.clone()
        # Pg_true[:, act_gen_indices] = Pg_active_true.to(dtype=torch.float64)
        # Pg_true = Pg_true * pg_denominator.T + pg_min_values.T
        
        mse_criterion = criterion(Gen_output * typ[slce], Gen_target * typ[slce])
        loss_gen = config.crit_weight * mse_criterion                 # supervised learning, difference from label
        violation_pg = config.pg_viol_weight * (                                                   # generator limit violation penatly   
            torch.mean(RELU(Pg - pg_max_values.T) ** 2) +
            torch.mean((RELU(0 - Pg)) ** 2)
        )
        violation_vm = config.vm_viol_weight * (                                                   # voltage limit violation penatly   
            torch.mean(RELU(Vm_nn - vm_max_values.T) ** 2) +
            torch.mean(RELU(vm_min_values.T - Vm_nn) ** 2)
        )
        
        
        if epoch > 100:
            # print(torch.mean(Pg_true, dim=0) - torch.mean(Pg, dim = 0))
            PF_loss = config.PF_weight * full_flow_check(Dem_train[slce], Gen_output, simulation_parameters)  
        else:
            PF_loss = 0

        total_loss = loss_gen + violation_pg + violation_vm + PF_loss

        total_loss.backward()
        optimizer.step()
        loss_sum += loss_gen
    
    loss_epoch = loss_sum / num_batches

    return mse_criterion, loss_epoch


"""" 
convex relaxation of the OP for power flow.

"""

def linearized_pf_opt(nn_input, nn_output, simulation_parameters):
    """ 
    My NN only predicts the active power of generators which have pg_max > 0 and which are not the slack bus.
    The NN also predicts voltages at all generators. 
    
    """
    device = nn_output.device
    
    # get indices of active gens and pv buses
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices) # we're only predicting gens with pg_max > 0
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
    
    # general system params
    n_gens = simulation_parameters['general']['n_gbus']
    n_loads = simulation_parameters['true_system']['n_lbus']
    n_bus = simulation_parameters['general']['n_buses']
    
    # Extract min/max Pg per generator:
    pg_max_values = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float64, device=device)[:n_gens] / 100 # divide by Sbase
    pg_min_values = torch.zeros_like(pg_max_values)
    vm_min_values = torch.tensor(simulation_parameters['true_system']['Volt_min'], dtype=torch.float64, device=device).unsqueeze(1)
    vm_max_values = torch.tensor(simulation_parameters['true_system']['Volt_max'], dtype=torch.float64, device=device).unsqueeze(1)
    pd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[:n_loads] / 100 # divide by Sbase
    qd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[n_loads:] / 100 # divide by Sbase
    
    # placeholder for Pg, and fill in with NN prediction
    Pg = torch.zeros((nn_output.shape[0], n_gens), dtype=torch.float64, device=device)
    Pg_active = nn_output[:, :n_act_gens] 
    Pg = Pg.clone()
    Pg[:, act_gen_indices] = Pg_active.to(dtype=torch.float64)
    
    # placeholder for Vm, and fill in with NN prediction
    Vm_nn = torch.zeros((nn_output.shape[0], n_bus), dtype=torch.float64, device=device)
    Vm_nn_g = nn_output[:, n_act_gens:] 
    Vm_nn = Vm_nn.clone()
    Vm_nn[:, pv_indices] = Vm_nn_g.to(dtype=torch.float64)
    
    # Fill in loads from NN input
    Pd = nn_input[:, :n_loads] 
    Qd = nn_input[:, n_loads:]
    
    # Add a small epsilon to denominators if you want (optional, but you had it for stability)
    eps = 1e-6
    pg_max_values += eps
    pg_min_values -= eps
    pg_denominator = pg_max_values - pg_min_values

    vm_max_values += eps
    vm_min_values -= eps
    vm_denominator = vm_max_values - vm_min_values
    
    # Inverse scaling of all states
    Pg = Pg * pg_denominator.T + pg_min_values.T  # shape (batch_size, n_gens)
    Vm_nn = Vm_nn * vm_denominator.T + vm_min_values.T  # shape (batch_size, n_bus)
    Pd = Pd * pd_nom.T
    Qd = Qd * qd_nom.T
    

    # mapping matrices gens and loads to buses
    Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float64, device=device)    # (n_bus-1, n_bus-1)
    Map_l = torch.tensor(simulation_parameters['true_system']['Map_L'], dtype=torch.float64, device=device)
    
    # map injections to buses
    Pg = Pg @ Map_g[:n_gens, :n_bus] 
    Pd = Pd @ Map_l[:n_loads, :n_bus] 
    Qd = Qd @ Map_l[n_loads:, n_bus:] 
    
    assert Pg_active.shape[1] == len(act_gen_indices)
    assert Vm_nn_g.shape[1] == len(pv_indices)

    # Run FDLF to get Vm and delta
    Vm, delta, Qg = fdlf_solver(Pg, Pd, Qd, Vm_nn, simulation_parameters)

    # For now, let's just call the original function as an example
    violation_loss, S_line = ac_power_flow_check(Vm, delta, simulation_parameters)


    return violation_loss




def fdlf_solver(Pg, Pd, Qd, Vm_nn, simulation_parameters, max_iter=100, tol=1e-3):
    """
    Difficult to converge if power balance not satisfied.
    Inputs:
        Pg: Active power generation from NN (batch_size, n_bus)
        Pd, Qd: Loads at buses (batch_size, n_bus)
        Vm_nn: Voltage magnitude at generator buses from NN (batch_size, n_bus)
        simulation_parameters: contains Ybus, Bp, Bpp, slack_bus, bus types etc.
        
    Outputs:
        Vm: voltage magnitudes (fixed at PV buses = Vm_nn)
        delta: voltage angles (solved)
        Qg: reactive power generation (computed after convergence)
    """
    device = Pg.device
    batch_size, n_bus = Pg.shape[0], simulation_parameters['general']['n_buses']
    n_gens, n_loads = simulation_parameters['general']['n_gbus'], simulation_parameters['true_system']['n_lbus']

    # Extract matrices (assumed precomputed and stored in simulation_parameters)
    Bp = torch.tensor(simulation_parameters['true_system']['Bp'], dtype=torch.float64, device=device)    # (n_bus-1, n_bus-1)
    Bpp = torch.tensor(simulation_parameters['true_system']['Bpp'], dtype=torch.float64, device=device)  # (n_bus-1, n_bus-1)
    Ybus = torch.tensor(simulation_parameters['true_system']['Ybus'], dtype=torch.complex128, device=device)  # complex matrix
    
    # Add regularization
    # ε = 1e-6
    # Ip = torch.eye(Bp.shape[0], dtype=torch.float64, device=device)
    # Ipp = torch.eye(Bpp.shape[0], dtype=torch.float64, device=device)
    # Bp = Bp + ε * Ip
    # Bpp = Bpp + ε * Ipp
    
    Bp = torch.linalg.inv(Bp)
    Bpp = torch.linalg.inv(Bpp)

    slack_bus = simulation_parameters['true_system']['slack_bus']
    pq_indices = torch.tensor(simulation_parameters['true_system']['pq_buses'], dtype=torch.long, device=device)
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
    pv_pq_indices = torch.cat([pv_indices, pq_indices])  # (n_bus - 1,)

    # Calculate net active power injection at each bus: P_inj = Pg - Pd
    P_inj = Pg  - Pd 

    # Initialize voltages: Vm = 1.0 p.u., delta = 0.0 radians
    Vm = torch.ones((batch_size, n_bus), dtype=torch.float64, device=device)
    delta = torch.zeros((batch_size, n_bus), dtype=torch.float64, device=device, requires_grad=True)

    # Set Vm at PV buses from NN output
    Vm = Vm.clone()
    Vm[:, pv_indices] = Vm_nn[:, pv_indices]

    max_step_d = 0.005
    max_step_v = 0.5
    

    for iteration in range(max_iter):
        # Calculate complex voltages
        V_complex = Vm * torch.exp(1j * delta)

        # Calculate complex power injections S = V * conj(Ybus * V)
        I_calc = torch.matmul(Ybus, V_complex.unsqueeze(-1)).squeeze(-1)
        S_calc = V_complex * torch.conj(I_calc)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        # Calculate mismatches
        dP = P_inj[:, pv_pq_indices] - P_calc[:, pv_pq_indices]  # active power mismatch
        
        if iteration == 0:
            print("Initial ||ΔP||:", torch.norm(dP, dim = 1).mean().item())
            #print(torch.mean(dP, dim = 0))
        
        dP=torch.divide(dP,Vm[:, pv_pq_indices])

        dP = torch.clamp(dP, min=-1, max=1)

        dDelta = torch.mm(Bp, dP.T).T   # (batch_size, n_bus-1)

        # Use dP and dQ norms as scale indicators
        mismatch_dP_norm = torch.norm(dP, dim=1, keepdim=True)  # shape (batch_size, 1)
        norm_dDelta = torch.norm(dDelta, dim=1, keepdim=True) + 1e-8

        # Gradient-descent style scale: move opposite mismatch direction with diminishing step
        scale_dDelta = torch.clamp(0.1 * mismatch_dP_norm / norm_dDelta, max=max_step_d)

        # Update delta and Vm separately
        delta_new = delta.clone()
        delta_new[:, pv_pq_indices] = delta[:, pv_pq_indices] + scale_dDelta * dDelta
        delta = delta_new
        
        # =========================
        
        # Calculate complex voltages
        V_complex = Vm * torch.exp(1j * delta)

        # Calculate complex power injections S = V * conj(Ybus * V)
        I_calc = torch.matmul(Ybus, V_complex.unsqueeze(-1)).squeeze(-1)
        S_calc = V_complex * torch.conj(I_calc)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        # Calculate mismatches
        dQ = - Qd[:, pq_indices] - Q_calc[:, pq_indices]            # reactive power mismatch (load buses only)
        dQ=torch.divide(dQ,Vm[:, pq_indices])
        dVm = torch.mm(Bpp, dQ.T).T     # (batch_size, n_bus-1)
        
        # Use dP and dQ norms as scale indicators
        mismatch_dQ_norm = torch.norm(dQ, dim=1, keepdim=True)
        norm_dVm = torch.norm(dVm, dim=1, keepdim=True) + 1e-8

        # Gradient-descent style scale: move opposite mismatch direction with diminishing step
        scale_dVm = torch.clamp(0.1 * mismatch_dQ_norm / norm_dVm, max=max_step_v)
        
        Vm_new = Vm.clone()
        Vm_new[:, pq_indices] = Vm[:, pq_indices] + scale_dVm * dVm
        Vm = Vm_new
        

        
        
        print(torch.max(torch.abs(dDelta)), torch.max(torch.abs(dVm)))
        print(torch.norm(dP, dim = 1).mean().item(), torch.norm(dQ, dim = 1).mean().item())
        # print(torch.sum(converged_mask).item())

        # Check convergence
        if torch.max(torch.abs(dDelta)) < tol and torch.max(torch.abs(dVm)) < tol:
            print("OMGGG CONVERGED!")
            break
        elif iteration == max_iter - 1: 
            print(f"dDelta is {torch.max(torch.abs(dDelta))}, dVm is {torch.max(torch.abs(dVm))}. FDLF not converged...")
            print(torch.mean(dP, dim = 0))
            print(torch.mean(dDelta, dim = 0))
            raise RuntimeError(f"FDLF did not converge after {max_iter} iterations.")

    assert torch.all(delta[:, slack_bus] == 0.0), "delta at slack bus is not zero"
    assert torch.all(Vm[:, slack_bus] == 1.0), "voltage at slack bus is not zero"
    
    # # Fix slack bus values explicitly
    # delta = delta.clone()
    # Vm = Vm.clone()

    # delta[:, slack_bus] = 0.0
    # Vm[:, slack_bus] = 1.0

    # After convergence, compute Qg at generator buses:
    # Qg = Q_calc at PV buses + Qd at PV buses (since loads don’t exist at generator buses, usually Qd=0 there)
    Qg = Q_calc[:, pv_indices] + Qd[:, pv_indices]

    return Vm, delta, Qg





def ac_power_flow_check(Vm, delta, simulation_parameters):
    """
    Compute line flows using voltage magnitudes and angles from FDLF.
    Inputs:
        Vm: (batch_size, n_bus) voltage magnitudes (pu)
        delta: (batch_size, n_bus) voltage angles (rad)
        simulation_parameters: dict containing line data and limits
    Returns:
        total_violation_loss: scalar tensor measuring line flow violations
        S_line: (batch_size, n_lines) complex power flows
    """
    device = Vm.device
    RELU = nn.ReLU()

    # Extract system data
    R = torch.tensor(simulation_parameters['true_system']['R'], dtype=torch.float64, device=device)
    X = torch.tensor(simulation_parameters['true_system']['X'], dtype=torch.float64, device=device)
    Pl_max = torch.tensor(simulation_parameters['true_system']['L_limit'], dtype=torch.float64, device=device)

    from_buses = torch.tensor(simulation_parameters['true_system']['from_bus'], dtype=torch.long, device=device)
    to_buses = torch.tensor(simulation_parameters['true_system']['to_bus'], dtype=torch.long, device=device)

    # Convert to complex voltages
    V_complex = Vm.type(torch.complex128) * torch.exp(1j * delta.type(torch.complex128))  # (batch_size, n_bus)

    # Voltages at line ends
    V_from = V_complex[:, from_buses]  # (batch_size, n_lines)
    V_to = V_complex[:, to_buses]      # (batch_size, n_lines)

    # Line impedance
    Z = R + 1j * X  # (n_lines,)

    # Line currents
    I_line = (V_from - V_to) / Z  # (batch_size, n_lines)

    # Complex power flows
    S_line = V_from * torch.conj(I_line)  # (batch_size, n_lines)

    # Apparent power magnitudes
    S_magnitude = torch.abs(S_line) * 100 # (batch_size, n_lines)

    # Violations
    violation_upper = RELU(S_magnitude - Pl_max)

    # Total violation loss
    total_violation_loss = torch.mean(violation_upper**2)

    return total_violation_loss, S_line




def full_flow_check(nn_input, nn_output, simulation_parameters):
    """ 
    My NN only predicts the active power of generators which have pg_max > 0 and which are not the slack bus.
    The NN also predicts voltages at all generators. 
    
    """
    device = nn_output.device
    
    # get indices of active gens and pv buses
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices) # we're only predicting gens with pg_max > 0
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
    
    # general system params
    n_gens = simulation_parameters['general']['n_gbus']
    n_loads = simulation_parameters['true_system']['n_lbus']
    n_bus = simulation_parameters['general']['n_buses']
    
    # Extract min/max Pg per generator:
    pg_max_values = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float64, device=device)[:n_gens] / 100 # divide by Sbase
    pg_min_values = torch.zeros_like(pg_max_values)
    vm_min_values = torch.tensor(simulation_parameters['true_system']['Volt_min'], dtype=torch.float64, device=device).unsqueeze(1)
    vm_max_values = torch.tensor(simulation_parameters['true_system']['Volt_max'], dtype=torch.float64, device=device).unsqueeze(1)
    pd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[:n_loads] / 100 # divide by Sbase
    qd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[n_loads:] / 100 # divide by Sbase
    
    # placeholder for Pg, and fill in with NN prediction
    Pg = torch.zeros((nn_output.shape[0], n_gens), dtype=torch.float64, device=device)
    Pg_active = nn_output[:, :n_act_gens] 
    Pg = Pg.clone()
    Pg[:, act_gen_indices] = Pg_active.to(dtype=torch.float64)
    
    # placeholder for Vm, and fill in with NN prediction
    Vm_nn = torch.zeros((nn_output.shape[0], n_bus), dtype=torch.float64, device=device)
    Vm_nn_g = nn_output[:, n_act_gens:] 
    Vm_nn = Vm_nn.clone()
    Vm_nn[:, pv_indices] = Vm_nn_g.to(dtype=torch.float64)
    
    # Fill in loads from NN input
    Pd = nn_input[:, :n_loads] 
    Qd = nn_input[:, n_loads:]
    
    # Add a small epsilon to denominators if you want (optional, but you had it for stability)
    eps = 1e-6
    pg_max_values += eps
    pg_min_values -= eps
    pg_denominator = pg_max_values - pg_min_values

    vm_max_values += eps
    vm_min_values -= eps
    vm_denominator = vm_max_values - vm_min_values
    
    # Inverse scaling of all states
    Pg = Pg * pg_denominator.T + pg_min_values.T  # shape (batch_size, n_gens)
    Vm_nn = Vm_nn * vm_denominator.T + vm_min_values.T  # shape (batch_size, n_bus)
    Pd = Pd * pd_nom.T
    Qd = Qd * qd_nom.T
    
    # print(torch.mean(Pg, dim= 0))
    # print(torch.mean(Pd, dim= 0))
    
    # print(torch.sum(torch.mean(Pg, dim= 0)))
    # print(torch.sum(torch.mean(Pd, dim= 0)))
    
    # mapping matrices gens and loads to buses
    Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float64, device=device)    # (n_bus-1, n_bus-1)
    Map_l = torch.tensor(simulation_parameters['true_system']['Map_L'], dtype=torch.float64, device=device)
    
    # map injections to buses
    Pg = Pg @ Map_g[:n_gens, :n_bus] 
    Pd = Pd @ Map_l[:n_loads, :n_bus] 
    Qd = Qd @ Map_l[n_loads:, n_bus:] 
    
    assert Pg_active.shape[1] == len(act_gen_indices)
    assert Vm_nn_g.shape[1] == len(pv_indices)

    # Run FDLF to get Vm and delta
    Vm, delta, Qg = fdlf_solver(Pg, Pd, Qd, Vm_nn, simulation_parameters)

    # For now, let's just call the original function as an example
    violation_loss, S_line = ac_power_flow_check(Vm, delta, simulation_parameters)


    return violation_loss



def wc_enriching(network_gen, config, Dem_train, Data_stat):
    n_adver = config.N_enrich

    # Forward pass to get generator outputs (Pg and Vm stacked)
    nn_output = network_gen.forward_aft(Dem_train).cpu().detach().numpy()
    
    n_gens = Data_stat['Gen_delta'].shape[0] // 2 
    n_loads = Dem_train.shape[1] // 2  # Pd and Qd stacked, so loads = half of input features

    # Split loads into Pd and Qd
    P_gen = nn_output[:, :n_gens]
    P_load = Dem_train[:, :n_loads].cpu().numpy()  # shape (n_loads, n_samples)

    # Only consider active power balance violation (Pg - Pd)
    # Sum over generators and loads per sample (axis=0)
    PB_P = np.sum(P_gen, axis=1) - np.sum(P_load, axis=1)  # shape (n_samples,)

    # Find indices with largest positive violations (overgeneration)
    ind_p = np.argpartition(PB_P, -n_adver // 2)[-n_adver // 2:]
    adv_p = GradAscnt(network_gen, Dem_train[ind_p, :].cpu().numpy(), Data_stat, sign=1)

    # Find indices with largest negative violations (undergeneration)
    ind_n = np.argpartition(-PB_P, -n_adver // 2)[-n_adver // 2:]
    adv_n = GradAscnt(network_gen, Dem_train[ind_n, :].cpu().numpy(), Data_stat, sign=-1)

    x_g = torch.tensor(np.concatenate([adv_n, adv_p], axis=0)).float()  # concatenate on sample axis
    y_g = torch.zeros(x_g.shape[0], nn_output.shape[1])
    y_type = torch.zeros(x_g.shape[0], 1)

    return x_g, y_g, y_type





def GradAscnt(Network, x_starting, Data_stat, sign=-1, Num_iteration=100, lr=0.0001):
    '''
    x_starting: Starting points for gradient ascent (shape: features x batch_size)
    sign: 1 to increase violation, -1 to decrease violation
    '''
    x = torch.tensor(x_starting, requires_grad=True).float()
    optimizer = torch.optim.SGD([x], lr=lr)

    n_gens = Data_stat['Gen_delta'].shape[0] // 2 
    n_loads = x.shape[1] // 2

    for _ in range(Num_iteration):
        optimizer.zero_grad()
        nn_output = Network.forward_aft(x)  # shape (n_gens, batch_size)

        # Active power from generator
        P_gen = nn_output[:, :n_gens]

        # Active power from load (input)
        P_load = x[:, :n_loads]  # Pd

        # Power balance violation (Pg - Pd)
        PB_P = torch.sum(P_gen, dim=1) - torch.sum(P_load, dim=1)

        loss = sign * torch.mean(PB_P)  # mean violation scaled by sign

        loss.backward()
        optimizer.step()

    return x.detach().numpy()


# def power_flow_check(P_Loads, P_Gens, simulation_parameters):
#     PTDF = torch.tensor(simulation_parameters['true_system']['PTDF'].to_numpy().astype(np.float64)).to(device)
#     Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'].astype(np.float64)).to(device)
#     Map_L = torch.tensor(simulation_parameters['true_system']['Map_L'].astype(np.float64)).to(device)
#     Pl_max = torch.tensor(simulation_parameters['true_system']['Pl_max'].astype(np.float64)).to(device)
#     RELU = nn.ReLU() # Used for violation calculation

#     # 1. Map generator outputs and loads to bus injections
#     P_gen_bus = P_Gens @ Map_g 
#     P_load_bus = P_Loads @ Map_L 

#     # 2. Calculate net injection at each bus
#     P_net_bus = P_gen_bus - P_load_bus

#     # 3. Calculate line flows
#     Line_flows_raw_T = PTDF.T @ P_net_bus.T
#     Line_flows_raw = Line_flows_raw_T.T

#     # 4. Calculate violations for upper and lower limits
#     violation_upper = RELU(Line_flows_raw - Pl_max) # (batch_size, n_lines)
#     violation_lower = RELU(-Pl_max - Line_flows_raw) # (batch_size, n_lines)

#     # 5. Compute a loss based on the violations (e.g., sum of squares or mean)
#     total_violation_loss = torch.mean(violation_upper**2 + violation_lower**2)
    
#     return total_violation_loss



# def wc_enriching(network_gen, config, Dem_train, Data_stat):
#     n_adver = config.N_enrich
#     Gen_output = network_gen.forward_aft(Dem_train).cpu().detach().numpy() # forward pass with clamping
#     PB = np.sum(Gen_output, axis=1) - np.sum(Dem_train.cpu().numpy(), axis=1) # power balance violation
    
#     # over generation - positive gradient ascent samples
#     ind_p = np.argpartition(PB, -4)[-n_adver // 2:] # identify 'n_adver' indices with worst-case violation
#     adv_p = GradAscnt(network_gen, Dem_train[ind_p].cpu().numpy(), Data_stat, sign=1)
    
#     # under generation - negative gradient descent samples
#     ind_n = np.argpartition(-PB, -4)[-n_adver // 2:]
#     adv_n = GradAscnt(network_gen, Dem_train[ind_n].cpu().numpy(), Data_stat)
#     x_g = torch.tensor(np.concatenate([adv_n, adv_p], axis=0)).float()
#     y_g = torch.zeros(x_g.shape[0], Gen_output.shape[1])
#     y_type = torch.zeros(x_g.shape[0], 1)
#     return x_g, y_g, y_type




# def GradAscnt(Network, x_starting, Data_stat, sign=-1, Num_iteration=100, lr=0.0001):
#     '''
#     x_starting: Starting points for the gradient ascent algorithm
#     x_min,x_max :  Minimum and maximum value of x ( default is 0 and 1)
#     Sign : direction for gradient ascent ( 1 --> Increase the violation , -1 --> reduce the violation (.i.e. make it more negative))
#     Num_iteration: Number of gradient steps
#     lr: larning rate
#     '''
#     x = torch.tensor(x_starting, requires_grad=True).float()
#     optimizer = torch.optim.SGD([x], lr=lr)

#     for _ in range(Num_iteration):
#         optimizer.zero_grad()
#         output = Network.forward_aft(x) # forward pass with clamping
#         PB = torch.sum(output, dim=1) # power balance
#         loss = sign * torch.mean(PB)
#         loss.backward()
#         optimizer.step()

#     return x.detach().numpy()