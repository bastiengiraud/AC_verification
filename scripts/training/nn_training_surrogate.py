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
from neural_network.lightning_nn_surrogate import NeuralNetwork
# from LiRPANet import LiRPANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_np(x):
    return x.detach().numpy()

def train(config):
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)

    # Training Data
    inj_train, vrvi_train = create_data(simulation_parameters=simulation_parameters)
    inj_train = torch.tensor(inj_train).float().to(device)
    vrvi_train = torch.tensor(vrvi_train).float().to(device)
    Gen_train_typ = torch.ones(vrvi_train.shape[0], 1).to(device)

    num_classes = vrvi_train.shape[1]
    _, inj_min, inj_max = min_max_scale_tensor(inj_train)
    inj_delta = inj_max - inj_min
    inj_delta[inj_delta <= 1e-12] = 1.0  # Safeguard

    _, vrvi_min, vrvi_max = min_max_scale_tensor(vrvi_train)
    vrvi_delta = vrvi_max - vrvi_min
    vrvi_delta[vrvi_delta <= 1e-12] = 1.0

    data_stat = {
        'inj_min': inj_min,
        'inj_delta': inj_delta,
        'vrvi_min': vrvi_min,
        'vrvi_delta': vrvi_delta,
    }


    # Test Data
    inj_test, vrvi_test = create_test_data(simulation_parameters=simulation_parameters)
    inj_test = torch.tensor(inj_test).float().to(device)
    vrvi_test = torch.tensor(vrvi_test).float().to(device)

    network_gen = build_network(inj_train.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed)
    network_gen = normalise_network(network_gen, inj_train, data_stat) # data is already normalized

    optimizer = torch.optim.Adam(network_gen.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) ** -config.lr_decay)

    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_save_directory = os.path.join(project_root_dir, 'models', 'surrogate')
    
    # ✅ Create the directory if it doesn't exist
    os.makedirs(model_save_directory, exist_ok=True)

    path = f'checkpoint_surrogate_{n_buses}_{config.hidden_layer_size}_{config.Algo}.pt'
    path_dir = os.path.join(model_save_directory, path)
    early_stopping = EarlyStopping(patience=100, verbose=False, NN_input=inj_train, path=path_dir)

    train_losses = []
    test_losses = []

    for epoch in range(config.epochs):
        # after every 100 epochs, enrich dataset with worst-case data
        if epoch % 100 == 0 and epoch != 0 and config.Enrich:
            X, Y, typ = wc_enriching(network_gen, config, inj_train, data_stat)
            InputNN = torch.cat((inj_train, X), 0).to(device)
            OutputNN = torch.cat((vrvi_train, Y), 0).to(device)
            typNN = torch.cat((Gen_train_typ, typ), 0).to(device)
            idx = torch.randperm(InputNN.shape[0])
            InputNN, OutputNN, typNN = InputNN[idx], OutputNN[idx], typNN[idx]
        else:
            InputNN = inj_train
            OutputNN = vrvi_train
            typNN = Gen_train_typ

        start_time = time.time()
        mse_criterion, training_loss = train_epoch(network_gen, InputNN, OutputNN, typNN, optimizer, config, simulation_parameters, epoch)
        validation_loss = validate_epoch(network_gen, inj_test, vrvi_test)
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
    
def min_max_scale_tensor(data):
    data_min = data.min(dim=0, keepdim=True).values
    data_max = data.max(dim=0, keepdim=True).values
    scaled = (data - data_min) / (data_max - data_min + 1e-8)
    return scaled, data_min, data_max


def build_network(n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed):
    hidden_layer_size = [hidden_layer_size] * n_hidden_layers
    model = NeuralNetwork(n_input_neurons, hidden_layer_size=hidden_layer_size,
                          num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed)
    return model.to(device)


def normalise_network(model, Dem_train, data_stat):
    inj_min = data_stat['inj_min']
    inj_delta = data_stat['inj_delta']
    vrvi_min = data_stat['vrvi_min']
    vrvi_delta = data_stat['vrvi_delta']

    input_stats = (inj_min.reshape(-1).double(), inj_delta.reshape(-1).double())
    output_stats = (vrvi_min.reshape(-1).double(), vrvi_delta.reshape(-1).double())


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
    
 
    preds = []
    targets = []

    for i in range(num_batches):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        Gen_output = network_gen.forward_train(Dem_train[slce])
        Gen_target = Gen_train[slce]
        
        
        mse_criterion = criterion(Gen_output * typ[slce], Gen_target * typ[slce])
        loss_gen = config.crit_weight * mse_criterion                 # supervised learning, difference from label
        

        total_loss = loss_gen 

        total_loss.backward()
        optimizer.step()
        loss_sum += loss_gen
    
    loss_epoch = loss_sum / num_batches

    return mse_criterion, loss_epoch


"""" 
convex relaxation of the OP for power flow.

"""





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