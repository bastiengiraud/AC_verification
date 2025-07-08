from types import SimpleNamespace
from nn_training_ac_crown import train


def create_config():
    parameters_dict = {
        'test_system': 118,
        'hidden_layer_size': 50,
        'n_hidden_layers': 3,
        'epochs': 1000,
        'batch_size': 50,
        'learning_rate': 1e-3,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 3,
        'pg_viol_weight': 0,
        'vm_viol_weight': 0,
        'crit_weight': 1e4,
        'PF_weight': 1e-4,
        'LPF_weight': 1e-4,
        'N_enrich': 50,
        'Algo': True, # if True, add worst-case violation CROWN bounds during training
        'Enrich': True,
        'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN
    }
    config = SimpleNamespace(**parameters_dict)
    return config


def main():
    config = create_config()
    train(config=config)


if __name__ == '__main__':
    main()
