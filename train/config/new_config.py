args_resnet = {
    'epochs': 185,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.001,
        #'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        #'T_max': 200
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-6,
    },
    'batch_size': 32,
}
args_densenet = {
    'epochs': 185,
    'optimizer_name': 'AdamW',#'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.001,
        #'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-6,
    },
    'batch_size': 32,
}

args_wideresnet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 32,
}