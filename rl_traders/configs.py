EIIE_CONFIG = {
    'lower_params': [{"name": "conv2d", "kernel_size": (3, 1),
                     "in_channels": 4, "out_channels": 2, "stride": 1},
                     {"name": "conv2d", "kernel_size": (48, 1),
                      "in_channels": 2, "out_channels": 20, "stride": 1},],
    'upper_params': [{"name": "conv2d", "kernel_size": (1, 1),
                     "in_channels": 21, "out_channels": 1, "stride": 1},
                     {"name": "flatten"}],
}


LR_SPEC = {"lr": 3.0e-3, 'name': 'adam'}
SCHEDULER_SPEC = {'name': 'plateau', 'patience': 1000, 'factor': 0.9, 'min_lr': 3.0e-5}