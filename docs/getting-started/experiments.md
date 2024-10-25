# Automating Experiments

In this tutorial, we will discuss how to automate running multiple experiments by customizing our experiment script. Note that we currently only support automation on one machine with the gRPC protocol. If you have not already read the [Getting Started](./getting-started.md) guide, we recommend you do so before proceeding.

## Running the Code
The `main_exp.py` file automates running experiments on one machine using gRPC. You can run this file with the command:
``` bash
python main_exp.py -host randomhost42.mit.edu
```

## Customizing the Experiments
To customize your experiment automation, make these changes in `main_exp.py`.

1. Specify your constant settings in `sys_config.py` and `algo_config.py`
2. Import the sys_config and algo_config setting objects you want to use for your experiments. 
``` python
from configs.algo_config import traditional_fl
from configs.sys_config import grpc_system_config
```

3. Write the experiment object like the example `exp_dict`, mapping each new experiment ID to the set of keys that you want to change per experiment. Specify the `algo_config` and its specific customizations in `algo`, and likewise for `sys_config` and `sys`. *Note every experiment must have a unique experiment path, and we recommend guarenteeing this by giving every experiment a unique experiment id.*
``` python
exp_dict = {
    "test_automation_1": {
        "algo_config": traditional_fl,
        "sys_config": grpc_system_config,
        "algo": {
            "num_users": 3,
            "num_rounds": 3,
        },
        "sys": {
            "seed": 3,
        },
    },
    "test_automation_2": {
        "algo_config": traditional_fl,
        "sys_config": grpc_system_config,
        "algo": {
            "num_users": 4,
            "num_rounds": 4,
        },
        "sys": {
            "seed": 4,
        },
    },
}
```


4. (Optional) Specify whether or not to run post hoc metrics and plots by setting the boolean at the top of the file.
``` bash
post_hoc_plot: bool = True
```

5. Start the experiments with the command. 
``` bash
python main_exp.py -host randomhost42.mit.edu
```