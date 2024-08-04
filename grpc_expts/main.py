import argparse
from scheduler import Scheduler
import gc
import torch
import copy
import logging
logging.getLogger('PIL').setLevel(logging.INFO)

b_default = "./configs/non_iid_clients.py"
parser = argparse.ArgumentParser(description='Run collaborative learning experiments')
parser.add_argument('-b', nargs='?', default=b_default, type=str,
                    help='filepath for benchmark config, default: {}'.format(b_default))
args = parser.parse_args()

scheduler = Scheduler()
scheduler.assign_config_by_path(args.b)
print("Config loaded")

# Run the experiment multiple times with different seeds
num_rep = scheduler.config.get("num_rep", 1)

# Copy source code only for the first run
should_copy_source_code = True
for i in range(0, num_rep):
    # Potential parameter to test
    test_key = scheduler.config.get("test_param", None)
    values = [""]
    if test_key is not None:
        values = scheduler.config["test_values"]
        
    for v in values:
        s = copy.deepcopy(scheduler)
        s.config["exp_id"] += str(v) 
        if num_rep > 1:
            # seed is different for each run across all clients
            s.config["seed"] = i * s.config["num_clients"] 
        if test_key is not None:
            s.config[test_key] = v
        s.install_config()
        s.initialize(should_copy_source_code)
        should_copy_source_code = False
        s.run_job()
        
        s = None
        gc.collect()
        torch.cuda.empty_cache()
