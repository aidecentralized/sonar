import asyncio
from algos.fl_rtc import FedRTCNode

async def test_fedrtc_node():
    # Sample configuration
    config = {
        "rounds": 3,  # Number of FL rounds
        "epochs_per_round": 1,
        "num_collaborators": 3,  # Total expected nodes
        # "topology": "ring",  # Example topology
        # "node_id": 1,  # Unique node identifier
        # Collaboration setup
        "algo": "fedrtc",
        "topology": {"name": "ring"}, # type: ignore
        # "rounds": 2,
        # "num_users": 3,
        # "exp_id": "test",
        # "dset": CIFAR10_DSET,
        # "dump_dir": DUMP_DIR,
        # "dpath": CIAR10_DPATH,
        # "seed": 2,
        # "device_ids": get_device_ids(num_users, gpu_ids),
        # # "algos": get_algo_configs(num_users=num_users, algo_configs=default_config_list),  # type: ignore
        # "algos": get_algo_configs(num_users=num_users, algo_configs=[fedstatic]),  # type: ignore
        # "samples_per_user": 50000 // num_users,  # distributed equally
        # "train_label_distribution": "non_iid",
        # "test_label_distribution": "iid",
        # "alpha_data": 1.0,
        # "exp_keys": [],
        # "dropout_dicts": dropout_dicts,
        # "test_samples_per_user": 200,
        

        # Model parameters
        # "model": "resnet10",
        # "model_lr": 3e-4,
        # "batch_size": 256,
    }

    # Create FedRTC Node instance
    node = FedRTCNode(config)

    # Run the protocol
    await node.run_async_protocol()

def main():
    asyncio.run(test_fedrtc_node())

if __name__ == "__main__":
    main()