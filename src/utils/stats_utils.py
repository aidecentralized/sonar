import numpy as np

def from_round_stats_per_round_per_client_to_dict_arrays(stats):
    # Stats is list of rounds stats
    # Rounds stats is list of clients stats
    # Client stats is dict of stats
    num_rounds = len(stats)
    num_clients = len(stats[0])
    stats_dict = {}
    
    # Build a dict of stats name to an array of stats per client per round
    for round, rounds_stats in enumerate(stats):
        for client_idx, client_stats in enumerate(rounds_stats):
            for key, value in client_stats.items():
                if key not in stats_dict:
                    stats_dict[key] = np.zeros((num_clients, num_rounds, len(value))) if isinstance(value, np.ndarray) else np.zeros((num_clients, num_rounds))
                stats_dict[key][client_idx][round] = value
    
    return stats_dict

def from_rounds_stats_per_client_per_round_to_dict_arrays(stats):
    # Stats is list of client stats
    # Client stats is list of client specific round stats
    # Round stats is dict of stats

    # Convert to list of round stats per round per client
    round_stats_per_round_per_client = []
    for client_stats in stats:
        for round, round_stats in enumerate(client_stats):
            if len(round_stats_per_round_per_client) <= round:
                round_stats_per_round_per_client.append([])
            round_stats_per_round_per_client[round].append(round_stats)
            
    return from_round_stats_per_round_per_client_to_dict_arrays(round_stats_per_round_per_client)
            
            
