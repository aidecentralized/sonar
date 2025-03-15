import numpy as np
from typing import Any, Dict, List, Union


def from_round_stats_per_round_per_client_to_dict_arrays(
    stats: List[List[Dict[str, Union[np.ndarray[Any, Any], float]]]]
) -> Dict[str, np.ndarray[Any, Any]]:
    """
    Convert round statistics per client to a dictionary of arrays.

    Args:
        stats (List[List[Dict[str, Union[np.ndarray, float]]]]): List of rounds stats,
        where each round contains stats for each client.

    Returns:
        Dict[str, np.ndarray[Any, Any]]: Dictionary of stats name to an array of stats per client per round.
    """
    num_rounds = len(stats)
    num_clients = len(stats[0])
    stats_dict: Dict[str, np.ndarray[Any, Any]] = {}

    for round_idx, rounds_stats in enumerate(stats):
        for client_idx, client_stats in enumerate(rounds_stats):
            for key, value in client_stats.items():
                if key not in stats_dict:
                    stats_dict[key] = (
                        np.zeros((num_clients, num_rounds, len(value)))
                        if isinstance(value, np.ndarray)
                        else np.zeros((num_clients, num_rounds))
                    )
                stats_dict[key][client_idx][round_idx] = value

    return stats_dict


def from_rounds_stats_per_client_per_round_to_dict_arrays(
    stats: List[List[Dict[str, Union[np.ndarray[Any, Any], float]]]]
) -> Dict[str, np.ndarray[Any, Any]]:
    """
    Convert list of client-specific round stats to a dictionary of arrays.

    Args:
        stats (List[List[Dict[str, Union[np.ndarray, float]]]]): List of client-specific round stats,
        where each client has a list of round stats.

    Returns:
        Dict[str, np.ndarray]: Dictionary of stats name to an array of stats per client per round.
    """
    round_stats_per_round_per_client: List[
        List[Dict[str, Union[np.ndarray[Any, Any], float]]]
    ] = []
    for client_stats in stats:
        for round_idx, round_stats in enumerate(client_stats):
            if len(round_stats_per_round_per_client) <= round_idx:
                round_stats_per_round_per_client.append([])
            round_stats_per_round_per_client[round_idx].append(round_stats)

    return from_round_stats_per_round_per_client_to_dict_arrays(
        round_stats_per_round_per_client
    )
