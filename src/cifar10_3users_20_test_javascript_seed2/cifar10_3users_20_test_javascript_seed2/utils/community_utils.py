import numpy as np
from typing import Dict, List


def get_random_communities(
    num_clients: int, num_communities: int
) -> Dict[int, List[int]]:
    assert num_clients % num_communities == 0

    clients_per_community = num_clients // num_communities
    indices = np.random.permutation(range(1, num_clients + 1))

    support: Dict[int, List[int]] = {}
    for i, c_id in enumerate(indices):
        idx = (i // clients_per_community) * clients_per_community
        support[c_id] = list(indices[idx : idx + clients_per_community])
    return support


def get_dset_balanced_communities(
    num_clients: int, num_communities: int, num_dset: int
) -> Dict[int, List[int]]:
    # Assume same dset are consecutive
    assert num_clients % num_communities == 0
    assert num_clients % num_dset == 0

    clients_per_dset = num_clients // num_dset
    clients_id_per_dset: List[List[int]] = []
    for dset in range(num_dset):
        clients_id_per_dset.append(
            list(
                np.random.permutation(
                    np.arange(
                        1 + dset * clients_per_dset, 1 + (dset + 1) * clients_per_dset
                    )
                )
            )
        )

    num_assigned = 0
    communities: Dict[int, List[int]] = {k: [] for k in range(num_communities)}
    while num_assigned < num_clients:
        communities_random_order = {
            k: v for k, v in enumerate(np.random.permutation(range(num_communities)))
        }
        communities_id = sorted(
            range(num_communities),
            key=lambda k: (len(communities[k]), communities_random_order[k]),
        )

        current_dset_clients = clients_id_per_dset.pop()
        while len(current_dset_clients) > 0:
            for community_id in communities_id:
                if len(current_dset_clients) == 0:
                    break
                communities[community_id].append(current_dset_clients.pop())
                num_assigned += 1

    support = {idx: idxs for idxs in communities.values() for idx in idxs}
    return support


def get_dset_communities(num_clients: int, num_dset: int) -> Dict[int, List[int]]:
    # Assume same dset are consecutive
    assert num_clients % num_dset == 0
    clients_per_dset = num_clients // num_dset
    support: Dict[int, List[int]] = {}
    for idx in range(num_clients):
        dset = idx // clients_per_dset
        support[idx + 1] = list(
            np.arange(1 + dset * clients_per_dset, 1 + (dset + 1) * clients_per_dset)
        )
    return support
