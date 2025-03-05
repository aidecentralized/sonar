from typing import TypeAlias, Dict, List, Union, Tuple, Optional

import torch

# FIXME: We need to somehow create a template for the ConfigType
# and that should be used for typechecking everywhere
# this approach is not scalable and won't catch errors
ConfigType: TypeAlias = Dict[
    str,
    Union[
        str,
        float,
        int,
        bool,
        List[str],
        List[int],
        List[float],
        List[bool],
        Dict[str, List[int]],
        Dict[str, str],
        Dict[str, List[str] | str],
        Dict[str, Dict[str, float]],
        Tuple[Union[int, str, float, bool, None], ...],
        Optional[List[int]],
    ],
]

TorchModelType: TypeAlias = Dict[str, torch.Tensor]