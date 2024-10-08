from typing import TypeAlias, Dict, List, Union, Tuple, Optional

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
        Tuple[Union[int, str, float, bool, None], ...],
        Optional[List[int]],
    ],
]