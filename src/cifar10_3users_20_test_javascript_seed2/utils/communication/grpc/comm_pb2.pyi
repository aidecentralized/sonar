from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Model(_message.Message):
    __slots__ = ("buffer",)
    BUFFER_FIELD_NUMBER: _ClassVar[int]
    buffer: bytes
    def __init__(self, buffer: _Optional[bytes] = ...) -> None: ...

class Data(_message.Message):
    __slots__ = ("id", "model")
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    id: str
    model: Model
    def __init__(self, id: _Optional[str] = ..., model: _Optional[_Union[Model, _Mapping]] = ...) -> None: ...

class Rank(_message.Message):
    __slots__ = ("rank",)
    RANK_FIELD_NUMBER: _ClassVar[int]
    rank: int
    def __init__(self, rank: _Optional[int] = ...) -> None: ...

class Round(_message.Message):
    __slots__ = ("round",)
    ROUND_FIELD_NUMBER: _ClassVar[int]
    round: int
    def __init__(self, round: _Optional[int] = ...) -> None: ...

class Port(_message.Message):
    __slots__ = ("port",)
    PORT_FIELD_NUMBER: _ClassVar[int]
    port: int
    def __init__(self, port: _Optional[int] = ...) -> None: ...

class PeerId(_message.Message):
    __slots__ = ("rank", "port", "ip")
    RANK_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    IP_FIELD_NUMBER: _ClassVar[int]
    rank: Rank
    port: Port
    ip: str
    def __init__(self, rank: _Optional[_Union[Rank, _Mapping]] = ..., port: _Optional[_Union[Port, _Mapping]] = ..., ip: _Optional[str] = ...) -> None: ...

class PeerIds(_message.Message):
    __slots__ = ("peer_ids",)
    class PeerIdsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: PeerId
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[PeerId, _Mapping]] = ...) -> None: ...
    PEER_IDS_FIELD_NUMBER: _ClassVar[int]
    peer_ids: _containers.MessageMap[int, PeerId]
    def __init__(self, peer_ids: _Optional[_Mapping[int, PeerId]] = ...) -> None: ...

class Quorum(_message.Message):
    __slots__ = ("quorum",)
    QUORUM_FIELD_NUMBER: _ClassVar[int]
    quorum: bool
    def __init__(self, quorum: bool = ...) -> None: ...
