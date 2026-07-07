"""Typed ZeroMQ queue wrappers used by Mini-SGLang processes."""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    """
    Synchronous ZMQ PUSH queue with msgpack serialization.
    """

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """
        Create or connect a PUSH socket.
        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        """
        Serialize and send one object.
        """

        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """
        Close the socket and terminate the owning context.
        """

        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    """
    Async ZMQ PUSH queue with msgpack serialization.
    """

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """
        Create or connect an async PUSH socket.
        """

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        """
        Serialize and asynchronously send one object.
        """

        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        """
        Close the socket and terminate the owning context.
        """

        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    """
    Synchronous ZMQ PULL queue with msgpack deserialization.
    """

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """
        Create or connect a PULL socket.
        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        """
        Receive and decode one object.
        """

        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        """
        Receive one raw serialized payload.
        """

        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        """
        Decode a raw payload received by ``get_raw``.
        """

        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        """
        Return whether the socket has no immediately available message.
        """

        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """
        Close the socket and terminate the owning context.
        """

        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    """
    Async ZMQ PULL queue with msgpack deserialization.
    """

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """
        Create or connect an async PULL socket.
        """

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        """
        Receive and decode one object asynchronously.
        """

        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        """
        Close the socket and terminate the owning context.
        """

        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """
    Synchronous ZMQ PUB queue used to fan out rank-0 scheduler messages.
    """

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """
        Create or connect a PUB socket.
        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        """
        Send an already-serialized payload.
        """

        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        """
        Serialize and publish one object.
        """

        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """
        Close the socket and terminate the owning context.
        """

        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    """
    Synchronous ZMQ SUB queue subscribed to all messages.
    """

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """
        Create or connect a SUB socket.
        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        """
        Receive and decode one published object.
        """

        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        """
        Return whether the socket has no immediately available message.
        """

        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """
        Close the socket and terminate the owning context.
        """

        self.socket.close()
        self.context.term()
