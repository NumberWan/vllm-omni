# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import threading
from dataclasses import asdict

import zmq

from .messages import InstanceEvent, StageStatus

logger = logging.getLogger(__name__)


class OmniCoordClientForStage:
    """Client used by stage instances to send events to OmniCoordinator.

    This client maintains a DEALER socket connected to OmniCoordinator's
    ROUTER endpoint and sends JSON-encoded events describing instance status.
    """

    def __init__(self, coord_zmq_addr: str, instance_zmq_addr: str, stage_id: int) -> None:
        """Initialize client and send initial registration / status-up event."""
        self._coord_zmq_addr = coord_zmq_addr
        self._instance_zmq_addr = instance_zmq_addr
        self._stage_id = stage_id

        # Dedicated ZMQ context for this stage client.
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        try:
            self._socket.connect(self._coord_zmq_addr)
        except zmq.ZMQError as e:
            self._socket.close()
            raise RuntimeError(f"Failed to connect to coordinator at {self._coord_zmq_addr}: {e}") from e

        self._status = StageStatus.UP
        self._queue_length = 0
        self._closed = False
        self._heartbeat_interval = 5.0
        self._stop_event = threading.Event()
        self._send_lock = threading.Lock()

        self._send_event("update")

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _send_event(self, event_type: str) -> None:
        """Send an InstanceEvent to OmniCoordinator.

        Wire format: zmq_addr, stage_id, status, queue_length, event_type.
        For "update": includes status and queue_length from instance state.
        For "heartbeat": status and queue_length are null.
        """
        if self._closed:
            raise RuntimeError("Client already closed")

        event = InstanceEvent(
            zmq_addr=self._instance_zmq_addr,
            stage_id=self._stage_id,
            event_type=event_type,
            status=self._status.value,
            queue_length=self._queue_length,
        )
        data = json.dumps(asdict(event)).encode("utf-8")
        with self._send_lock:
            try:
                self._socket.send(data, flags=zmq.NOBLOCK)
            except zmq.Again:
                logger.debug("Send buffer full, dropping event")

    def update_info(
        self,
        status: StageStatus | None = None,
        queue_length: int | None = None,
    ) -> None:
        """Update instance information and notify OmniCoordinator.

        At least one of ``status`` or ``queue_length`` must be provided.
        """
        if status is None and queue_length is None:
            raise ValueError("At least one of status or queue_length must be provided")

        if status is not None:
            self._status = status
        if queue_length is not None:
            self._queue_length = queue_length

        self._send_event("update")

    def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat events while the client is alive."""
        while not self._stop_event.wait(timeout=self._heartbeat_interval):
            if self._closed:
                break

            try:
                self._send_event("heartbeat")
            except (RuntimeError, zmq.ZMQError):
                break

    def close(self) -> None:
        """Send a final down event and close the underlying socket."""
        if self._closed:
            raise RuntimeError("Client already closed")

        # Stop heartbeat thread first to avoid concurrent sends during shutdown.
        self._stop_event.set()
        if hasattr(self, "_heartbeat_thread"):
            self._heartbeat_thread.join(timeout=1.0)

        # Mark status as DOWN and send one last update.
        self._status = StageStatus.DOWN
        try:
            self._send_event("update")
        except zmq.ZMQError:
            pass  # Socket may already be broken, proceed with close

        # Close DEALER socket and terminate this client's context.
        self._socket.close(0)
        try:
            self._ctx.term()
        except zmq.ZMQError:
            pass
        self._closed = True
