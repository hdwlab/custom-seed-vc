# Copyright (C) 2025 Human Dataware Lab.
# Created by HDL members
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Shared runtime coordination for API and Socket.IO access."""

import threading
from typing import Optional

from seed_vc.socketio.schemas import ConnectionErrorType


class ServerRuntimeCoordinator:
    """Coordinate shared model access between realtime and offline paths."""

    def __init__(self) -> None:
        """Initialize shared runtime state."""
        self.model_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active_client_sids: set = set()
        self._offline_job_active = False

    def client_count(self) -> int:
        """Return the number of active Socket.IO clients."""
        with self._state_lock:
            return len(self._active_client_sids)

    def has_client(self, sid: str) -> bool:
        """Return whether the client SID is currently registered.

        Args:
            sid: Client session ID.
        """
        with self._state_lock:
            return sid in self._active_client_sids

    def try_register_client(self, sid: str, max_clients: int) -> Optional[ConnectionErrorType]:
        """Try to register a realtime client.

        Args:
            sid: Client session ID.
            max_clients: Maximum number of concurrent clients allowed.

        Returns:
            The blocking error type when registration is rejected, otherwise None.
        """
        with self._state_lock:
            if self._offline_job_active:
                return ConnectionErrorType.OFFLINE_BUSY
            if len(self._active_client_sids) >= max_clients:
                return ConnectionErrorType.MAX_CLIENTS_REACHED
            self._active_client_sids.add(sid)
        return None

    def unregister_client(self, sid: str) -> None:
        """Remove a realtime client registration if present.

        Args:
            sid: Client session ID.
        """
        with self._state_lock:
            self._active_client_sids.discard(sid)

    def try_begin_offline_job(self) -> bool:
        """Try to reserve exclusive access for an offline conversion job.

        Returns:
            True if the reservation succeeded, False otherwise.
        """
        with self._state_lock:
            if self._offline_job_active or self._active_client_sids:
                return False
            self._offline_job_active = True
        return True

    def finish_offline_job(self) -> None:
        """Release the offline conversion reservation."""
        with self._state_lock:
            self._offline_job_active = False

    def is_offline_job_active(self) -> bool:
        """Return whether an offline conversion job is currently active."""
        with self._state_lock:
            return self._offline_job_active
