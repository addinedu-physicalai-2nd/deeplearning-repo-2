"""상대 서버별 연결 상태 전환과 송수신 실패만 기록한다."""


class ConnectionEventLogger:
    """반복되는 정상 통신 로그와 동일 장애 로그를 억제한다."""

    def __init__(self, peer_name, state, state_key, logger):
        self.peer_name = peer_name
        self.state = state
        self.state_key = state_key
        self.logger = logger
        self.connected = False
        self.endpoint = None
        self._send_failure_active = False
        self._receive_failure_active = False
        self.state.update(**{self.state_key: False})

    def connection_succeeded(self, endpoint=None):
        """미연결 상태에서 정상 통신이 확인될 때 한 번만 기록한다."""
        endpoint_changed = (
            self.connected
            and endpoint is not None
            and endpoint != self.endpoint
        )
        if endpoint_changed:
            self.logger.warning(
                "[%s] connection lost: endpoint changed from %s",
                self.peer_name,
                self.endpoint,
            )
            self.connected = False

        if not self.connected:
            if endpoint is None:
                self.logger.info("[%s] connection established", self.peer_name)
            else:
                self.logger.info(
                    "[%s] connection established: %s",
                    self.peer_name,
                    endpoint,
                )

        self.connected = True
        if endpoint is not None:
            self.endpoint = endpoint
        self._send_failure_active = False
        self._receive_failure_active = False
        self.state.update(**{self.state_key: True})

    def connection_lost(self, reason):
        """연결되어 있던 상대가 끊어졌을 때 한 번만 기록한다."""
        if self.connected:
            self.logger.warning(
                "[%s] connection lost: %s",
                self.peer_name,
                reason,
            )
        self.connected = False
        self._send_failure_active = True
        self._receive_failure_active = True
        self.state.update(**{self.state_key: False})

    def send_failed(self, error):
        """연속된 동일 장애 구간에는 송신 실패를 한 번만 기록한다."""
        if not self._send_failure_active:
            self.logger.error(
                "[%s] data send failed: %s",
                self.peer_name,
                error,
            )
        self._send_failure_active = True

    def receive_failed(self, error):
        """연속된 동일 장애 구간에는 수신 실패를 한 번만 기록한다."""
        if not self._receive_failure_active:
            self.logger.error(
                "[%s] data receive failed: %s",
                self.peer_name,
                error,
            )
        self._receive_failure_active = True
