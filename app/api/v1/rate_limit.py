from collections import defaultdict, deque
from threading import Lock
from time import monotonic


class QueryLimiter:
    """Small per-process guard for a single-instance portfolio demo."""

    def __init__(self, limit: int = 6, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_ip: str) -> bool:
        now = monotonic()
        with self._lock:
            hits = self._hits[client_ip]
            while hits and now - hits[0] >= self.window_seconds:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


query_limiter = QueryLimiter()
