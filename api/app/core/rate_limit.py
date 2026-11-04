import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Janela deslizante de 60 s por cliente, em memória.

    Limite: estado por processo e identificado pelo IP da conexão (não lê X-Forwarded-For, que o cliente forja).
    Atrás de proxy ou com vários workers, use o limite do proxy ou um store compartilhado (Redis).
    """

    def __init__(self, per_minute: int, window: float = 60.0):
        self.per_minute, self.window = per_minute, window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[client]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True
