from contextlib import contextmanager
from typing import Iterator

from ml_service.config import Settings


class LangfuseTracer:
    def __init__(self, settings: Settings) -> None:
        self.enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self.client = None
        if not self.enabled:
            return
        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            self.enabled = False
            self.client = None

    @contextmanager
    def span(self, name: str, **metadata: object) -> Iterator[None]:
        if not self.enabled or self.client is None:
            yield
            return
        try:
            with self.client.start_as_current_observation(name=name, metadata=metadata):
                yield
        except Exception:
            yield

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                pass
