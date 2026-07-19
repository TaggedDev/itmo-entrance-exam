from contextlib import contextmanager
from typing import Iterator

from ml_service.config import Settings


class NoopObservation:
    trace_id: str | None = None

    def update(self, **_kwargs: object) -> None:
        return None


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
    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: object | None = None,
        output: object | None = None,
        metadata: object | None = None,
        model: str | None = None,
        model_parameters: dict[str, object] | None = None,
    ) -> Iterator[object]:
        if not self.enabled or self.client is None:
            yield NoopObservation()
            return

        manager = None
        observation = NoopObservation()
        try:
            manager = self.client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=input,
                output=output,
                metadata=metadata,
                model=model,
                model_parameters=model_parameters,
            )
            observation = manager.__enter__()
        except Exception:
            yield observation
            return

        try:
            yield observation
        except Exception as exc:
            try:
                observation.update(level="ERROR", status_message=str(exc))
            except Exception:
                pass
            raise
        finally:
            try:
                if manager is not None:
                    manager.__exit__(None, None, None)
            except Exception:
                pass

    @contextmanager
    def span(self, name: str, **metadata: object) -> Iterator[object]:
        with self.observation(name, as_type="span", metadata=metadata) as observation:
            yield observation

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                pass
