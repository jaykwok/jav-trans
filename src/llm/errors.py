"""Shared translation exceptions.

Backends and the orchestration layer must use the same exception hierarchy so
that cancellation and retry decisions survive module boundaries.
"""


class TranslationError(RuntimeError):
    """Base class for translation failures."""


class TranslationCancelledError(TranslationError):
    """Raised when the caller requests cancellation."""


class RetryableTranslationError(TranslationError):
    """A transient transport or response-shape failure that may be retried."""


class RetryableTranslationFormatError(RetryableTranslationError):
    """The model returned incomplete or invalid structured output."""


class ResponseTruncatedError(TranslationError):
    """The model stopped because it hit this request's output-token budget.

    Deliberately NOT a `RetryableTranslationError`: the generic retry path
    reissues the identical request, which truncates identically and only burns
    budget. Reissuing with a *larger* budget is a different request, so the
    limit that bound travels with the error - both to size that retry and so
    the message can name the number that actually stopped the reply.
    """

    def __init__(self, message: str, *, limit: int) -> None:
        super().__init__(message)
        self.limit = int(limit)
