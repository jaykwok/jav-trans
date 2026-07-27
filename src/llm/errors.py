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


class TranslationContextLengthError(TranslationError):
    """The request cannot fit within the selected backend context window."""
