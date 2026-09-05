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


class ContentPolicyRefusalError(TranslationError):
    """The provider's content filter rejected the request or its reply.

    Deliberately NOT a `RetryableTranslationError`. The filter judges the text
    that was sent, so reissuing it - or reissuing a smaller slice of it, which
    is what the span-narrowing retry does - asks the same question and gets the
    same answer. On 2026-09-04 one batch spent 21 minutes on four attempts and
    two narrowings before failing on the same `cyber_policy` code it started
    with, having burned the reasoning tokens of all four.

    Failing fast is cheap here: batches that already came back are written to
    the translation cache as they land, so retrying the job resumes from them.
    """


class MaxTokensRejectedError(TranslationError):
    """The endpoint refused the request because `max_tokens` was out of range.

    Not about the reply at all - the request never ran, so nothing was
    generated and nothing was billed. Deliberately NOT a
    `RetryableTranslationError`: reissuing the same number gets the same
    refusal. The caller retries with a smaller one, which is why the number
    that was refused travels with the error, along with the ceiling the
    endpoint named if it named one.

    `learnable` is False when the refusal is about this request rather than
    about the endpoint - an `input + max_output_tokens <= N` limit is real, and
    a smaller budget does satisfy it, but N moves with the prompt. Kept out of
    the capability cache, where it would otherwise clamp every shorter batch of
    the film to what the longest one could not have.
    """

    def __init__(
        self,
        message: str,
        *,
        sent: int,
        limit: int | None = None,
        learnable: bool = True,
    ) -> None:
        super().__init__(message)
        self.sent = int(sent)
        self.limit = int(limit) if limit is not None else None
        self.learnable = bool(learnable)


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
