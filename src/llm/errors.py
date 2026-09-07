"""Shared translation exceptions.

Backends and the orchestration layer must use the same exception hierarchy so
that cancellation and retry decisions survive module boundaries.
"""


class TranslationError(RuntimeError):
    """Base class for translation failures."""


class TranslationCancelledError(TranslationError):
    """Raised when the caller requests cancellation."""


class BackendUnavailableError(TranslationError):
    """Admission timed out; cleanup continues and resource ownership is retained."""


class BackendLeaseInvalidatedError(TranslationError):
    """The backend instance a task was using has been taken out of service.

    Deliberately not retryable: the instance is gone for good (the settings were
    reset, or its last holder closed it), and the only correct answer is to stop
    this task rather than continue on a replacement it never claimed - the two
    halves of one video would come from two different models.

    Lives here rather than in `llm.backends` so the backends themselves can
    raise it: an instance has to refuse *at the moment of use*, since a reset can
    land between a caller's check and its call.
    """


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


# The failures no retry, no narrowing and no tier change can help, in the order
# they tend to appear: the caller asked to stop, the instance this task held is
# gone, the provider judged the text that was sent. Each class already says so
# in its own docstring; naming them together is what lets a multi-stage caller
# ask the question once instead of remembering three answers.
#
# The repair pass is why this exists. Its per-stage `except Exception` caught a
# content refusal, escalated the reasoning tier, reissued the same text, and
# reported `translation_repair_failed` - a documented terminal condition turned
# into a generic one, with a second refusal paid for on the way. Anything listed
# here propagates through every stage unchanged, so the type that reaches the
# caller is the type that was raised.
TERMINAL_TRANSLATION_ERRORS: tuple[type[BaseException], ...] = (
    TranslationCancelledError,
    BackendLeaseInvalidatedError,
    ContentPolicyRefusalError,
)


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
