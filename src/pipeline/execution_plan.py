"""What a run is going to do, decided before it starts doing it.

Three things about a run were settled inside the implementation that carries it
out: whether a translation backend would be touched, which language the subtitle
writer should style for, and what the output file is called. Each was re-derived
from `ctx.skip_translation` at the point it was needed, which is how a Japanese-
only run came to wait on a local server's retirement for a resource it was never
going to use - the wait was taken before the branch that would have said so.

A plan states it once, up front:

* which stages this run has, and what each of them needs from the machine;
* whether anything is sent to a translation backend at all;
* how the subtitle output is spelled - language, bilingual, filename.

The resources are the point of the declaration. "Needs the translation backend"
is what decides whether to queue behind a lease; "needs the GPU" is what the ASR
stage claims. A stage that declares nothing takes nothing, which is the whole
reason a skip-translation run should never have queued.

The plan does not execute anything and holds no run state - it is what the run
*is*, so it can be built and asserted on without a video, a model or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.job_context import JobContext
from pipeline.output import resolve_subtitle_bilingual_for_ctx

# What a stage takes from the machine while it runs.
GPU = "gpu"
TRANSLATION_BACKEND = "translation_backend"
TRANSLATION_CACHE = "translation_cache"
DISK = "disk"


@dataclass(frozen=True)
class StagePlan:
    name: str
    resources: frozenset[str]

    def needs(self, resource: str) -> bool:
        return resource in self.resources


@dataclass(frozen=True)
class ExecutionPlan:
    """The shape of one run: its stages, and how its output is spelled."""

    stages: tuple[StagePlan, ...]
    translates: bool
    bilingual: bool
    subtitle_language: str

    # ---------------------------------------------------------------- build

    @classmethod
    def for_asr(cls, ctx: JobContext) -> "ExecutionPlan":
        """The plan as far as it is known before ASR has produced segments.

        `translates` is provisional here in one direction only: a run that will
        not translate is already decided, while a run that intends to can still
        turn out to have nothing to translate. `after_asr` settles it.
        """
        return cls._build(ctx, has_segments=True)

    @classmethod
    def for_run(cls, ctx: JobContext, *, has_segments: bool) -> "ExecutionPlan":
        return cls._build(ctx, has_segments=has_segments)

    @classmethod
    def _build(cls, ctx: JobContext, *, has_segments: bool) -> "ExecutionPlan":
        skip = bool(ctx.skip_translation)
        translates = has_segments and not skip
        # Bilingual is a translated-output option; a Japanese-only run has one
        # language to show, and asking for both would have produced a file whose
        # two lines are the same text.
        bilingual = False if skip else resolve_subtitle_bilingual_for_ctx(ctx)
        stages = [
            StagePlan("asr", frozenset({GPU, DISK})),
            StagePlan("layout", frozenset({DISK})),
        ]
        if translates:
            stages.append(
                StagePlan(
                    "translation",
                    frozenset({TRANSLATION_BACKEND, TRANSLATION_CACHE, DISK}),
                )
            )
            stages.append(
                StagePlan(
                    "repair",
                    frozenset({TRANSLATION_BACKEND, TRANSLATION_CACHE, DISK}),
                )
            )
        stages.append(StagePlan("publish", frozenset({DISK})))
        return cls(
            stages=tuple(stages),
            translates=translates,
            bilingual=bilingual,
            subtitle_language="ja" if skip else "zh",
        )

    def after_asr(self, *, segment_count: int) -> "ExecutionPlan":
        """The same plan, knowing whether ASR found anything.

        Segments only shrink from here - the post-gate filters remove cues and
        never add them - so a run that is empty at this point stays empty.
        """
        if segment_count > 0 or not self.translates:
            return self
        return ExecutionPlan(
            stages=tuple(
                stage for stage in self.stages if stage.name not in {"translation", "repair"}
            ),
            translates=False,
            bilingual=self.bilingual,
            subtitle_language=self.subtitle_language,
        )

    # ----------------------------------------------------------------- read

    def needs(self, resource: str) -> bool:
        """Does any stage of this run take that resource?"""
        return any(stage.needs(resource) for stage in self.stages)

    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    def srt_filename(self, video_stem: str) -> str:
        """`<stem>.ja.srt` for a Japanese-only run, `<stem>.srt` otherwise.

        The suffix is part of the plan because it is decided by the same fact as
        everything else here, and because a run that changed its mind about
        translating halfway would otherwise write the other file's name.
        """
        return (
            f"{video_stem}.ja.srt"
            if self.subtitle_language == "ja"
            else f"{video_stem}.srt"
        )


__all__ = [
    "DISK",
    "GPU",
    "TRANSLATION_BACKEND",
    "TRANSLATION_CACHE",
    "ExecutionPlan",
    "StagePlan",
]
