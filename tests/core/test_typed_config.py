"""One set of rules for turning a configured string into a value.

Every module used to parse its own settings and they disagreed: some clamped,
some took the string as written, some raised at import time. A negative linger,
a NaN reading speed and a mistyped batch size each produced a different kind of
wrong - silently accepted, silently propagated, or a traceback from three layers
below whatever the user was doing.

These tests pin the three answers that are now the same everywhere: unset means
default, unusable means default *and* a recorded problem, and out of range means
either the bound or the default depending on what the bound means for that
field.
"""

from __future__ import annotations

import pytest

from core import typed_config


@pytest.fixture(autouse=True)
def _clean_registry():
    typed_config.clear_configuration_problems()
    yield
    typed_config.clear_configuration_problems()


class TestWhatIsNotAnError:
    def test_an_unset_field_is_simply_the_default(self, monkeypatch):
        monkeypatch.delenv("A_NUMBER", raising=False)

        assert typed_config.env_float("A_NUMBER", 2.5) == 2.5
        assert typed_config.configuration_problems() == []

    def test_an_empty_field_is_the_default_too(self, monkeypatch):
        # `.env` and the settings page both write `KEY=` to mean "not set", and
        # that is most of the file.
        monkeypatch.setenv("A_NUMBER", "")
        monkeypatch.setenv("A_COUNT", "   ")

        assert typed_config.env_float("A_NUMBER", 2.5) == 2.5
        assert typed_config.env_int("A_COUNT", 7) == 7
        assert typed_config.configuration_problems() == []

    def test_an_integer_written_as_a_decimal_is_accepted(self, monkeypatch):
        monkeypatch.setenv("A_COUNT", "4.0")

        assert typed_config.env_int("A_COUNT", 7) == 4
        assert typed_config.configuration_problems() == []


class TestWhatIsRecorded:
    @pytest.mark.parametrize("reader,default", [
        (typed_config.env_int, 7), (typed_config.env_float, 2.5),
        (typed_config.env_bool, True),
    ])
    def test_clearing_a_value_clears_its_old_warning(self, monkeypatch, reader, default):
        monkeypatch.setenv("A_SETTING", "invalid")
        reader("A_SETTING", default)
        assert typed_config.configuration_problems()
        monkeypatch.setenv("A_SETTING", "  ")
        assert reader("A_SETTING", default) == default
        assert typed_config.configuration_problems() == []

    def test_fractional_counts_are_not_silently_truncated(self, monkeypatch):
        monkeypatch.setenv("A_COUNT", "4.5")
        assert typed_config.env_int("A_COUNT", 7) == 7
        assert typed_config.configuration_problems()

    def test_a_value_that_is_not_a_number_falls_back_and_says_so(self, monkeypatch):
        monkeypatch.setenv("A_NUMBER", "0.6s")

        assert typed_config.env_float("A_NUMBER", 2.5) == 2.5
        assert typed_config.configuration_problems() == [
            "A_NUMBER='0.6s'：不是数字，已按默认值 2.5 处理"
        ]

    def test_a_non_finite_value_is_refused(self, monkeypatch):
        # `float("nan")` parses. It then propagates through every duration
        # derived from it, comparing false against everything, without raising.
        monkeypatch.setenv("A_NUMBER", "nan")
        monkeypatch.setenv("A_COUNT", "inf")

        assert typed_config.env_float("A_NUMBER", 2.5) == 2.5
        assert typed_config.env_int("A_COUNT", 7) == 7
        assert len(typed_config.configuration_problems()) == 2

    def test_an_unknown_switch_word_is_not_read_as_false(self, monkeypatch):
        monkeypatch.setenv("A_SWITCH", "maybe")

        assert typed_config.env_bool("A_SWITCH", True) is True
        assert typed_config.configuration_problems()[0].startswith("A_SWITCH=")

    def test_a_switch_accepts_the_words_people_write(self, monkeypatch):
        for word in ("1", "true", "YES", "on"):
            monkeypatch.setenv("A_SWITCH", word)
            assert typed_config.env_bool("A_SWITCH", False) is True
        for word in ("0", "false", "No", "off"):
            monkeypatch.setenv("A_SWITCH", word)
            assert typed_config.env_bool("A_SWITCH", True) is False
        assert typed_config.configuration_problems() == []

    def test_a_value_outside_a_named_set_is_recorded(self, monkeypatch):
        monkeypatch.setenv("A_MODE", "reading-ish")

        value = typed_config.env_choice(
            "A_MODE", "alignment", choices=("alignment", "reading")
        )

        assert value == "alignment"
        assert typed_config.configuration_problems()[0].startswith("A_MODE=")

    def test_the_problems_are_listed_together_field_by_field(self, monkeypatch):
        # The point of collecting them: four mistakes in `.env` produce four
        # lines once, not one crash per run with the first field named.
        monkeypatch.setenv("A_NUMBER", "x")
        monkeypatch.setenv("A_COUNT", "y")
        monkeypatch.setenv("A_SWITCH", "z")

        typed_config.env_float("A_NUMBER", 1.0)
        typed_config.env_int("A_COUNT", 1)
        typed_config.env_bool("A_SWITCH", True)

        problems = typed_config.configuration_problems()
        assert [item.split("=")[0] for item in problems] == [
            "A_COUNT",
            "A_NUMBER",
            "A_SWITCH",
        ]

    def test_a_field_stops_complaining_once_it_is_fixed(self, monkeypatch):
        # These values are re-read at call time because the settings page edits
        # them while the process runs, so correcting one has to be enough.
        monkeypatch.setenv("A_NUMBER", "x")
        typed_config.env_float("A_NUMBER", 1.0)
        assert typed_config.configuration_problems()

        monkeypatch.setenv("A_NUMBER", "3.0")
        assert typed_config.env_float("A_NUMBER", 1.0) == 3.0
        assert typed_config.configuration_problems() == []


class TestWhatOutOfRangeMeans:
    def test_a_cap_is_clamped_to_the_nearest_legal_value(self, monkeypatch):
        monkeypatch.setenv("A_COUNT", "5000")

        assert typed_config.env_int("A_COUNT", 200, minimum=8, maximum=400) == 400
        assert "已按该边界处理" in typed_config.configuration_problems()[0]

    def test_a_rate_falls_back_instead_of_clamping(self, monkeypatch):
        """Clamping is the wrong reading when the bound is where the quantity
        stops meaning anything: a token rate clamped to its floor gives every
        chunk a budget of nearly zero, which truncates the whole film. The
        configured default is the safe answer to a typo."""
        monkeypatch.setenv("A_RATE", "-5")

        value = typed_config.env_float(
            "A_RATE", 10.0, minimum=0.001, out_of_range=typed_config.FALL_BACK
        )

        assert value == 10.0
        assert "已按默认值" in typed_config.configuration_problems()[0]

    def test_a_value_inside_the_range_is_untouched(self, monkeypatch):
        monkeypatch.setenv("A_COUNT", "200")

        assert typed_config.env_int("A_COUNT", 100, minimum=8, maximum=400) == 200
        assert typed_config.configuration_problems() == []


class TestTheFieldsThatUseIt:
    def test_invalid_reading_rate_uses_default_not_the_slowest_rate(self, monkeypatch):
        from subtitles.options import SubtitleOptions

        monkeypatch.setenv("SUBTITLE_READING_CPS", "-1")
        assert SubtitleOptions.from_env().reading_cps == 7.0
    def test_subtitle_options_no_longer_raise_on_a_typo(self, monkeypatch):
        """A mistyped duration used to end the run with a bare ValueError from
        whichever field happened to parse first."""
        from subtitles.options import SubtitleOptions

        monkeypatch.setenv("SUBTITLE_MAX_SOURCE_CHARS", "twenty")
        monkeypatch.setenv("SUBTITLE_LINGER_S", "-1")
        monkeypatch.setenv("SUBTITLE_READING_CPS", "nan")

        options = SubtitleOptions.from_env()

        assert options.max_source_chars == 32
        assert options.linger_s == 0.0  # a duration, clamped at its floor
        assert options.reading_cps == 7.0
        assert len(typed_config.configuration_problems()) == 3

    def test_a_version_stamp_is_still_refused_outright(self, monkeypatch):
        """The one field that must not be corrected into something usable: it
        names the output, so accepting an unknown value would stamp cues with a
        layout that did not make them."""
        from subtitles.options import SubtitleOptions

        monkeypatch.setenv("SUBTITLE_LAYOUT_ENGINE", "anchor_aware_dp_v2")

        with pytest.raises(ValueError, match="SUBTITLE_LAYOUT_ENGINE"):
            SubtitleOptions.from_env()
