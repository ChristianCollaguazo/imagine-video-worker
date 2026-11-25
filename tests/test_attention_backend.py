import pytest

from app.attention_backend import select_attention


def test_auto_enables_sage_only_after_sm120_probe():
    assert select_attention("auto", (12, 0), (True, "ready")) == (
        "sage", "sm120_probe_passed"
    )


@pytest.mark.parametrize("capability", [None, (8, 9), (10, 0), (12, 1)])
def test_auto_falls_back_outside_sm120(capability):
    assert select_attention("auto", capability, (True, "ready"))[0] == "pytorch_sdpa"


def test_auto_falls_back_when_sage_probe_fails():
    backend, reason = select_attention("auto", (12, 0), (False, "ImportError"))
    assert backend == "pytorch_sdpa"
    assert reason == "sage_probe_failed:ImportError"


def test_off_never_probes_sage():
    assert select_attention("off", (12, 0), (False, "not_probed")) == (
        "pytorch_sdpa", "disabled_by_configuration"
    )


@pytest.mark.parametrize(
    ("capability", "probe", "message"),
    [((8, 9), (True, "ready"), "requires_sm120"),
     ((12, 0), (False, "RuntimeError"), "sage_probe_failed:RuntimeError")],
)
def test_on_fails_loudly_if_sage_is_unavailable(capability, probe, message):
    with pytest.raises(RuntimeError, match=message):
        select_attention("on", capability, probe)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="IMAGINE_SAGE_ATTENTION"):
        select_attention("sometimes", (12, 0), (True, "ready"))
