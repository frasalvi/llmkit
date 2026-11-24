import pytest

from llmkit.azure import arm as arm_mod
from llmkit.azure import source


def test_unit_tests_cannot_build_a_real_arm():
    with pytest.raises(AssertionError, match="real Arm"):
        source.Arm()


@pytest.mark.live
def test_live_tests_get_the_real_arm():
    assert source.Arm is arm_mod.Arm
