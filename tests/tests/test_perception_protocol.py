from apps.data_flywheel.perception.protocol import (
    PerceptionBackendError,
    UnsupportedCapabilityError,
)


def test_unsupported_capability_error_message():
    err = UnsupportedCapabilityError("sam3", "segment")

    assert err.backend_name == "sam3"
    assert err.capability == "segment"
    assert str(err) == "Backend 'sam3' does not support capability 'segment'"


def test_perception_backend_error_wraps_cause():
    cause = ValueError("boom")
    err = PerceptionBackendError("detvlm", cause)

    assert err.backend_name == "detvlm"
    assert err.cause is cause
    assert str(err) == "Backend 'detvlm' failed: boom"
