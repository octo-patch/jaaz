import pytest

from services.config_service import config_service
from tools.video_providers.minimax_provider import MiniMaxVideoProvider


def make_provider(monkeypatch, url: str = "https://api.minimax.io") -> MiniMaxVideoProvider:
    monkeypatch.setitem(
        config_service.app_config,
        "minimax",
        {
            "api_key": "test-key",
            "url": url,
            "model_name": "MiniMax-H3",
        },
    )
    return MiniMaxVideoProvider()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.minimax.io", "https://api.minimax.io/v2/video_generation"),
        ("https://api.minimax.io/v2", "https://api.minimax.io/v2/video_generation"),
        (
            "https://api.minimax.io/v2/video_generation",
            "https://api.minimax.io/v2/video_generation",
        ),
    ],
)
def test_minimax_builds_video_endpoint(monkeypatch, url: str, expected: str):
    provider = make_provider(monkeypatch, url)
    assert provider._build_api_url() == expected


def test_minimax_builds_frame_roles(monkeypatch):
    provider = make_provider(monkeypatch)
    content = provider._build_content(
        "A cinematic scene",
        input_images=["data:image/png;base64,one", "data:image/png;base64,two"],
    )

    assert content[0] == {"type": "text", "text": "A cinematic scene"}
    assert content[1]["role"] == "first_frame"
    assert content[2]["role"] == "last_frame"


def test_minimax_builds_reference_roles(monkeypatch):
    provider = make_provider(monkeypatch)
    content = provider._build_content(
        "A cinematic scene",
        reference_images=["data:image/png;base64,one"],
        reference_videos=["https://example.com/ref.mp4"],
        reference_audios=["https://example.com/ref.mp3"],
    )

    roles = [item.get("role") for item in content[1:]]
    assert roles == ["reference_image", "reference_video", "reference_audio"]


def test_minimax_rejects_mixed_roles(monkeypatch):
    provider = make_provider(monkeypatch)

    with pytest.raises(ValueError, match="mutually exclusive"):
        provider._build_content(
            "A cinematic scene",
            input_images=["data:image/png;base64,one"],
            reference_images=["data:image/png;base64,two"],
        )


@pytest.mark.parametrize(
    ("resolution", "duration", "ratio", "message"),
    [
        ("1080p", 5, "16:9", "2K resolution"),
        ("2K", 3, "16:9", "between 4 and 15"),
        ("2K", 5, "adaptive", "concrete ratio"),
    ],
)
def test_minimax_validates_request(monkeypatch, resolution, duration, ratio, message):
    provider = make_provider(monkeypatch)
    content = provider._build_content("A cinematic scene")

    with pytest.raises(ValueError, match=message):
        provider._validate_request(resolution, duration, ratio, content)
