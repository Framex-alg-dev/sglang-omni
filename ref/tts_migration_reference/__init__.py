"""Portable reference implementation of the project's realtime TTS path."""

from .contracts import SpeechAudioSink, SpeechSynthesisRequest, SpeechSynthesisResult, SpeechSynthesizer
from .realtime_ws_tts import RealtimeWsSpeechSynthesizer, RealtimeWsTtsConfig, RealtimeWsTtsError, SynthesisTrace
from .streaming_bridge import StreamingSpeech, StreamingSpeechMode, StreamingSpeechResult

__all__ = [
    "RealtimeWsSpeechSynthesizer", "RealtimeWsTtsConfig", "RealtimeWsTtsError",
    "SpeechAudioSink", "SpeechSynthesisRequest", "SpeechSynthesisResult",
    "SpeechSynthesizer", "StreamingSpeech", "StreamingSpeechMode",
    "StreamingSpeechResult", "SynthesisTrace",
]
