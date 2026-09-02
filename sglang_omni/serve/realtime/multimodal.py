"""Public compatibility facade for the multimodal realtime service.

The implementation lives in focused packages and
:mod:`sglang_omni.serve.realtime.multimodal_session`.  Keeping this module
stable preserves the public import path and observability hooks used by
deployments and tests.
"""

from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.models import *  # noqa: F403
from sglang_omni.serve.realtime.multimodal_session import *  # noqa: F403
from sglang_omni.serve.realtime.multimodal_session import random
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.utils.structured_logs import emit_structured_log
