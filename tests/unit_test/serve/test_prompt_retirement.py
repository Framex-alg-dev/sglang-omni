import importlib.util
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.protocol import ChatCompletionRequest


def test_retired_prompt_knobs_are_rejected_instead_of_injecting_hidden_system_text():
 for extra in [{'runtime_prompt_profile':'reply'}]:
  with pytest.raises(ValidationError):ChatCompletionRequest(model='model',messages=[{'role':'user','content':'hello'}],**extra)


def test_old_editor_and_session_routes_are_absent_while_shared_routes_remain():
 app=create_app(object(),model_name='model',enable_resource_monitor=False)
 paths={getattr(route,'path','') for route in app.routes}
 assert '/admin/runtime-prompts' not in paths and '/admin/runtime-prompts/{key}' not in paths
 assert '/v1/realtime' not in paths
 assert '/v1/chat/completions/realtime' in paths and '/v1/performance-control' in paths
 assert '/v1/audio/speech' in paths
 assert not (Path(__file__).resolve().parents[4]/'sglang_omni/serve/realtime/__init__.py').exists()


def test_current_host_reply_language_hint_remains_supported_without_profile_injection():
 value=ChatCompletionRequest(model="model",messages=[{"role":"user","content":"hello"}],runtime_prompt_language="en-US")
 assert value.runtime_prompt_language=="en-US"
