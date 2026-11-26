from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.models.qwen3_omni.request_grammar import RequestGrammarCompiler


def request(regex=None, schema=None):
    return SimpleNamespace(sampling_params=SimpleNamespace(regex=regex, json_schema=schema), grammar=None)


def test_unconstrained_request_does_not_initialize_a_backend():
    compiler = RequestGrammarCompiler(None, 10)
    req = request()
    compiler.attach(req)
    assert req.grammar is None
    assert compiler._backend is None


def test_cached_template_has_an_independent_matcher_for_each_request():
    compiler = RequestGrammarCompiler(None, 10)
    first, second = object(), object()
    template = Mock()
    template.copy.side_effect = [first, second]
    backend = Mock()
    backend.dispatch_regex.return_value = template
    compiler._backend = backend
    a, b = request("1N"), request("1N")
    compiler.attach(a); compiler.attach(b)
    assert a.grammar is first and b.grammar is second
    backend.dispatch_regex.assert_called_once_with("1N")


def test_compilation_failure_does_not_admit_an_unconstrained_reply():
    from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject

    compiler = RequestGrammarCompiler(None, 10)
    compiler._backend = Mock()
    compiler._backend.dispatch_regex.return_value = InvalidGrammarObject()
    req = request("[")
    with pytest.raises(ValueError, match="could not be compiled"):
        compiler.attach(req)
    assert req.grammar is None
    assert not compiler._cache


def test_conflicting_grammars_are_rejected():
    compiler = RequestGrammarCompiler(None, 10)
    with pytest.raises(ValueError, match="cannot both constrain"):
        compiler.attach(request("x", "{}"))


def test_template_cache_is_bounded_and_json_schema_uses_the_same_seam():
    compiler = RequestGrammarCompiler(None, 10, cache_size=1)
    compiler._backend = Mock()
    compiler.attach(request("first"))
    compiler.attach(request(schema="{}"))
    compiler._backend.dispatch_json.assert_called_once_with("{}")
    assert list(compiler._cache) == [("json", "{}")]
