"""Compile request-owned masks before the tokenizer-free Omni scheduler."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any


class RequestGrammarCompiler:
    """Cache immutable grammar templates, with a fresh matcher per request.

    OmniScheduler receives already-built Req objects and has no upstream grammar
    admission queue. SamplingParams alone therefore does not install a mask.
    Compile on the existing CPU request-build path and attach the matcher before
    scheduling; compilation failures must reject the request.
    """

    def __init__(self, tokenizer: Any, vocab_size: int, *, cache_size: int = 32):
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._cache_size = cache_size
        self._backend = None
        self._cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = threading.Lock()

    def attach(self, req: Any) -> None:
        sampling = req.sampling_params
        regex = sampling.regex
        schema = sampling.json_schema
        if regex is None and schema is None:
            return
        if regex is not None and schema is not None:
            raise ValueError("regex and JSON schema cannot both constrain a request")
        key = ("regex", regex) if regex is not None else ("json", schema)
        with self._lock:
            grammar = self._cache.get(key)
            if grammar is None:
                if self._backend is None:
                    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend

                    eos = getattr(self._tokenizer, "eos_token_id", None)
                    self._backend = XGrammarGrammarBackend(
                        self._tokenizer, self._vocab_size,
                        model_eos_token_ids=[eos] if isinstance(eos, int) else eos,
                    )
                if key[0] == "regex":
                    grammar = self._backend.dispatch_regex(key[1])
                else:
                    grammar = self._backend.dispatch_json(key[1])
                from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject

                if grammar is None or isinstance(grammar, InvalidGrammarObject):
                    raise ValueError("request output grammar could not be compiled")
                self._cache[key] = grammar
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            req.grammar = grammar.copy()
            req.grammar_key = key
