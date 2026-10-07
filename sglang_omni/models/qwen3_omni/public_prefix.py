"""Server-side provenance and exact token verification for public catalogs."""

import hashlib

from functools import lru_cache

from sglang_omni.models.qwen3_omni.global_action_catalog import load_runtime_action_catalog


# Public prompts owned by services outside the action catalog are approved by
# exact content digest. A prompt edit fails closed until its new digest is
# reviewed here, preventing caller-provided system text from entering the
# cross-session KV namespace. This digest is the turn-router v5 fixed-candidate
# scoring prompt assembled by task_classification.client_model.
_PUBLISHED_EXTERNAL_PROMPT_SHA256 = frozenset(
    {"ee2921ee7b1be5518c0ff8dd545d76e1b966c3e32d2c65787e7a54b8ec9da545"}
)


@lru_cache(maxsize=1)
def published_prompts():
    catalog = load_runtime_action_catalog()
    if catalog.direct_action_selection:
        return frozenset(catalog.action_system_prompt_for(locale, origin)
                         for locale in ("zh-CN", "en-US")
                         for origin in ("user", "proactive"))
    prompts = set(catalog.category_system_prompts_by_locale.values())
    for localized in (catalog.child_system_prompts_by_locale,
                      catalog.proactive_child_system_prompts_by_locale):
        for children in localized.values():
            prompts.update(children.values())
    return frozenset(prompts)


def is_published_prompt(prompt):
    if not isinstance(prompt, str):
        return False
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return digest in _PUBLISHED_EXTERNAL_PROMPT_SHA256 or prompt in published_prompts()


class PublicPrefixVerifier:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor
        # This bounded cache contains only server-published text, never persona.
        self._tokens = lru_cache(maxsize=256)(self._encode)

    def _encode(self, prompt):
        if self.processor is not None:
            rendered = self.processor.apply_chat_template(
                [{"role": "system", "content": [{"type": "text", "text": prompt}]}],
                tokenize=False, add_generation_prompt=False,
            )
            return tuple(self.tokenizer.encode(rendered, add_special_tokens=False))
        return tuple(self.tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt}], tokenize=True,
            add_generation_prompt=False))

    def boundary(self, prompt, full_ids):
        if not is_published_prompt(prompt):
            return 0
        tokens = self._tokens(prompt)
        return len(tokens) if tokens and tuple(full_ids[:len(tokens)]) == tokens else 0
