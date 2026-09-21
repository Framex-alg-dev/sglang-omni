"""Server-side provenance and exact token verification for public catalogs."""

from functools import lru_cache

from sglang_omni.models.qwen3_omni.global_action_catalog import load_runtime_action_catalog


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
        if not isinstance(prompt, str) or prompt not in published_prompts():
            return 0
        tokens = self._tokens(prompt)
        return len(tokens) if tokens and tuple(full_ids[:len(tokens)]) == tokens else 0
