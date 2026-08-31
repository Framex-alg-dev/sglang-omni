"""Explicit composition support for stateless realtime pipeline components."""

from __future__ import annotations

from typing import Any, TypeVar

_T = TypeVar("_T", bound=type[Any])


def compose_components(*components: type[Any]):
    """Install component descriptors on a concrete owner class.

    Components in the realtime pipeline are intentionally stateless. Runtime
    state belongs to the final session object, so descriptors can be composed
    directly without proxy objects or an order-sensitive multiple-inheritance
    hierarchy. Earlier components win on name conflicts, matching the prior
    left-to-right behavior.
    """

    def decorate(owner: _T) -> _T:
        installed: dict[str, type[Any]] = {}
        for component in components:
            providers = getattr(
                component,
                "__component_providers__",
                tuple(item for item in component.__mro__ if item is not object),
            )
            for provider in providers:
                for name, descriptor in provider.__dict__.items():
                    if name.startswith("__") or name in owner.__dict__:
                        continue
                    if name in installed:
                        continue
                    setattr(owner, name, descriptor)
                    installed[name] = provider
        owner.__components__ = components
        owner.__component_providers__ = (
            owner,
            *tuple(
                provider
                for provider in dict.fromkeys(installed.values())
                if provider is not owner
            ),
        )
        owner.__component_owners__ = installed
        return owner

    return decorate
