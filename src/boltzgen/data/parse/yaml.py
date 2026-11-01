"""Safe YAML loading with diagnostics for repeated explicit mapping keys."""

from collections.abc import Hashable
from typing import Any, TextIO

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class _UniqueKeyLoader(yaml.SafeLoader):
    def __init__(self, stream: str | TextIO) -> None:
        super().__init__(stream)
        self._checked_mappings: set[MappingNode] = set()

    def flatten_mapping(self, node: MappingNode) -> None:
        # Check before flattening: inherited keys and explicit overrides may
        # legitimately overlap. Aliases can revisit an already flattened node.
        if node not in self._checked_mappings:
            self._checked_mappings.add(node)
            keys = set()
            merge_seen = False
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    key = "<<"
                    duplicate = merge_seen
                    merge_seen = True
                else:
                    if key_node.tag == "tag:yaml.org,2002:value":
                        key_node.tag = "tag:yaml.org,2002:str"
                    key = self.construct_object(key_node)
                    if not isinstance(key, Hashable):
                        raise ConstructorError(
                            "while constructing a mapping",
                            node.start_mark,
                            "found unhashable key",
                            key_node.start_mark,
                        )
                    duplicate = key in keys
                    keys.add(key)
                if duplicate:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
        super().flatten_mapping(node)


def load_yaml(stream: str | TextIO) -> Any:
    """Load one safe YAML document, rejecting duplicate explicit keys.

    Merge keys retain normal YAML precedence, including explicit overrides.
    Open file streams retain their filename in parser error messages.
    """
    return yaml.load(stream, Loader=_UniqueKeyLoader)
