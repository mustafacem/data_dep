import os
from dataclasses import dataclass


@dataclass
class PromptDescriptor:
    name: str
    prompt: str


class PromptStore:
    def __init__(self, directory: str) -> None:
        self.store: dict[str, PromptDescriptor] = {}
        self.load(directory)

    def load(self, directory: str) -> None:
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                name, _ = os.path.splitext(filename)
                with open(
                    os.path.join(directory, filename), "r", encoding="utf-8"
                ) as file:
                    prompt = file.read()
                    entity_info = PromptDescriptor(name=name, prompt=prompt)
                    self.store[name.upper()] = entity_info

    def __getitem__(self, name: str) -> PromptDescriptor:
        return self.store[name]
