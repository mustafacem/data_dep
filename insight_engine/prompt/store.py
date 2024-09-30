import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import yaml
from kd_logging import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


def find_vars(text: str) -> list[str]:
    """
    Find and return all variables in a text enclosed within curly braces.

    Args:
        text (str): The input string that may contain variables in the form
            `{variable_name}`.

    Returns:
        list[str]: A list of variable names found in the input string.
    """
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, text)
    return matches


@dataclass
class PromptItem:
    """
    A class to represent a prompt and its metadata.

    Attributes:
        prompt (str): The content of the prompt (usually in md format).
        metadata (dict[str, Any]): A dictionary containing metadata for
            the prompt.
    """

    prompt: str
    metadata: dict[str, Any]


class PromptStore:
    """
    A store that loads and holds prompts from markdown files in a given
    directory. It assumes that prompts are stored in a markdown format
    with front matter yaml metadata.

    It is intended to create one `PromptStore` for one folder of prompts.

    Attributes:
        store (dict[str, PromptItem]): A dictionary that stores prompt
            items with their filenames (without extension) as keys.
    """

    def __init__(self, directory: str) -> None:
        """
        Initializes the PromptStore by loading prompt items from a directory.

        Args:
            directory (str): The directory path where markdown files (.md)
                containing prompt items are stored.
        """
        self.store: dict[str, PromptItem] = {}
        self.load(directory)

    def load_md(self, filepath: str) -> PromptItem:
        """
        Loads a markdown file with optional yaml front matter, extracts the
        prompt and metadata, and returns a PromptItem.

        Args:
            filepath (str): The path to the markdown file to be loaded. If markdown
                has a front matter, it will be extracted into metadata.

        Returns:
            PromptItem: A PromptItem object containing the prompt content
                and metadata extracted from the file.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

            if not content.startswith("---\n"):
                logger.info(
                    f"The prompt in {filepath} is probably missing frontmatter!"
                )
                front_matter, md = "", content
            else:
                _, front_matter, md = content.split("---\n", 2)
            data: dict = yaml.safe_load(front_matter)

        return PromptItem(prompt=md, metadata=data)

    def load(self, directory: str) -> None:
        """
        Loads all markdown files in a given directory and stores them in the
        prompt store.

        Args:
            directory (str): The directory path to scan for markdown files (.md).
        """
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                name, _ = os.path.splitext(filename)
                prompt = self.load_md(filepath=os.path.join(directory, filename))
                self.store[name] = prompt

    def __getitem__(self, name: str) -> PromptItem:
        """
        Retrieves a PromptItem by its name (the filename without the extension).

        Args:
            name (str): The name of the prompt to retrieve.

        Returns:
            PromptItem: The PromptItem object corresponding to the given name.
        """
        return self.store[name]
