from enum import Enum

from insight_engine.prompt.system_prompts import EntityInfo

with open("prompts/audience/CEO.md", "r", encoding="utf-8") as file:
    CEO = EntityInfo(name="CEO", description=file.read())
with open("prompts/audience/CFO.md", "r", encoding="utf-8") as file:
    CFO = EntityInfo(name="CFO", description=file.read())


class AudienceType(Enum):
    CEO = CEO
    CFO = CFO
