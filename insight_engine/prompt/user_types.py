from enum import Enum

from insight_engine.prompt.system_prompts import EntityInfo

with open("prompts/user_type/CRP.md", "r", encoding="utf-8") as file:
    CRP = EntityInfo(name="Client Relationship Partner", description=file.read())
with open("prompts/user_type/transaction_lawyer.md", "r", encoding="utf-8") as file:
    TRANS_LAWYER = EntityInfo(name="Transactional Lawyer", description=file.read())
with open("prompts/user_type/BD.md", "r", encoding="utf-8") as file:
    BD = EntityInfo(name="Business Development Professional", description=file.read())


class UserType(Enum):
    CRP = CRP
    TRANS_LAWYER = TRANS_LAWYER
    BD = BD
