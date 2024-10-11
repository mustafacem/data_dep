import os

from dotenv import load_dotenv

load_dotenv()


MAIN_COLLECTION = os.environ["DB_COLLECTION"]
DB_PATH = "mounted_data/chromadb"
