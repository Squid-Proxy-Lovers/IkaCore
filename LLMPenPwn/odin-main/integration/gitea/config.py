import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    BASE_URL: str = os.getenv("ODIN_BASE_URL", "https://git.uscg.win")
    WORKERS: int = int(os.getenv("ODIN_WORKERS", "4"))

settings = Settings()