import logging
from dataclasses import dataclass
from typing import Any, Optional, Union
from omegaconf import MISSING



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)



