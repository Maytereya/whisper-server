from configparser import ConfigParser
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "secrets.ini"

_parser = ConfigParser()

if not _CONFIG_PATH.exists():
    raise RuntimeError(
        f"Secrets file not found: {_CONFIG_PATH}. "
        f"Create it from secrets.example.ini"
    )

_parser.read(_CONFIG_PATH)

def get_secret(section: str, key: str) -> str:
    try:
        value = _parser[section][key].strip()
    except KeyError:
        raise RuntimeError(f"Missing [{section}] {key} in secrets.ini")

    if not value:
        raise RuntimeError(f"Empty [{section}] {key} in secrets.ini")

    return value