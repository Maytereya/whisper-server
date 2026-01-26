from configparser import ConfigParser
from pathlib import Path

def get_secret(section: str, key: str, path: str = "whisper_server/secrets.ini") -> str:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Secrets file not found: {p}")

    parser = ConfigParser()
    parser.read(p)

    try:
        value = parser[section][key].strip()
    except KeyError:
        raise RuntimeError(f"Missing [{section}] {key} in {p}")

    if not value:
        raise RuntimeError(f"Empty [{section}] {key} in {p}")

    return value