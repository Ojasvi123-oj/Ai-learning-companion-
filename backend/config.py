import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


def get_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} cannot be empty.")
    return value


def get_int_env(name: str, default: int) -> int:
    value = get_env(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid integer, received {value!r}."
        ) from exc


def get_path_env(name: str, default: Path) -> Path:
    value = os.getenv(name, str(default)).strip()
    if not value:
        raise ValueError(f"{name} cannot be empty.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


GROQ_API_KEY = get_env("GROQ_API_KEY", "")

if GROQ_API_KEY == "your_groq_api_key_here":
    raise RuntimeError(
        "Replace the placeholder GROQ_API_KEY in backend/.env "
        "with your actual Groq API key."
    )

LLM_MODEL = get_env(
    "LLM_MODEL",
    "openai/gpt-oss-120b",
)

EMBED_MODEL = get_env(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

TOP_K = get_int_env("TOP_K", 10)
CHUNK_SIZE = get_int_env("CHUNK_SIZE", 500)
CHUNK_OVERLAP = get_int_env("CHUNK_OVERLAP", 100)

if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1.")

if CHUNK_SIZE < 1:
    raise ValueError("CHUNK_SIZE must be at least 1.")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE."
    )

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = get_path_env(
    "UPLOAD_DIR",
    DATA_DIR / "uploads",
)

INDEX_PATH = get_path_env(
    "INDEX_PATH",
    DATA_DIR / "index" / "index.faiss",
)

META_PATH = get_path_env(
    "META_PATH",
    DATA_DIR / "index" / "meta.pkl",
)

DOC_STORE = get_path_env(
    "DOC_STORE",
    DATA_DIR / "document_store.json",
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
META_PATH.parent.mkdir(parents=True, exist_ok=True)
DOC_STORE.parent.mkdir(parents=True, exist_ok=True)
