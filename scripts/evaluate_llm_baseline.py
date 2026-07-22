"""Evaluate a strict zero-shot LLM baseline on the locked validation split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_IDS = (0, 1, 2)
LABEL_MAPPING = {0: "Tiêu cực", 1: "Bình thường", 2: "Tích cực"}
LABEL_TO_ID = {label: label_id for label_id, label in LABEL_MAPPING.items()}
VALIDATION_COLUMNS = (
    "sample_id",
    "text",
    "text_clean",
    "text_plm",
    "label",
    "label_id",
    "word_count",
    "source",
    "original_split",
)
PREDICTION_COLUMNS = (
    "run_name",
    "provider",
    "model",
    "prompt_version",
    "timestamp_utc",
    "sample_id",
    "text",
    "true_label_id",
    "true_label",
    "raw_response",
    "pred_label_id",
    "pred_label",
    "request_success",
    "parse_valid",
    "error_type",
    "error_message",
    "attempts",
    "latency_ms",
    "response_id",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)
EXPERIMENT_LOG_COLUMNS = (
    "date",
    "model",
    "config",
    "train_size",
    "epoch",
    "val_acc",
    "val_f1_macro",
    "val_f1_negative",
    "val_f1_neutral",
    "val_f1_positive",
    "train_time_min",
    "notes",
)
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_RUN_NAME = "llm_zero_shot_offline_check"
PROMPT_VERSION = "vi_sentiment_zero_shot_v1"
ZERO_SHOT_TEMPLATE = """Bạn là công cụ phân loại cảm xúc văn bản tiếng Việt.
Đọc câu dưới đây và trả lời CHỈ MỘT trong ba nhãn sau:
- Tiêu cực
- Bình thường
- Tích cực

Không giải thích. Không thêm dấu câu. Chỉ trả về đúng một nhãn.

Câu: {text}
Nhãn:"""
PROMPT_HASH = hashlib.sha256(ZERO_SHOT_TEMPLATE.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderResponse:
    """Provider-independent response values used by the artifact writer."""

    text: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics over every selected validation sample, including failures."""

    accuracy: float
    f1_macro: float
    f1_negative: float
    f1_neutral: float
    f1_positive: float
    request_success: int
    request_failure: int
    parse_valid: int
    parse_invalid: int
    mean_latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class RunIdentity:
    """Exact validation snapshot and ordered sample set bound to one run."""

    text_column: str
    sample_count: int
    selected_sample_ids_sha256: str
    validation_file_sha256: str


class NonRetryableProviderError(RuntimeError):
    """A sanitized permanent provider error that must stop the current run."""

    def __init__(
        self,
        error_type: str,
        error_message: str,
        attempts: int,
        latency_ms: float,
    ) -> None:
        super().__init__(
            f"Non-retryable provider error {error_type}: {error_message}"
        )
        self.error_type = error_type
        self.error_message = error_message
        self.attempts = attempts
        self.latency_ms = latency_ms


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a config section after checking that it is a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _repo_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def _utc_now() -> str:
    """Return a compact, timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def metadata_path_for_run(output_path: Path, run_name: str) -> Path:
    """Return the run-scoped metadata path beside a prediction CSV."""
    return output_path.with_name(f"{output_path.stem}.{run_name}.metadata.json")


def sha256_file(path: Path) -> str:
    """Hash every byte of a file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_identity(
    args: argparse.Namespace, selected: pd.DataFrame
) -> RunIdentity:
    """Bind a run to its text column, CSV bytes, and ordered sample IDs."""
    ordered_sample_ids = [str(value) for value in selected["sample_id"]]
    serialized_ids = json.dumps(
        ordered_sample_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return RunIdentity(
        text_column=args.text_column,
        sample_count=len(ordered_sample_ids),
        selected_sample_ids_sha256=hashlib.sha256(serialized_ids).hexdigest(),
        validation_file_sha256=sha256_file(args.input_path),
    )


def load_config(path: Path) -> Mapping[str, Any]:
    """Load the YAML project configuration."""
    resolved_path = _repo_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return _mapping(config, "root")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse offline, preflight, and explicitly enabled execution options."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    preliminary, _ = bootstrap.parse_known_args(argv)
    config = load_config(preliminary.config)
    data_config = _mapping(config.get("data"), "data")
    if "val_path" not in data_config:
        raise ValueError("Config is missing data.val_path.")

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute a strict zero-shot LLM sentiment baseline on "
            "validation data only. API access requires --execute."
        )
    )
    parser.add_argument("--config", type=Path, default=preliminary.config)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(str(data_config["val_path"])),
        help="Validation CSV path (default: data.val_path from config).",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Validation text column sent to the LLM (default: %(default)s).",
    )
    parser.add_argument(
        "--provider",
        choices=("offline", "openai", "gemini"),
        default="offline",
        help="Execution provider (default: %(default)s).",
    )
    parser.add_argument("--model", help="Provider model name; required online.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the selected API and write artifacts.",
    )
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Evaluate every validation row instead of a balanced smoke sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/llm_predictions.csv"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append-experiment-log", action="store_true")
    parser.add_argument(
        "--experiment-log-path",
        type=Path,
        default=Path("results/experiments_log.csv"),
    )
    parser.add_argument("--show-full-prompt", action="store_true")
    parser.add_argument("--print-responses", action="store_true")

    args = parser.parse_args(argv)
    for attribute in ("config", "input_path", "output_path", "experiment_log_path"):
        setattr(args, attribute, _repo_path(getattr(args, attribute)))
    if isinstance(args.model, str):
        args.model = args.model.strip()
    return args


def validate_args(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Validate CLI values and enforce validation-only, results-only paths."""
    data_config = _mapping(config.get("data"), "data")
    training_config = _mapping(config.get("training"), "training")
    if "val_path" not in data_config:
        raise ValueError("Config is missing data.val_path.")

    configured_val_path = _repo_path(Path(str(data_config["val_path"]))).resolve()
    input_path = args.input_path.resolve()
    if input_path != configured_val_path:
        raise ValueError(
            "LLM baseline can only run on the configured validation split "
            f"data.val_path={configured_val_path}; received {input_path}."
        )
    if training_config.get("test_set_locked") is not True:
        raise ValueError("Config must explicitly keep training.test_set_locked=true.")
    if not args.input_path.is_file():
        raise FileNotFoundError(f"Validation CSV does not exist: {args.input_path}")
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError(
            "--run-name must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )

    results_dir = (REPO_ROOT / "results").resolve()
    output_path = args.output_path.resolve()
    metadata_path = metadata_path_for_run(args.output_path, args.run_name).resolve()
    try:
        output_path.relative_to(results_dir)
        metadata_path.relative_to(results_dir)
    except ValueError:
        raise ValueError(
            "Prediction and metadata outputs must remain inside the repository "
            f"results directory: {results_dir}"
        ) from None

    protected_paths = {args.config.resolve(), args.experiment_log_path.resolve()}
    for key in ("train_path", "val_path", "test_path"):
        configured_path = data_config.get(key)
        if configured_path is not None:
            protected_paths.add(
                _repo_path(Path(str(configured_path))).resolve()
            )
    if output_path in protected_paths or metadata_path in protected_paths:
        raise ValueError(
            "Prediction/metadata path conflicts with a protected config, data, "
            "or experiment-log file."
        )

    if args.provider != "offline" and not args.model:
        raise ValueError("--model is required when --provider is openai or gemini.")
    if args.execute and args.provider == "offline":
        raise ValueError("--execute cannot be used with --provider offline.")
    if args.execute and args.run_name == DEFAULT_RUN_NAME:
        raise ValueError(
            "Online execution requires an explicit --run-name; the default "
            "offline run name is not allowed with --execute."
        )
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    if args.append_experiment_log and not args.execute:
        raise ValueError("--append-experiment-log requires --execute.")
    if args.append_experiment_log and not args.full_validation:
        raise ValueError("--append-experiment-log requires --full-validation.")
    if args.sample_size < 1:
        raise ValueError("--sample-size must be at least 1.")
    if not args.text_column.strip():
        raise ValueError("--text-column must be non-empty.")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")
    if args.retry_base_seconds < 0.0 or args.request_delay_seconds < 0.0:
        raise ValueError("Retry and request delays must be non-negative.")


def load_validation_frame(path: Path, text_column: str) -> pd.DataFrame:
    """Read and strictly validate the validation schema and three-class labels."""
    frame = pd.read_csv(path, encoding="utf-8")
    if frame.empty:
        raise ValueError(f"Validation CSV is empty: {path}")

    required = set(VALIDATION_COLUMNS).union({text_column})
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Validation CSV is missing required columns: {missing_columns}")

    checked_columns = ["sample_id", text_column, "label", "label_id"]
    null_counts = frame[checked_columns].isna().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(
            "Validation CSV contains missing required values: "
            f"{null_counts[null_counts > 0].to_dict()}"
        )
    if frame["sample_id"].duplicated().any():
        duplicate_count = int(frame["sample_id"].duplicated().sum())
        raise ValueError(f"sample_id contains {duplicate_count} duplicates.")

    try:
        numeric_labels = pd.to_numeric(frame["label_id"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("label_id must contain integers only.") from exc
    if not (numeric_labels == numeric_labels.astype(int)).all():
        raise ValueError("label_id must contain integers only.")
    frame = frame.copy()
    frame["label_id"] = numeric_labels.astype(int)
    invalid_ids = sorted(set(frame["label_id"]).difference(LABEL_IDS))
    if invalid_ids:
        raise ValueError(f"label_id must belong to {LABEL_IDS}; found {invalid_ids}.")

    expected_labels = frame["label_id"].map(LABEL_MAPPING)
    mismatched = frame.loc[
        expected_labels != frame["label"], ["sample_id", "label", "label_id"]
    ]
    if not mismatched.empty:
        examples = mismatched.head(5).to_dict(orient="records")
        raise ValueError(f"label and label_id mapping mismatch; examples: {examples}")
    return frame


def balanced_smoke_sample(
    frame: pd.DataFrame, sample_size: int, seed: int
) -> pd.DataFrame:
    """Select a deterministic round-robin sample across the three labels."""
    if sample_size > len(frame):
        raise ValueError(
            f"--sample-size={sample_size} exceeds validation size {len(frame)}."
        )
    groups: dict[int, pd.DataFrame] = {}
    for label_id in LABEL_IDS:
        group = frame.loc[frame["label_id"] == label_id]
        if group.empty:
            raise ValueError(f"Validation split has no samples for label_id={label_id}.")
        groups[label_id] = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rows: list[pd.Series] = []
    positions = {label_id: 0 for label_id in LABEL_IDS}
    while len(rows) < sample_size:
        progressed = False
        for label_id in LABEL_IDS:
            position = positions[label_id]
            if position < len(groups[label_id]):
                rows.append(groups[label_id].iloc[position])
                positions[label_id] += 1
                progressed = True
                if len(rows) == sample_size:
                    break
        if not progressed:
            raise RuntimeError("Could not construct the requested balanced sample.")
    return pd.DataFrame(rows).reset_index(drop=True)


def build_zero_shot_prompt(text: str) -> str:
    """Insert NFC-normalized, edge-trimmed text into the one fixed template."""
    normalized_text = unicodedata.normalize("NFC", str(text)).strip()
    if not normalized_text:
        raise ValueError("Cannot build a prompt from empty text.")
    return ZERO_SHOT_TEMPLATE.format(text=normalized_text)


def parse_label(raw_response: str) -> tuple[int | None, str | None]:
    """Accept only one exact label, optionally inside one matching quote pair."""
    normalized = unicodedata.normalize("NFC", str(raw_response)).strip()
    if normalized in LABEL_TO_ID:
        return LABEL_TO_ID[normalized], normalized
    quote_pairs = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’"}
    if len(normalized) >= 2 and normalized[0] in quote_pairs:
        if normalized[-1] == quote_pairs[normalized[0]]:
            inner = normalized[1:-1]
            if inner in LABEL_TO_ID:
                return LABEL_TO_ID[inner], inner
    return None, None


def run_parser_self_check() -> None:
    """Exercise accepted quote styles and required rejection cases."""
    valid_cases = {
        "Tiêu cực": (0, "Tiêu cực"),
        " Bình thường ": (1, "Bình thường"),
        '"Tích cực"': (2, "Tích cực"),
        "'Tiêu cực'": (0, "Tiêu cực"),
        "`Bình thường`": (1, "Bình thường"),
        "“Tích cực”": (2, "Tích cực"),
        "‘Tiêu cực’": (0, "Tiêu cực"),
    }
    invalid_cases = (
        "Tích cực.",
        "Nhãn: Tích cực",
        "Câu này là Tích cực",
        "Trung lập",
        "\"Tích cực'",
        "\" Tích cực \"",
        "",
    )
    for raw, expected in valid_cases.items():
        actual = parse_label(raw)
        if actual != expected:
            raise AssertionError(f"Valid parser case failed: {raw!r} -> {actual!r}")
    for raw in invalid_cases:
        actual = parse_label(raw)
        if actual != (None, None):
            raise AssertionError(f"Invalid parser case was accepted: {raw!r} -> {actual!r}")


def print_preflight(
    args: argparse.Namespace, frame: pd.DataFrame, selected: pd.DataFrame
) -> None:
    """Print protocol details and a bounded prompt preview."""
    distribution = frame["label_id"].value_counts().sort_index()
    title = "OFFLINE LLM ZERO-SHOT CHECK" if args.provider == "offline" else "LLM API PREFLIGHT"
    print(f"===== {title} =====")
    print("Input split: validation only")
    print(f"Validation path: {args.input_path}")
    print(f"Validation rows: {len(frame)}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(not applicable)'}")
    print(f"Run name: {args.run_name}")
    print(f"Text column: {args.text_column}")
    print(f"Selected samples: {len(selected)}")
    print(f"Selection mode: {'full validation' if args.full_validation else 'balanced smoke'}")
    print(f"Output path: {args.output_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Label mapping: {json.dumps(LABEL_MAPPING, ensure_ascii=False)}")
    print("Label distribution:")
    for label_id in LABEL_IDS:
        print(f"  {label_id} ({LABEL_MAPPING[label_id]}): {int(distribution.get(label_id, 0))}")
    print("Parser self-check: ALL PASSED")

    preview = selected if not args.full_validation else selected.head(3)
    print("\nSelected prompt preview:")
    for index, row in preview.iterrows():
        prompt = build_zero_shot_prompt(row[args.text_column])
        print(
            f"\n[{index + 1}] sample_id={row['sample_id']} | "
            f"true={row['label_id']} ({row['label']})"
        )
        if args.show_full_prompt:
            print(prompt)
        else:
            compact_text = " ".join(str(row[args.text_column]).split())
            if len(compact_text) > 180:
                compact_text = compact_text[:177] + "..."
            print(f"text={compact_text}")
            print(f"prompt_chars={len(prompt)}")
    if args.full_validation and len(selected) > len(preview):
        print(f"\n(prompt preview limited to {len(preview)} of {len(selected)} samples)")


def _safe_int(value: Any) -> int | None:
    """Convert an optional SDK usage value to int."""
    return None if value is None else int(value)


def create_provider_client(args: argparse.Namespace) -> tuple[Any, Callable[[Any, str], ProviderResponse], str]:
    """Load the selected key only for execution and construct its installed SDK."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
    key_name = "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"Missing {key_name}; set it in the environment or repository .env file.")

    if args.provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, max_retries=0)

        def request(openai_client: Any, prompt: str) -> ProviderResponse:
            response = openai_client.responses.create(
                model=args.model,
                input=prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                store=False,
            )
            usage = getattr(response, "usage", None)
            return ProviderResponse(
                text=response.output_text or "",
                response_id=getattr(response, "id", None),
                input_tokens=_safe_int(getattr(usage, "input_tokens", None)),
                output_tokens=_safe_int(getattr(usage, "output_tokens", None)),
                total_tokens=_safe_int(getattr(usage, "total_tokens", None)),
            )

        return client, request, api_key

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def request(gemini_client: Any, prompt: str) -> ProviderResponse:
        response = gemini_client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            text=response.text or "",
            response_id=getattr(response, "response_id", None),
            input_tokens=_safe_int(getattr(usage, "prompt_token_count", None)),
            output_tokens=_safe_int(getattr(usage, "candidates_token_count", None)),
            total_tokens=_safe_int(getattr(usage, "total_token_count", None)),
        )

    return client, request, api_key


def sanitize_error(exc: BaseException, secrets: Sequence[str]) -> str:
    """Return a bounded, single-line error message with credentials redacted."""
    message = " ".join(str(exc).split())
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(r"\bAIza[A-Za-z0-9_-]{12,}\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key\s*[=:]\s*)\S+", r"\1[REDACTED]", message
    )
    return message[:1000]


def _exception_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from common OpenAI/Google exception shapes."""
    candidates = (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "code", None),
    )
    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def is_retryable_error(exc: BaseException) -> bool:
    """Retry only known transient network, timeout, rate-limit, and 5xx errors."""
    status_code = _exception_status_code(exc)
    if status_code in (408, 429) or (
        status_code is not None and 500 <= status_code <= 599
    ):
        return True
    if status_code is not None:
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    error_name = type(exc).__name__.lower()
    transient_markers = (
        "timeout",
        "connection",
        "network",
        "ratelimit",
        "rate_limit",
        "temporarilyunavailable",
        "serviceunavailable",
        "internalserver",
        "servererror",
    )
    return any(marker in error_name for marker in transient_markers)


def request_with_retry(
    client: Any,
    request: Callable[[Any, str], ProviderResponse],
    prompt: str,
    max_retries: int,
    retry_base_seconds: float,
    api_key: str,
    *,
    jitter_random: Callable[[], float] = random.random,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ProviderResponse | None, int, str, str, float]:
    """Retry transient errors; raise immediately for permanent/unknown errors."""
    started = time.perf_counter()
    attempts = 0
    error_type = ""
    error_message = ""
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            response = request(client, prompt)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return response, attempts, "", "", latency_ms
        except Exception as exc:  # provider exception families vary by SDK
            error_type = type(exc).__name__
            error_message = sanitize_error(exc, (api_key,))
            if not is_retryable_error(exc):
                latency_ms = (time.perf_counter() - started) * 1000.0
                raise NonRetryableProviderError(
                    error_type,
                    error_message,
                    attempts,
                    latency_ms,
                ) from None
            if attempt < max_retries:
                base_delay = retry_base_seconds * (2**attempt)
                jitter = base_delay * 0.1 * jitter_random()
                sleep(base_delay + jitter)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return None, attempts, error_type, error_message, latency_ms


def _truthy(value: Any) -> bool:
    """Interpret CSV booleans consistently after round-tripping through pandas."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_existing_predictions(path: Path) -> pd.DataFrame:
    """Load and validate a resumable prediction artifact."""
    try:
        frame = pd.read_csv(path, encoding="utf-8", dtype={"sample_id": "string"})
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Existing prediction file is empty: {path}") from exc
    missing = sorted(set(PREDICTION_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Existing prediction file is missing columns: {missing}")
    frame = frame.loc[:, list(PREDICTION_COLUMNS)]
    assert_unique_prediction_keys(frame)
    return frame


def assert_unique_prediction_keys(frame: pd.DataFrame) -> None:
    """Require at most one artifact row for each (run_name, sample_id) pair."""
    if frame.empty:
        return
    duplicate_mask = frame.duplicated(
        subset=["run_name", "sample_id"], keep=False
    )
    if duplicate_mask.any():
        examples = (
            frame.loc[duplicate_mask, ["run_name", "sample_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Duplicate prediction keys (run_name, sample_id) detected before "
            f"atomic write: {examples}"
        )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Replace a CSV atomically from a temporary file in the same directory."""
    assert_unique_prediction_keys(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            frame.to_csv(temporary, index=False, columns=list(PREDICTION_COLUMNS))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Replace a JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def expected_resume_metadata(
    args: argparse.Namespace, run_identity: RunIdentity
) -> dict[str, Any]:
    """Return immutable run settings that must match during resume."""
    return {
        "provider": args.provider,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "validation_path": str(args.input_path.resolve()),
        "prompt_hash_sha256": PROMPT_HASH,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
        "full_validation": args.full_validation,
        **asdict(run_identity),
    }


def validate_resume_metadata(
    args: argparse.Namespace, run_identity: RunIdentity
) -> Path:
    """Require the current run's metadata and exact immutable configuration."""
    metadata_path = metadata_path_for_run(args.output_path, args.run_name)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume run {args.run_name!r}: metadata does not exist: "
            f"{metadata_path}"
        )
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Resume metadata is invalid JSON: {metadata_path}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Resume metadata must contain a JSON object: {metadata_path}")

    mismatches: list[str] = []
    for field, expected in expected_resume_metadata(args, run_identity).items():
        actual = metadata.get(field)
        if field == "validation_path" and actual is not None:
            matches = Path(str(actual)).resolve() == Path(str(expected)).resolve()
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{field}: existing={actual!r}, requested={expected!r}")
    if mismatches:
        raise ValueError(
            f"Cannot resume run {args.run_name!r} with a different configuration: "
            + "; ".join(mismatches)
        )
    return metadata_path


def validate_resume_prediction_config(
    run_rows: pd.DataFrame, args: argparse.Namespace
) -> None:
    """Require every existing row for a run to match its provider configuration."""
    if run_rows.empty:
        return
    expected = {
        "provider": args.provider,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
    }
    mismatches: list[str] = []
    for field, expected_value in expected.items():
        values = run_rows[field]
        mismatch_mask = values.isna() | (
            values.astype(str) != str(expected_value)
        )
        if mismatch_mask.any():
            actual_values = sorted(
                {"<missing>" if pd.isna(value) else str(value) for value in values}
            )
            mismatches.append(
                f"{field}: existing={actual_values!r}, requested={expected_value!r}"
            )
    if mismatches:
        raise ValueError(
            f"Cannot resume run {args.run_name!r}; prediction configuration "
            "does not match: " + "; ".join(mismatches)
        )


def prepare_output(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    run_identity: RunIdentity,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    """Validate output policy and determine which selected samples need requests."""
    empty = pd.DataFrame(columns=PREDICTION_COLUMNS)
    if not args.output_path.exists():
        if args.resume:
            raise FileNotFoundError(
                f"Cannot resume because prediction file does not exist: {args.output_path}"
            )
        return empty, {}, [str(value) for value in selected["sample_id"]]
    if args.overwrite:
        if args.output_path.stat().st_size == 0:
            base = empty
        else:
            existing = load_existing_predictions(args.output_path)
            base = existing.loc[
                existing["run_name"].astype(str) != args.run_name
            ].copy()
        return base, {}, [str(value) for value in selected["sample_id"]]
    if not args.resume:
        raise FileExistsError(
            f"Prediction file already exists; use --resume or --overwrite: {args.output_path}"
        )

    validate_resume_metadata(args, run_identity)
    existing = load_existing_predictions(args.output_path)
    run_rows = existing.loc[existing["run_name"].astype(str) == args.run_name]
    validate_resume_prediction_config(run_rows, args)
    selected_keys = {str(value) for value in selected["sample_id"]}
    successful: dict[str, dict[str, Any]] = {}
    for _, row in run_rows.iterrows():
        key = str(row["sample_id"])
        if key in selected_keys and _truthy(row["request_success"]):
            successful[key] = row.to_dict()

    retained_mask = ~(
        (existing["run_name"].astype(str) == args.run_name)
        & existing["sample_id"].astype(str).isin(selected_keys)
    )
    base = existing.loc[retained_mask].copy()
    pending = [
        str(value) for value in selected["sample_id"] if str(value) not in successful
    ]
    return base, successful, pending


def build_prediction_row(
    args: argparse.Namespace,
    row: pd.Series,
    response: ProviderResponse | None,
    attempts: int,
    error_type: str,
    error_message: str,
    latency_ms: float,
) -> dict[str, Any]:
    """Build one stable-schema artifact row."""
    raw_response = response.text if response is not None else ""
    pred_id, pred_label = parse_label(raw_response) if response is not None else (None, None)
    if response is not None and pred_id is None:
        error_type = "InvalidResponse"
        error_message = "Response did not exactly match one allowed label."
    return {
        "run_name": args.run_name,
        "provider": args.provider,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "timestamp_utc": _utc_now(),
        "sample_id": str(row["sample_id"]),
        "text": unicodedata.normalize("NFC", str(row[args.text_column])).strip(),
        "true_label_id": int(row["label_id"]),
        "true_label": row["label"],
        "raw_response": raw_response,
        "pred_label_id": pred_id,
        "pred_label": pred_label,
        "request_success": response is not None,
        "parse_valid": pred_id is not None,
        "error_type": error_type,
        "error_message": error_message,
        "attempts": attempts,
        "latency_ms": round(latency_ms, 3),
        "response_id": response.response_id if response is not None else None,
        "input_tokens": response.input_tokens if response is not None else None,
        "output_tokens": response.output_tokens if response is not None else None,
        "total_tokens": response.total_tokens if response is not None else None,
    }


def calculate_metrics(records: Sequence[Mapping[str, Any]]) -> EvaluationMetrics:
    """Score every selected row, mapping request/parse failures to an invalid ID."""
    targets = [int(record["true_label_id"]) for record in records]
    predictions = [
        int(record["pred_label_id"])
        if record.get("pred_label_id") is not None and not pd.isna(record.get("pred_label_id"))
        else -1
        for record in records
    ]
    per_class = f1_score(
        targets, predictions, labels=list(LABEL_IDS), average=None, zero_division=0
    )
    success_count = sum(_truthy(record["request_success"]) for record in records)
    valid_count = sum(_truthy(record["parse_valid"]) for record in records)
    latencies = [float(record["latency_ms"]) for record in records]

    def token_sum(name: str) -> int | None:
        values = [
            int(float(record[name]))
            for record in records
            if record.get(name) is not None and not pd.isna(record.get(name))
        ]
        return sum(values) if values else None

    return EvaluationMetrics(
        accuracy=float(accuracy_score(targets, predictions)),
        f1_macro=float(
            f1_score(
                targets,
                predictions,
                labels=list(LABEL_IDS),
                average="macro",
                zero_division=0,
            )
        ),
        f1_negative=float(per_class[0]),
        f1_neutral=float(per_class[1]),
        f1_positive=float(per_class[2]),
        request_success=success_count,
        request_failure=len(records) - success_count,
        parse_valid=valid_count,
        parse_invalid=len(records) - valid_count,
        mean_latency_ms=sum(latencies) / len(latencies),
        input_tokens=token_sum("input_tokens"),
        output_tokens=token_sum("output_tokens"),
        total_tokens=token_sum("total_tokens"),
    )


def print_metrics(metrics: EvaluationMetrics) -> None:
    """Print the requested evaluation summary."""
    print("\n===== VALIDATION METRICS =====")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Macro-F1: {metrics.f1_macro:.4f}")
    print(f"F1 Negative: {metrics.f1_negative:.4f}")
    print(f"F1 Neutral: {metrics.f1_neutral:.4f}")
    print(f"F1 Positive: {metrics.f1_positive:.4f}")
    print(f"Requests successful/failed: {metrics.request_success}/{metrics.request_failure}")
    print(f"Responses parse valid/invalid: {metrics.parse_valid}/{metrics.parse_invalid}")
    print(f"Mean latency: {metrics.mean_latency_ms:.3f} ms")
    print(
        "Token usage (input/output/total): "
        f"{metrics.input_tokens}/{metrics.output_tokens}/{metrics.total_tokens}"
    )


def experiment_log_fieldnames(path: Path) -> list[str]:
    """Validate and return the existing experiment-log schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Experiment log does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        fieldnames = next((row for row in reader if row), None)
    if not fieldnames:
        raise ValueError(f"Experiment log has no header: {path}")
    if list(fieldnames) != list(EXPERIMENT_LOG_COLUMNS):
        raise ValueError(
            f"Experiment log schema mismatch; expected {list(EXPERIMENT_LOG_COLUMNS)}, "
            f"found {fieldnames}."
        )
    return list(fieldnames)


def ensure_run_not_logged(path: Path, run_name: str) -> None:
    """Reject a duplicate run name found in structured config or notes."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            config_value = row.get("config", "")
            try:
                config = json.loads(config_value)
            except (TypeError, json.JSONDecodeError):
                config = {}
            if isinstance(config, Mapping) and config.get("run_name") == run_name:
                raise ValueError(f"Experiment log already contains run_name={run_name!r}.")
            if f"run_name={run_name}" in row.get("notes", ""):
                raise ValueError(f"Experiment log already contains run_name={run_name!r}.")


def append_experiment_log(
    args: argparse.Namespace,
    validation_size: int,
    metrics: EvaluationMetrics,
    elapsed_seconds: float,
) -> None:
    """Append one completed full-validation run without rewriting old rows."""
    fieldnames = experiment_log_fieldnames(args.experiment_log_path)
    ensure_run_not_logged(args.experiment_log_path, args.run_name)
    config = {
        "run_name": args.run_name,
        "provider": args.provider,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
    }
    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": f"LLM_{args.provider.upper()}_{args.model}",
        "config": json.dumps(config, ensure_ascii=False, separators=(",", ":")),
        "train_size": 0,
        "epoch": "-",
        "val_acc": f"{metrics.accuracy:.4f}",
        "val_f1_macro": f"{metrics.f1_macro:.4f}",
        "val_f1_negative": f"{metrics.f1_negative:.4f}",
        "val_f1_neutral": f"{metrics.f1_neutral:.4f}",
        "val_f1_positive": f"{metrics.f1_positive:.4f}",
        "train_time_min": f"{elapsed_seconds / 60.0:.4f}",
        "notes": (
            f"run_name={args.run_name}; zero-shot validation; validation_size="
            f"{validation_size}; test.csv locked and not used"
        ),
    }
    with args.experiment_log_path.open("a", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=fieldnames).writerow(row)


def build_run_metadata(
    args: argparse.Namespace,
    run_identity: RunIdentity,
    status: str,
    metrics: EvaluationMetrics | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Build run-scoped metadata for both resumable and completed states."""
    return {
        "run_name": args.run_name,
        **expected_resume_metadata(args, run_identity),
        "prompt_template": ZERO_SHOT_TEMPLATE,
        "status": status,
        "timestamp_utc": _utc_now(),
        "metrics": asdict(metrics) if metrics is not None else None,
        "elapsed_seconds": elapsed_seconds,
        "invalid_response_policy": (
            "Request failures and responses not exactly matching one allowed label "
            "are assigned an invalid prediction and counted as incorrect."
        ),
        "test_set_used": False,
        "test_set_confirmation": "test.csv was not read or used",
    }


def write_prediction_checkpoint(
    base: pd.DataFrame,
    current_records: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Write base plus one unique current-run record per selected sample."""
    run_frame = pd.DataFrame(current_records.values(), columns=PREDICTION_COLUMNS)
    checkpoint = pd.concat([base, run_frame], ignore_index=True)
    assert_unique_prediction_keys(checkpoint)
    atomic_write_csv(checkpoint, output_path)


def run_online(
    args: argparse.Namespace, selected: pd.DataFrame
) -> tuple[EvaluationMetrics, Path, int, float]:
    """Execute pending requests, checkpoint each result, and save metadata."""
    run_identity = build_run_identity(args, selected)
    base, successful, pending = prepare_output(args, selected, run_identity)
    selected_by_key = {
        str(row["sample_id"]): row for _, row in selected.iterrows()
    }
    current_records: dict[str, dict[str, Any]] = dict(successful)
    run_started = time.perf_counter()
    api_calls = 0
    client: Any | None = None
    retry_rng = random.Random(args.seed)
    metadata_path = metadata_path_for_run(args.output_path, args.run_name)
    atomic_write_json(
        build_run_metadata(args, run_identity, status="in_progress"),
        metadata_path,
    )

    try:
        if pending:
            client, request, api_key = create_provider_client(args)
            for position, sample_key in enumerate(pending):
                row = selected_by_key[sample_key]
                prompt = build_zero_shot_prompt(row[args.text_column])
                try:
                    response, attempts, error_type, error_message, latency_ms = (
                        request_with_retry(
                            client,
                            request,
                            prompt,
                            args.max_retries,
                            args.retry_base_seconds,
                            api_key,
                            jitter_random=retry_rng.random,
                        )
                    )
                except NonRetryableProviderError as exc:
                    api_calls += exc.attempts
                    record = build_prediction_row(
                        args,
                        row,
                        None,
                        exc.attempts,
                        exc.error_type,
                        exc.error_message,
                        exc.latency_ms,
                    )
                    current_records[sample_key] = record
                    write_prediction_checkpoint(
                        base, current_records, args.output_path
                    )
                    print(
                        f"[{len(current_records)}/{len(selected)}] "
                        f"sample_id={sample_key}: permanent_error "
                        f"({exc.error_type}), attempts={exc.attempts}"
                    )
                    raise
                api_calls += attempts
                record = build_prediction_row(
                    args,
                    row,
                    response,
                    attempts,
                    error_type,
                    error_message,
                    latency_ms,
                )
                current_records[sample_key] = record
                write_prediction_checkpoint(base, current_records, args.output_path)
                if response is None:
                    status = f"failed ({error_type})"
                elif not record["parse_valid"]:
                    status = "invalid_response"
                else:
                    status = "ok"
                print(
                    f"[{len(current_records)}/{len(selected)}] sample_id={sample_key}: "
                    f"{status}, attempts={attempts}"
                )
                if args.print_responses:
                    print(f"raw_response={record['raw_response']!r}")
                if position + 1 < len(pending) and args.request_delay_seconds:
                    time.sleep(args.request_delay_seconds)
    finally:
        if client is not None:
            client.close()

    missing = [
        str(value) for value in selected["sample_id"] if str(value) not in current_records
    ]
    if missing:
        raise RuntimeError(f"Run finished without prediction records for samples: {missing[:5]}")
    ordered_records = [
        current_records[str(value)] for value in selected["sample_id"]
    ]
    metrics = calculate_metrics(ordered_records)
    elapsed_seconds = time.perf_counter() - run_started
    completion_status = (
        "complete" if metrics.request_failure == 0 else "complete_with_failures"
    )
    atomic_write_json(
        build_run_metadata(
            args,
            run_identity,
            status=completion_status,
            metrics=metrics,
            elapsed_seconds=elapsed_seconds,
        ),
        metadata_path,
    )

    if args.append_experiment_log:
        if len(selected) != len(selected_by_key):
            raise RuntimeError("Selected validation sample IDs are not unique.")
        if metrics.request_failure:
            raise RuntimeError(
                "Cannot append experiment log while request failures remain; resume first."
            )
        append_experiment_log(args, len(selected), metrics, elapsed_seconds)
    return metrics, metadata_path, api_calls, elapsed_seconds


def main(argv: Sequence[str] | None = None) -> None:
    """Run offline checks, API preflight, or explicit online evaluation."""
    args = parse_args(argv)
    config = load_config(args.config)
    validate_args(args, config)
    run_parser_self_check()
    frame = load_validation_frame(args.input_path, args.text_column)
    selected = (
        frame.reset_index(drop=True)
        if args.full_validation
        else balanced_smoke_sample(frame, args.sample_size, args.seed)
    )
    print_preflight(args, frame, selected)

    if not args.execute:
        suffix = "" if args.provider == "offline" else " (--execute not supplied)"
        print(f"\nAPI called: no{suffix}")
        print("Files written: no")
        print("test.csv used: no")
        if args.provider == "offline":
            print("Offline zero-shot scaffold: ALL PASSED")
        else:
            print("API preflight: ALL PASSED")
        return

    metrics, metadata_path, api_calls, elapsed_seconds = run_online(args, selected)
    print_metrics(metrics)
    print(f"\nAPI request attempts: {api_calls}")
    print(f"Prediction path: {args.output_path}")
    print(f"Metadata path: {metadata_path}")
    print(f"Experiment log updated: {'yes' if args.append_experiment_log else 'no'}")
    print(f"Elapsed time: {elapsed_seconds:.3f} seconds")
    print("test.csv used: no")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
