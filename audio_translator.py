"""Translate ASR transcription files (eng.txt) to Japanese.

Translates English lecture transcriptions to Japanese using a local
Qwen3.5 model (default) or OpenAI API fallback. Outputs translation.txt
files in the same format as existing legacy translations.

Usage:
    python audio_translator.py <input>
    python audio_translator.py <input> --provider openai
    python audio_translator.py <input> --output-dir ./output
    python audio_translator.py <input> --config my_config.json

Input:
    Accepts a single file, a directory, or a glob pattern:
      - "Lecture 36 - Module 1 eng.txt"   (single file)
      - "./_text/MSE403/"                  (all eng.txt in directory)
      - "./_text/**/*eng.txt"              (glob pattern)

Output:
    [MM:SS.mmm --> MM:SS.mmm]  English sentence.
     日本語翻訳文。

    One English-Japanese pair per block, followed by a full English
    text block at the end.

Configuration:
    Settings are read from config.json["translation"].
    CLI flags override config values. See config.json for schema.
"""
import argparse
import csv
import glob as globmod
import json
import os
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# eng.txt parser
# ---------------------------------------------------------------------------

_TS_PATTERN = re.compile(r"^\[(.+?)\]\s*(.+)$")


def parse_eng_txt(path: str | Path) -> list[dict]:
    """Parse eng.txt into list of {line, timestamp, text}."""
    sentences = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            m = _TS_PATTERN.match(raw)
            if m:
                sentences.append({
                    "line": raw,
                    "timestamp": m.group(1),
                    "text": m.group(2),
                })
    return sentences


# ---------------------------------------------------------------------------
# Glossary loader
# ---------------------------------------------------------------------------

def load_glossary(tsv_path: str | Path) -> dict[str, str]:
    """Load EN->JA glossary from TSV file."""
    glossary: dict[str, str] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            glossary[row["en"]] = row["ja"]
    return glossary


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_TRANSLATION_DEFAULTS = {
    "provider": "local",
    "model": "Qwen/Qwen3.5-9B",
    "dtype": "bfloat16",
    "glossary": "_glossary/mse.tsv",
    "target_language": "ja",
    "max_sentences_per_chunk": 100,
}


def load_translation_config(
    config_path: str = "config.json", **cli_overrides: str | None
) -> dict:
    """Load translation config from config.json with CLI overrides."""
    config_file = Path(config_path)
    if config_file.exists():
        full = json.loads(config_file.read_text(encoding="utf-8"))
        config = full.get("translation", {})
    else:
        config = {}

    for key, default in _TRANSLATION_DEFAULTS.items():
        config.setdefault(key, default)

    for key, val in cli_overrides.items():
        if val is not None:
            config[key] = val

    return config


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def resolve_input_files(input_path: str) -> list[Path]:
    """Resolve input to a list of eng.txt file paths."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(
            f for f in p.iterdir()
            if f.is_file() and f.name.endswith("eng.txt")
        )
    # glob pattern
    matches = globmod.glob(input_path, recursive=True)
    return sorted(
        Path(m) for m in matches
        if Path(m).is_file() and m.endswith("eng.txt")
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
あなたは材料工学の専門翻訳者です。以下のルールに厳密に従って英文を日本語に翻訳してください。

## ルール
- 番号付きの英文が与えられます。同じ番号で日本語訳を返してください。
- 専門用語は下記の用語集に従ってください。
- 講義の口語表現（um, you know, basically 等のフィラー）は自然に整えてください。ただし意味は変えないこと。
- 内容の追加・省略・解釈は一切禁止です。
- 出力は「番号: 日本語訳」の形式のみ。説明や注釈は不要です。

## 用語集（必ず遵守）
{glossary_table}
"""


def build_prompt(texts: list[str], glossary: dict[str, str]) -> tuple[str, str]:
    """Build system and user prompt strings.

    Returns (system_prompt, user_prompt).
    """
    glossary_table = "\n".join(
        f"| {en} | {ja} |" for en, ja in glossary.items()
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(glossary_table=glossary_table)
    user_prompt = "\n".join(
        f"{i + 1}: {t}" for i, t in enumerate(texts)
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_RESP_PATTERN = re.compile(r"^(\d+):\s*(.+)$")
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_translations(response_text: str, expected_count: int) -> list[str]:
    """Parse numbered translations from model response.

    Strips any <think>...</think> blocks (Qwen3.5 thinking mode).
    """
    cleaned = _THINK_PATTERN.sub("", response_text).strip()
    translations: dict[int, str] = {}
    duplicates: list[int] = []
    for line in cleaned.splitlines():
        m = _RESP_PATTERN.match(line.strip())
        if m:
            idx = int(m.group(1))
            if idx in translations:
                duplicates.append(idx)
            translations[idx] = m.group(2)

    if duplicates:
        raise ValueError(
            f"Duplicate translation indices: {sorted(set(duplicates))}"
        )

    expected_indices = set(range(1, expected_count + 1))
    actual_indices = set(translations.keys())
    if actual_indices != expected_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        raise ValueError(
            f"Expected translations 1..{expected_count}, "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [translations[i + 1] for i in range(expected_count)]


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------

OVERLAP_SENTENCES = 5


def chunk_sentences(
    texts: list[str],
    max_sentences: int,
    tokenizer=None,
    context_budget_ratio: float = 0.4,
) -> list[tuple[int, int]]:
    """Split sentence indices into chunks.

    Returns list of (start_idx, end_idx) tuples (end exclusive).
    Uses token budget as primary metric when tokenizer is available,
    with max_sentences as a cap.
    """
    if not texts:
        return []
    if max_sentences < 1:
        raise ValueError(f"max_sentences_per_chunk must be >= 1, got {max_sentences}")

    # Ensure overlap doesn't exceed half of chunk size to guarantee forward progress
    effective_overlap = min(OVERLAP_SENTENCES, max_sentences // 2)

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        model_max = getattr(tokenizer, "model_max_length", 32768)
        token_budget = int(model_max * context_budget_ratio)
        chunks = []
        start = 0
        while start < len(texts):
            end = start
            total_tokens = 0
            while end < len(texts) and (end - start) < max_sentences:
                line_tokens = len(tokenizer.encode(texts[end], add_special_tokens=False))
                if total_tokens + line_tokens > token_budget and end > start:
                    break
                total_tokens += line_tokens
                end += 1
                if end == start + 1 and total_tokens > token_budget:
                    print(f"  WARNING: Sentence {start + 1} exceeds token budget "
                          f"({total_tokens} > {token_budget})")
                    break
            chunks.append((start, end))
            if end >= len(texts):
                break
            start = end - effective_overlap
            if start <= chunks[-1][0]:
                start = end
        return chunks
    else:
        # Sentence-count only (API path)
        chunks = []
        start = 0
        while start < len(texts):
            end = min(start + max_sentences, len(texts))
            chunks.append((start, end))
            if end >= len(texts):
                break
            start = end - effective_overlap
            if start <= chunks[-1][0]:
                start = end
        return chunks


def merge_chunk_translations(
    chunks: list[tuple[int, int]],
    chunk_results: list[list[str]],
    total_count: int,
) -> list[str]:
    """Merge translations from overlapping chunks.

    For overlapping regions, the later chunk's translation is used.
    """
    merged = [""] * total_count
    for (start, end), translations in zip(chunks, chunk_results):
        for i, t in enumerate(translations):
            merged[start + i] = t
    return merged


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_translation_txt(
    sentences: list[dict],
    translations: list[str],
    output_path: Path,
) -> None:
    """Write translation.txt in the existing legacy format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent, ja in zip(sentences, translations):
            f.write(f"{sent['line']}\n")
            f.write(f"{ja}\n")
            f.write("\n")

        full_text = " ".join(s["text"] for s in sentences)
        f.write(f'" {full_text}"\n')


# ---------------------------------------------------------------------------
# Local model translation
# ---------------------------------------------------------------------------

def load_local_model(
    model_id: str, dtype_str: str, device: str
) -> tuple:
    """Download (if needed) and load translation model.

    Returns (tokenizer, model).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from _utils.model import ensure_model_downloaded

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    model_name = model_id.split("/")[-1]
    local_dir = PROJECT_ROOT / "models" / model_name
    model_path = ensure_model_downloaded(model_id, local_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=dtype, device_map=device,
    )
    return tokenizer, model


def translate_local(
    system_prompt: str,
    user_prompt: str,
    tokenizer,
    model,
) -> str:
    """Run local model inference and return raw text."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=0.3,
            do_sample=True,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# OpenAI API translation
# ---------------------------------------------------------------------------

_OPENAI_CONFIG = {
    "api_key_env": "OPENAI_API_KEY",
    "default_model": "gpt-4.1",
}


def create_api_client(provider: str):
    """Create an API client for the given provider.

    Returns (client, default_model).
    """
    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get(_OPENAI_CONFIG["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"Environment variable {_OPENAI_CONFIG['api_key_env']} is not set.\n"
                f"Set it with: set {_OPENAI_CONFIG['api_key_env']}=your-key"
            )
        return OpenAI(api_key=api_key), _OPENAI_CONFIG["default_model"]
    else:
        raise ValueError(
            f"Unsupported API provider: '{provider}'. "
            f"Supported: 'local', 'openai'"
        )


def translate_openai(
    system_prompt: str,
    user_prompt: str,
    client,
    model: str,
) -> str:
    """Translate via OpenAI API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Unified translate_sentences
# ---------------------------------------------------------------------------

def translate_sentences(
    texts: list[str],
    glossary: dict[str, str],
    *,
    provider: str = "local",
    tokenizer=None,
    model=None,
    api_client=None,
    api_model: str | None = None,
    max_retries: int = 1,
) -> list[str]:
    """Translate a list of English sentences to Japanese.

    Common interface for all providers. Returns list[str] of same length.
    """
    system_prompt, user_prompt = build_prompt(texts, glossary)

    for attempt in range(1 + max_retries):
        try:
            if provider == "local":
                raw = translate_local(system_prompt, user_prompt, tokenizer, model)
            elif provider == "openai":
                raw = translate_openai(system_prompt, user_prompt, api_client, api_model)
            else:
                raise ValueError(f"Unsupported provider: '{provider}'")

            return parse_translations(raw, len(texts))

        except ValueError as e:
            if attempt < max_retries and provider == "local":
                print(f"  Retry ({attempt + 1}/{max_retries}): {e}")
                continue
            raise


# ---------------------------------------------------------------------------
# File-level translation
# ---------------------------------------------------------------------------

def translate_file(
    eng_path: Path,
    output_path: Path,
    glossary: dict[str, str],
    config: dict,
    *,
    tokenizer=None,
    model=None,
    api_client=None,
    api_model: str | None = None,
) -> None:
    """Translate a single eng.txt file."""
    provider = config["provider"]
    max_sentences = config.get("max_sentences_per_chunk", 100)

    sentences = parse_eng_txt(eng_path)
    if not sentences:
        print(f"  SKIP (empty): {eng_path}")
        return

    texts = [s["text"] for s in sentences]

    # Chunk if needed
    chunks = chunk_sentences(
        texts, max_sentences,
        tokenizer=tokenizer if provider == "local" else None,
    )

    if len(chunks) == 1:
        translations = translate_sentences(
            texts, glossary,
            provider=provider,
            tokenizer=tokenizer, model=model,
            api_client=api_client, api_model=api_model,
        )
    else:
        print(f"  Splitting into {len(chunks)} chunks")
        chunk_results = []
        for ci, (start, end) in enumerate(chunks):
            chunk_texts = texts[start:end]
            print(f"    Chunk {ci + 1}/{len(chunks)}: "
                  f"sentences {start + 1}-{end} ({len(chunk_texts)} sentences)")
            result = translate_sentences(
                chunk_texts, glossary,
                provider=provider,
                tokenizer=tokenizer, model=model,
                api_client=api_client, api_model=api_model,
            )
            chunk_results.append(result)
        translations = merge_chunk_translations(chunks, chunk_results, len(texts))

    write_translation_txt(sentences, translations, output_path)
    print(f"  OK: {output_path.name} ({len(sentences)} sentences)")


def translate_files(files: list[Path], config: dict) -> None:
    """Translate multiple eng.txt files."""
    provider = config["provider"]
    glossary_path = config.get("glossary", "_glossary/mse.tsv")
    glossary = load_glossary(PROJECT_ROOT / glossary_path)
    output_dir = config.get("output_dir")

    tokenizer = None
    model = None
    api_client = None
    api_model = None

    if provider == "local":
        import torch
        from _utils.model import print_vram

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Provider:  local")
        print(f"Model:     {config['model']}")
        print(f"Device:    {device}")
        if torch.cuda.is_available():
            print(f"GPU:       {torch.cuda.get_device_name()}")
        print_vram("start")

        tokenizer, model = load_local_model(
            config["model"], config.get("dtype", "bfloat16"), device,
        )
        print_vram("after model load")
    elif provider == "openai":
        api_client, default_model = create_api_client(provider)
        cli_model = config.get("model")
        # Use CLI --model only if it looks like an OpenAI model (not a HF repo ID)
        if cli_model and "/" not in cli_model:
            api_model = cli_model
        else:
            api_model = default_model
        print(f"Provider:  openai")
        print(f"Model:     {api_model}")
    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported: 'local', 'openai'"
        )

    print(f"Glossary:  {glossary_path} ({len(glossary)} terms)")
    print(f"Files:     {len(files)}")
    print()

    for i, file in enumerate(files):
        file_output_dir = output_dir or str(file.parent)
        stem = file.stem
        if stem.endswith(" eng"):
            out_name = stem[:-4] + " translation.txt"
        else:
            out_name = stem + " translation.txt"
        out_path = Path(file_output_dir) / out_name

        if out_path.exists():
            print(f"[{i + 1}/{len(files)}] Skipping (exists): {file.name}")
            continue

        print(f"[{i + 1}/{len(files)}] Translating: {file.name}")
        t0 = time.time()
        translate_file(
            file, out_path, glossary, config,
            tokenizer=tokenizer, model=model,
            api_client=api_client, api_model=api_model,
        )
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate eng.txt lecture transcriptions to Japanese"
    )
    parser.add_argument(
        "input",
        help="Path to eng.txt file, directory, or glob pattern",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Translation provider: local (default), openai",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/ID (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Config file path (default: config.json)",
    )
    args = parser.parse_args()

    config = load_translation_config(
        args.config,
        provider=args.provider,
        model=args.model,
        output_dir=args.output_dir,
    )

    files = resolve_input_files(args.input)
    if not files:
        print(f"No eng.txt files found for: {args.input}")
        return

    translate_files(files, config)


if __name__ == "__main__":
    main()
