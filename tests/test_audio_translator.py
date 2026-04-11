"""Tests for audio_translator.py."""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audio_translator import (
    parse_eng_txt,
    load_glossary,
    build_prompt,
    parse_translations,
    write_translation_txt,
    chunk_sentences,
    merge_chunk_translations,
    create_api_client,
    load_translation_config,
    resolve_input_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_eng_txt(tmp_path: Path) -> Path:
    content = (
        "[00:01.340 --> 00:11.220] Alright, so today we're going to discuss sintering.\n"
        "[00:11.900 --> 00:19.100] It's convenient to break up sintering into three stages.\n"
        "[00:19.360 --> 00:24.440] They're basically named initial, middle, and final.\n"
    )
    p = tmp_path / "Lecture 01 - Module 1 eng.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_glossary_tsv(tmp_path: Path) -> Path:
    content = "en\tja\nsintering\t焼結\ngrain boundary\t粒界\n"
    p = tmp_path / "glossary.tsv"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# test_parse_eng_txt
# ---------------------------------------------------------------------------

def test_parse_eng_txt(sample_eng_txt: Path) -> None:
    result = parse_eng_txt(sample_eng_txt)
    assert len(result) == 3
    assert result[0]["timestamp"] == "00:01.340 --> 00:11.220"
    assert "sintering" in result[0]["text"]
    assert result[0]["line"].startswith("[")


def test_parse_eng_txt_skips_non_timestamp_lines(tmp_path: Path) -> None:
    content = (
        "[00:01.000 --> 00:02.000] Hello.\n"
        "This line has no timestamp.\n"
        "\n"
        "[00:03.000 --> 00:04.000] World.\n"
    )
    p = tmp_path / "test eng.txt"
    p.write_text(content, encoding="utf-8")
    result = parse_eng_txt(p)
    assert len(result) == 2


def test_parse_eng_txt_real_file() -> None:
    real_path = Path("_text/__legacy/MSE403/Lecture 36 - Module 1 eng.txt")
    if real_path.exists():
        result = parse_eng_txt(real_path)
        assert len(result) > 10


# ---------------------------------------------------------------------------
# test_load_glossary
# ---------------------------------------------------------------------------

def test_load_glossary(sample_glossary_tsv: Path) -> None:
    glossary = load_glossary(sample_glossary_tsv)
    assert glossary["sintering"] == "焼結"
    assert glossary["grain boundary"] == "粒界"
    assert len(glossary) == 2


# ---------------------------------------------------------------------------
# test_build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt() -> None:
    texts = ["Hello world.", "Sintering is important."]
    glossary = {"sintering": "焼結", "grain boundary": "粒界"}
    system, user = build_prompt(texts, glossary)
    assert "用語集" in system
    assert "焼結" in system
    assert "1: Hello world." in user
    assert "2: Sintering is important." in user


# ---------------------------------------------------------------------------
# test_parse_translations
# ---------------------------------------------------------------------------

def test_parse_translations() -> None:
    response = "1: こんにちは。\n2: 焼結は重要です。\n"
    result = parse_translations(response, 2)
    assert result == ["こんにちは。", "焼結は重要です。"]


def test_parse_translations_count_mismatch() -> None:
    response = "1: こんにちは。\n"
    with pytest.raises(ValueError, match="missing="):
        parse_translations(response, 2)


def test_parse_translations_non_sequential_indices() -> None:
    response = "2: テスト二。\n3: テスト三。\n"
    with pytest.raises(ValueError, match="missing="):
        parse_translations(response, 2)


def test_parse_translations_duplicate_indices() -> None:
    response = "1: テストA。\n1: テストB。\n2: テストC。\n"
    with pytest.raises(ValueError, match="Duplicate"):
        parse_translations(response, 2)


def test_parse_translations_strips_think_block() -> None:
    response = (
        "<think>\nLet me think about this...\n</think>\n"
        "1: こんにちは。\n2: さようなら。\n"
    )
    result = parse_translations(response, 2)
    assert result == ["こんにちは。", "さようなら。"]


def test_parse_translations_with_extra_whitespace() -> None:
    response = "\n  1: テスト一。\n  2: テスト二。  \n\n"
    result = parse_translations(response, 2)
    assert result == ["テスト一。", "テスト二。"]


# ---------------------------------------------------------------------------
# test_write_translation_txt
# ---------------------------------------------------------------------------

def test_write_translation_txt(tmp_path: Path) -> None:
    sentences = [
        {"line": "[00:01.000 --> 00:02.000] Hello.", "timestamp": "00:01.000 --> 00:02.000", "text": "Hello."},
        {"line": "[00:03.000 --> 00:04.000] World.", "timestamp": "00:03.000 --> 00:04.000", "text": "World."},
    ]
    translations = ["こんにちは。", "世界。"]
    out = tmp_path / "translation.txt"
    write_translation_txt(sentences, translations, out)

    content = out.read_text(encoding="utf-8")
    lines = content.splitlines()

    # English line + Japanese line + empty line, repeated
    assert lines[0] == "[00:01.000 --> 00:02.000] Hello."
    assert lines[1] == "こんにちは。"
    assert lines[2] == ""
    assert lines[3] == "[00:03.000 --> 00:04.000] World."
    assert lines[4] == "世界。"
    assert lines[5] == ""
    # Full english block at end
    assert lines[6].startswith('" ')
    assert "Hello." in lines[6]
    assert "World." in lines[6]


# ---------------------------------------------------------------------------
# test_create_api_client
# ---------------------------------------------------------------------------

def test_create_api_client_missing_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            create_api_client("openai")


def test_create_api_client_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_api_client("invalid_provider")


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_create_api_client_openai() -> None:
    client, default_model = create_api_client("openai")
    assert default_model == "gpt-4.1"
    assert client is not None


# ---------------------------------------------------------------------------
# test_chunk_translation
# ---------------------------------------------------------------------------

def test_chunk_sentences_invalid_max() -> None:
    with pytest.raises(ValueError, match="max_sentences_per_chunk must be >= 1"):
        chunk_sentences(["a", "b"], max_sentences=0)


def test_chunk_sentences_small_input() -> None:
    texts = [f"Sentence {i}." for i in range(10)]
    chunks = chunk_sentences(texts, max_sentences=100)
    assert len(chunks) == 1
    assert chunks[0] == (0, 10)


def test_chunk_sentences_large_input() -> None:
    texts = [f"Sentence {i}." for i in range(200)]
    chunks = chunk_sentences(texts, max_sentences=100)
    assert len(chunks) >= 2
    # All sentences covered
    covered = set()
    for start, end in chunks:
        for i in range(start, end):
            covered.add(i)
    assert covered == set(range(200))


def test_chunk_sentences_overlap() -> None:
    texts = [f"Sentence {i}." for i in range(30)]
    chunks = chunk_sentences(texts, max_sentences=10)
    # Should have overlapping chunks (OVERLAP_SENTENCES=5, max=10 -> overlap works)
    assert len(chunks) >= 3
    # Check that overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        _, end1 = chunks[i]
        start2, _ = chunks[i + 1]
        assert start2 < end1, f"Chunks {i} and {i+1} should overlap: {chunks}"


def test_chunk_sentences_with_tokenizer() -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 1000
    mock_tokenizer.encode.return_value = list(range(50))  # 50 tokens per sentence

    texts = [f"Sentence {i}." for i in range(200)]
    # Budget = 1000 * 0.4 = 400 tokens. 50 tokens/sentence -> 8 sentences/chunk
    chunks = chunk_sentences(texts, max_sentences=100, tokenizer=mock_tokenizer)
    assert len(chunks) > 2  # Should be split by token budget


def test_merge_chunk_translations() -> None:
    chunks = [(0, 10), (5, 15)]
    chunk_results = [
        [f"Translation A{i}" for i in range(10)],
        [f"Translation B{i}" for i in range(10)],
    ]
    merged = merge_chunk_translations(chunks, chunk_results, 15)
    # First 5 from chunk A, rest from chunk B (overlap uses later chunk)
    assert merged[0] == "Translation A0"
    assert merged[4] == "Translation A4"
    assert merged[5] == "Translation B0"  # Overlap: chunk B wins
    assert merged[14] == "Translation B9"


# ---------------------------------------------------------------------------
# test_error_handling
# ---------------------------------------------------------------------------

def test_error_handling_missing_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            create_api_client("openai")


def test_error_handling_line_count_mismatch() -> None:
    with pytest.raises(ValueError, match="missing="):
        parse_translations("1: A\n2: B\n", 3)


# ---------------------------------------------------------------------------
# test_config
# ---------------------------------------------------------------------------

def test_load_translation_config_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text('{"model": {}}', encoding="utf-8")
    config = load_translation_config(str(config_file))
    assert config["provider"] == "local"
    assert config["model"] == "Qwen/Qwen3.5-9B"
    assert config["max_sentences_per_chunk"] == 100


def test_load_translation_config_cli_override(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text('{"translation": {"provider": "local"}}', encoding="utf-8")
    config = load_translation_config(str(config_file), provider="openai")
    assert config["provider"] == "openai"


# ---------------------------------------------------------------------------
# test_resolve_input_files
# ---------------------------------------------------------------------------

def test_resolve_input_files_single(sample_eng_txt: Path) -> None:
    result = resolve_input_files(str(sample_eng_txt))
    assert len(result) == 1
    assert result[0] == sample_eng_txt


def test_resolve_input_files_directory(tmp_path: Path) -> None:
    (tmp_path / "Lecture 01 eng.txt").write_text("[00:00.000 --> 00:01.000] Hi.\n")
    (tmp_path / "Lecture 02 eng.txt").write_text("[00:00.000 --> 00:01.000] Bye.\n")
    (tmp_path / "other.txt").write_text("not an eng file\n")
    result = resolve_input_files(str(tmp_path))
    assert len(result) == 2
