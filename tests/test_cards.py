import json

import pytest

from cardgen.models import (
    Card,
    CardValidationError,
    derive_seed,
    is_safe_id,
    load_cards,
    output_filename,
    resolve_seed,
    slugify,
    validate_cards,
)


def errors_of(raw):
    with pytest.raises(CardValidationError) as info:
        validate_cards(raw)
    return info.value.errors


def test_valid_cards_load(project):
    cards = load_cards(project / "cards.json").cards
    assert [c.id for c in cards] == ["village_archer", "brave_peasant", "court_jester"]
    assert cards[0].seed == 1001 and cards[2].seed is None


def test_example_cards_file_is_valid():
    from pathlib import Path
    cards = load_cards(Path(__file__).resolve().parents[1] / "cards.json").cards
    assert len(cards) == 3


def test_name_defaults_to_id_and_prompt_whitespace_is_normalised():
    card = validate_cards([{"id": "a1", "prompt": "  two\n  lines "}]).cards[0]
    assert card.name == "a1"
    assert card.prompt == "two lines"


def test_malformed_json(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text('[{"id": "a", "prompt": "x",}]')
    with pytest.raises(CardValidationError) as info:
        load_cards(path)
    assert "malformed JSON at line 1" in info.value.errors[0]


def test_missing_file(tmp_path):
    with pytest.raises(CardValidationError, match="problem"):
        load_cards(tmp_path / "missing.json")


@pytest.mark.parametrize("raw, message", [
    ({}, "top level must be a JSON list"),
    ([], "empty"),
])
def test_top_level_shape(raw, message):
    assert message in errors_of(raw)[0]


def test_missing_fields_reported_with_position():
    errs = errors_of([{"name": "Sir Bob", "prompt": "x"}, {"id": "b"}])
    assert any("card #1" in e and "missing 'id'" in e and "sir_bob" in e for e in errs)
    assert any("card #2" in e and "'prompt'" in e for e in errs)


def test_duplicate_ids_case_insensitive():
    errs = errors_of([{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}])
    assert any("duplicate id 'a'" in e and "card #1" in e for e in errs)


@pytest.mark.parametrize("bad_id", ["Bad Id", "../etc", "a/b", "-start", "UPPER", "x" * 65, "", 5, "café"])
def test_unsafe_ids_rejected(bad_id):
    errs = errors_of([{"id": bad_id, "prompt": "x"}])
    assert errs


def test_invalid_cards_are_never_silently_dropped():
    raw = [{"id": "good", "prompt": "x"}, {"id": "bad id", "prompt": "y"}, {"id": "good2", "prompt": ""}]
    errs = errors_of(raw)
    assert len(errs) == 2


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "7", True])
def test_invalid_seed(seed):
    assert any("seed" in e for e in errors_of([{"id": "a", "prompt": "x", "seed": seed}]))


def test_overrides_validated():
    ok = validate_cards([{"id": "a", "prompt": "x",
                          "overrides": {"steps": 30, "guidance_scale": 7, "negative_prompt": "dogs"}}])
    assert ok.cards[0].overrides["steps"] == 30
    errs = errors_of([{"id": "a", "prompt": "x", "overrides": {"steps": 0, "sampler": "x"}}])
    assert any("sampler" in e for e in errs) and any("steps" in e for e in errs)
    errs = errors_of([{"id": "a", "prompt": "x", "overrides": {"width": 512}}])
    assert any("both 'width' and 'height'" in e for e in errs)


def test_unknown_card_field_is_an_error():
    assert any("unknown field 'promt'" in e for e in errors_of([{"id": "a", "promt": "x"}]))


def test_slugify_and_filenames():
    assert slugify("Sir Gö! the  Brave") == "sir_go_the_brave"
    assert slugify("../../etc/passwd") == "etc_passwd"
    assert slugify("!!!") == ""
    assert is_safe_id(slugify("Court Jester (v2)"))
    assert output_filename("court_jester") == "court_jester.png"
    with pytest.raises(ValueError):
        output_filename("../x")


def test_seeds_stable_under_reorder_and_additions():
    a = derive_seed(42, "village_archer")
    assert a == derive_seed(42, "village_archer")
    assert a != derive_seed(43, "village_archer")
    assert a != derive_seed(42, "court_jester")
    assert 0 <= a < 2**32
    explicit = Card(id="x", name="x", prompt="p", seed=7)
    assert resolve_seed(explicit, 42) == 7 == resolve_seed(explicit, 99)
    # Seeds depend only on the card, not its position.
    one = validate_cards([{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]).cards
    two = validate_cards([{"id": "new", "prompt": "z"}, {"id": "b", "prompt": "y"},
                          {"id": "a", "prompt": "x"}]).cards
    seeds1 = {c.id: resolve_seed(c, 5) for c in one}
    seeds2 = {c.id: resolve_seed(c, 5) for c in two}
    assert seeds1["a"] == seeds2["a"] and seeds1["b"] == seeds2["b"]


def test_csv_import(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text("id,name,prompt,seed,steps\n"
                    "archer,Archer,An archer,11,\n"
                    "jester,Jester,\"A jester, juggling\",,30\n", encoding="utf-8")
    cards = load_cards(path).cards
    assert cards[0].seed == 11 and cards[0].overrides == {}
    assert cards[1].prompt == "A jester, juggling" and cards[1].seed is None
    assert cards[1].overrides == {"steps": 30}


def test_csv_errors_reported(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text("id,prompt,seed\narcher,An archer,abc\narcher,dup,\n,no id,\n")
    with pytest.raises(CardValidationError) as info:
        load_cards(path)
    errs = info.value.errors
    assert any("seed" in e for e in errs)
    assert any("duplicate" in e for e in errs)
    assert any("missing 'id'" in e for e in errs)


def test_csv_needs_header(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text("archer,An archer\n")
    with pytest.raises(CardValidationError):
        load_cards(path)
