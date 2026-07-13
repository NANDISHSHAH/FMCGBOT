import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.intent import IntentRouter, normalize_entities
from src.core.memory import SessionMemory


def test_greeting_detected():
    router = IntentRouter()
    mem = SessionMemory("t1")
    result = router.classify("hello", mem)
    assert result.turn_type == "greeting"


def test_capability_detected():
    router = IntentRouter()
    mem = SessionMemory("t2")
    result = router.classify("what can you do?", mem)
    assert result.turn_type == "capability"


def test_out_of_scope_detected():
    router = IntentRouter()
    mem = SessionMemory("t3")
    result = router.classify("what is the capital of France", mem)
    assert result.turn_type == "out_of_scope"


def test_in_scope_question_resolves_brand():
    router = IntentRouter()
    mem = SessionMemory("t4")
    result = router.classify("how did NutriOat perform in North", mem)
    assert result.turn_type == "question"
    assert result.resolved_entities.get("brand") == "NutriOat"
    assert result.resolved_entities.get("region") == "North"


def test_typo_correction():
    entities = normalize_entities("how is nutrioaat doing")
    assert entities.get("brand") == "NutriOat"


def test_ambiguous_followup_without_context():
    router = IntentRouter()
    mem = SessionMemory("t5")
    result = router.classify("what about that", mem)
    assert result.turn_type == "ambiguous"
