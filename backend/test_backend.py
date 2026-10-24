"""Offline regression tests: no live Gemini/Fish API key or billable calls needed."""
import asyncio
import io
import wave

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from backend.main import Script, cast_voices, emotion_text, read_wav, search_voices


def script(**extra):
    data = {'title': 'Cookie', 'characters': [{'id': 'c1', 'name': 'Grandpa', 'voice_query': 'gravelly old man'}], 'lines': [{'id': 'l1', 'panel': 1, 'character_id': 'c1', 'text': 'My cookie!', 'emotion': 'angry'}]}
    data.update(extra)
    return Script.model_validate(data)


def test_script_rejects_unknown_character():
    with pytest.raises(ValidationError):
        script(lines=[{'id': 'l1', 'panel': 1, 'character_id': 'ghost', 'text': 'Oops'}])


def test_script_rejects_duplicate_lines_and_wrong_panel_order():
    with pytest.raises(ValidationError):
        script(lines=[{'id': 'l1', 'panel': 1, 'character_id': 'c1', 'text': 'a'}, {'id': 'l1', 'panel': 1, 'character_id': 'c1', 'text': 'b'}])
    with pytest.raises(ValidationError):
        script(lines=[{'id': 'l1', 'panel': 2, 'character_id': 'c1', 'text': 'a'}, {'id': 'l2', 'panel': 1, 'character_id': 'c1', 'text': 'b'}])


def test_emotion_uses_model_syntax_and_removes_speaker_tokens():
    assert emotion_text('Hi <|speaker:9|>there', 'happy', 's2-pro') == '[happy] Hi there'
    assert emotion_text('Hi', 'angry', 's1') == '(angry) Hi'


def test_wav_parser_counts_frames_and_rejects_fake_audio():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b'\0\0' * 2400)
    assert read_wav(buffer.getvalue()) == (b'\0\0' * 2400, 2400)
    with pytest.raises(HTTPException):
        read_wav(b'not real audio')


def test_fish_search_and_cast_use_documented_endpoint(monkeypatch):
    monkeypatch.setenv('FISH_API_KEY', 'test-key')
    calls = []
    def handler(request):
        calls.append(request)
        assert request.url.path == '/model'
        assert request.headers['authorization'] == 'Bearer test-key'
        return httpx.Response(200, json={'items': [{'_id': 'voice_a', 'title': 'Gravelly old man', 'state': 'trained', 'tags': ['old'], 'description': 'low voice'}, {'_id': 'bad', 'title': 'Unavailable', 'state': 'failed'}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            voices = await search_voices('old', client)
            assert len(voices) == 1
            result = await cast_voices(script(), client)
            assert result.characters[0].voice_id == 'voice_a'
            assert result.characters[0].voice_name == 'Gravelly old man'
    asyncio.run(run())
    assert calls[0].url.params['title'] == 'old'


def test_no_match_warns_without_inventing_voice(monkeypatch):
    monkeypatch.setenv('FISH_API_KEY', 'test-key')
    monkeypatch.delenv('FISH_DEFAULT_VOICE_ID', raising=False)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'items': []}))) as client:
            result = await cast_voices(script(), client)
            assert result.characters[0].voice_id is None
            assert any('No voice match' in message for message in result.warnings)
    asyncio.run(run())


def test_blank_voice_id_normalizes_to_unset():
    result = script(characters=[{'id': 'c1', 'name': 'Grandpa', 'voice_id': '  '}])
    assert result.characters[0].voice_id is None


def test_parallel_cast_search_and_distinct_selection(monkeypatch):
    monkeypatch.setenv('FISH_API_KEY', 'test-key')
    active = 0
    peak = 0
    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json={'items': [{'_id': f'voice_{i}', 'title': 'Old voice', 'state': 'trained'} for i in range(3)]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await cast_voices(script(characters=[{'id': f'c{i}', 'name': f'Person {i}', 'voice_query': 'old'} for i in range(1, 4)]), client)
            assert len({c.voice_id for c in result.characters}) == 3
    asyncio.run(run())
    assert peak == 3
