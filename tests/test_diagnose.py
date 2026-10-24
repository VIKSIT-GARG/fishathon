"""Diagnostic CLI logic without live provider requests."""
import asyncio
import json
import os

from backend import main
import diagnose


def test_diagnostic_covers_text_image_writer_and_restores_fish_key(monkeypatch, tmp_path, capsys):
    image = tmp_path / 'comic.png'
    image.write_bytes(b'placeholder passed to mock analyzer')
    source = main.Script.model_validate({'context': 'A dog and a person.', 'characters': [{'id': 'dog', 'name': 'Dog'}], 'lines': []})
    story = main.Script.model_validate({'characters': [{'id': 'dog', 'name': 'Dog'}], 'lines': [{'id': 'l1', 'panel': 1, 'character_id': 'dog', 'text': 'Hello.'}]})
    monkeypatch.setenv('GEMINI_API_KEY', 'hidden-gemini-value')
    monkeypatch.setenv('FISH_API_KEY', 'hidden-fish-value')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-3.8-flash')
    calls = []
    async def generate(*args, **kwargs):
        calls.append('text')
        return json.dumps({'status': 'ok'})
    async def analyze(upload):
        calls.append('image')
        assert await upload.read() == image.read_bytes()
        return source
    async def direct(request):
        calls.append('story')
        assert not os.getenv('FISH_API_KEY')
        return story
    monkeypatch.setattr(main, 'gemini_generate', generate)
    monkeypatch.setattr(main, 'analyze', analyze)
    monkeypatch.setattr(main, 'direct', direct)
    assert asyncio.run(diagnose.diagnose(image, True)) == 0
    assert calls == ['text', 'image', 'story']
    assert os.environ['FISH_API_KEY'] == 'hidden-fish-value'
    output = capsys.readouterr().out
    assert 'hidden-gemini-value' not in output
    assert 'hidden-fish-value' not in output


def test_diagnostic_cli_missing_key_exits_with_actionable_safe_message(monkeypatch, capsys):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setattr('sys.argv', ['diagnose.py'])
    assert diagnose.cli() == 1
    assert 'GEMINI_API_KEY' in capsys.readouterr().out
