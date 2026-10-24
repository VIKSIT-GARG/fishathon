"""Black-box API checks with deterministic Gemini/Fish HTTP responses."""
import base64
import io
import json
import wave

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import main

REAL_ASYNC_CLIENT = httpx.AsyncClient


def gemini_response(value):
    text = value if isinstance(value, str) else json.dumps(value)
    return {'candidates': [{'content': {'parts': [{'text': text}]}, 'finishReason': 'STOP'}]}


def wav_bytes(frames=2400):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b'\x01\x00' * frames)
    return buf.getvalue()


def comic_bytes():
    buf = io.BytesIO()
    Image.new('RGB', (32, 16), 'white').save(buf, format='PNG')
    return buf.getvalue()


def script():
    return {'title': 'Test play', 'context': 'An old man encounters a robot at his door.', 'tts_model': 's2.1-pro-free', 'render_mode': 'segments', 'characters': [
        {'id': 'c1', 'name': 'Old man', 'kind': 'human', 'observed_traits': 'Grey beard and walking stick', 'personality': 'Grumpy but kind', 'voice_direction': 'Gravelly and deliberate', 'voice_query': 'gravelly', 'voice_id': 'old'},
        {'id': 'c2', 'name': 'Robot', 'kind': 'robot', 'observed_traits': 'A box-shaped robot with an antenna', 'personality': 'Curious and cheerful', 'voice_direction': 'Bright and precise', 'voice_query': 'robot', 'voice_id': 'robot'},
    ], 'lines': [
        {'id': 'l1', 'panel': 1, 'character_id': 'c1', 'text': 'Who is there?', 'emotion': 'angry', 'pause_after_ms': 160},
        {'id': 'l2', 'panel': 1, 'character_id': 'c2', 'text': 'Just me.', 'emotion': 'calm', 'pause_after_ms': 350},
        {'id': 'l3', 'panel': 2, 'character_id': 'c1', 'text': 'Oh!', 'emotion': 'surprised'},
    ]}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv('FISH_API_KEY', 'unit-test-secret')
    monkeypatch.setenv('FISH_MODEL', 's2.1-pro-free')
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-gemini-secret')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-3.8-flash')
    monkeypatch.setattr(main, 'OUTPUTS', tmp_path)
    return TestClient(main.app), tmp_path


def mock_http(monkeypatch, handler):
    monkeypatch.setattr(main.httpx, 'AsyncClient', lambda **kwargs: REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs))


def test_analyze_decodes_photo_and_uses_minimal_gemini_json_request(setup, monkeypatch):
    client, _ = setup
    captured = []
    response_script = script()
    def handler(request):
        captured.append(request)
        assert request.url.host == 'generativelanguage.googleapis.com'
        assert request.url.path == '/v1beta/models/gemini-3.8-flash:generateContent'
        payload = json.loads(request.content)
        assert request.headers['x-goog-api-key'] == 'unit-gemini-secret'
        assert 'key' not in request.url.params
        assert 'unit-gemini-secret' not in request.url.path
        assert payload['generationConfig']['responseMimeType'] == 'application/json'
        assert payload['generationConfig'] == {'responseMimeType': 'application/json'}
        assert 'responseJsonSchema' not in json.dumps(payload)
        assert 'thinkingConfig' not in json.dumps(payload)
        guidance = ' '.join(part.get('text', '') for part in payload.get('systemInstruction', {}).get('parts', []))
        assert 'characters' in guidance and 'properties' in guidance
        parts = payload['contents'][0]['parts']
        image = next(part['inlineData'] for part in parts if 'inlineData' in part)
        assert image['mimeType'] == 'image/jpeg'
        data = base64.b64decode(image['data'])
        assert Image.open(io.BytesIO(data)).format == 'JPEG'
        return httpx.Response(200, json=gemini_response(response_script))
    mock_http(monkeypatch, handler)
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 200, response.text
    assert response.json()['lines'][1]['character_id'] == 'c2'
    assert len(captured) == 1


def test_render_wav_timeline_and_emotional_requests(setup, monkeypatch):
    client, output = setup
    requests = []
    def handler(request):
        assert request.url.path == '/v1/tts'
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers['authorization'] == 'Bearer unit-test-secret'
        assert request.headers['model'] == 's2.1-pro-free'
        assert payload['format'] == 'wav'
        assert payload['sample_rate'] == 24000
        return httpx.Response(200, content=wav_bytes())
    mock_http(monkeypatch, handler)
    response = client.post('/api/render', json=script())
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(requests) == 3
    assert {r['text'] for r in requests} == {'[Gravelly and deliberate; angry] Who is there?', '[Bright and precise; calm] Just me.', '[Gravelly and deliberate; surprised] Oh!'}
    assert [s['start_ms'] for s in body['segments']] == [0, 260, 710]
    assert [s['end_ms'] for s in body['segments']] == [100, 360, 810]
    target = output / body['audio_url'].split('/outputs/', 1)[1]
    with wave.open(str(target)) as stream:
        assert stream.getnframes() == 19440
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 24000
    assert len(list(target.parent.glob('*.wav'))) == 4


@pytest.mark.parametrize('status,expected', [(401, 'API key was rejected'), (402, 'insufficient credits'), (429, 'rate limit')])
def test_provider_error_is_sanitized_and_partial_outputs_removed(setup, monkeypatch, status, expected):
    client, output = setup
    mock_http(monkeypatch, lambda request: httpx.Response(status, json={'message': 'sensitive upstream unit-test-secret'}))
    response = client.post('/api/render', json=script())
    assert response.status_code == 502
    assert expected in response.json()['detail']
    assert 'unit-test-secret' not in response.text
    assert not list(output.iterdir())


def test_missing_key_is_actionable(setup, monkeypatch):
    client, _ = setup
    monkeypatch.delenv('FISH_API_KEY')
    response = client.post('/api/render', json=script())
    assert response.status_code == 503
    assert 'FISH_API_KEY' in response.json()['detail']


def test_rejects_invalid_photo_without_upstream_call(setup, monkeypatch):
    client, _ = setup
    def handler(request):
        pytest.fail('Invalid image must not reach upstream')
    mock_http(monkeypatch, handler)
    response = client.post('/api/analyze', files={'image': ('bad.png', b'not image data', 'image/png')})
    assert response.status_code == 422


def test_rejects_oversized_upload(setup):
    client, _ = setup
    response = client.post('/api/analyze', files={'image': ('huge.png', b'0' * (main.MAX_UPLOAD + 1), 'image/png')})
    assert response.status_code == 413


def test_malformed_gemini_response_is_actionable(setup, monkeypatch):
    client, _ = setup
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=gemini_response('{bad json')))
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 502
    assert 'invalid comic context' in response.json()['detail']


def test_render_rejects_invalid_wav_and_removes_output(setup, monkeypatch):
    client, output = setup
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=b'invalid wav'))
    response = client.post('/api/render', json=script())
    assert response.status_code == 502
    assert 'invalid audio' in response.json()['detail']
    assert not list(output.iterdir())


def test_direct_casts_distinct_voices_from_fish_search(setup, monkeypatch):
    client, _ = setup
    uncased = script()
    for character in uncased['characters']:
        character['voice_id'] = None
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path.endswith(':generateContent'):
            return httpx.Response(200, json=gemini_response(uncased))
        assert request.url.path == '/model'
        assert request.headers['authorization'] == 'Bearer unit-test-secret'
        return httpx.Response(200, json={'items': [
            {'_id': 'old', 'title': 'Gravelly old man', 'description': 'gravelly narration', 'tags': ['gravelly']},
            {'_id': 'robot', 'title': 'Robot voice', 'description': 'robot effects', 'tags': ['robot']},
        ]})
    mock_http(monkeypatch, handler)
    response = client.post('/api/direct', json={'source': uncased, 'model': 's2.1-pro-free'})
    assert response.status_code == 200, response.text
    assert [c['voice_id'] for c in response.json()['characters']] == ['old', 'robot']
    assert len(seen) == 3


def test_analyze_without_key_returns_observations_not_generated_personality(setup, monkeypatch):
    client, _ = setup
    monkeypatch.delenv('FISH_API_KEY')
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=gemini_response(script())))
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 200
    assert response.json()['generated'] is False
    assert all(not c['personality'] and not c['voice_direction'] for c in response.json()['characters'])


@pytest.mark.parametrize('model', ['s2.1-pro-free', 'drama-3'])
def test_native_scene_uses_one_fish_request_and_stable_voice_array(setup, monkeypatch, model):
    client, output = setup
    story = script()
    story.update(tts_model=model, render_mode='scene', generated=True)
    calls = []
    def handler(request):
        assert request.url.path == '/v1/tts'
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.headers['model'] == model
        assert payload['reference_id'] == ['old', 'robot']
        assert payload['text'].count('<|speaker:0|>') == 2
        assert payload['text'].count('<|speaker:1|>') == 1
        for spoken in ('Who is there?', 'Just me.', 'Oh!'):
            assert spoken in payload['text']
        return httpx.Response(200, content=wav_bytes())
    mock_http(monkeypatch, handler)
    response = client.post('/api/render', json=story)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(calls) == 1
    assert data['model'] == model
    assert data['segments'] == []
    assert data['timing_available'] is False
    assert data['duration_ms'] == 100
    assert data['prompt_text'] == calls[0]['text']
    assert len(list(output.rglob('*.wav'))) == 1


@pytest.mark.parametrize('model', ['s1', 's2-pro', 's2.1-pro', 'unknown'])
def test_model_selection_rejects_unsupported_values(setup, monkeypatch, model):
    client, _ = setup
    def handler(request):
        pytest.fail('Unsupported model must not cause provider request')
    mock_http(monkeypatch, handler)
    story = script()
    story['tts_model'] = model
    response = client.post('/api/render', json=story)
    assert response.status_code == 422


def test_preview_requires_no_key_or_provider_and_strips_speaker_injection(setup, monkeypatch):
    client, _ = setup
    monkeypatch.delenv('FISH_API_KEY')
    def handler(request):
        pytest.fail('Preview must not call any provider')
    mock_http(monkeypatch, handler)
    story = script()
    story['render_mode'] = 'scene'
    story['lines'][0]['text'] = '<|speaker:99|>Who is there?'
    story['lines'][0]['direction'] = 'quietly [shout] <|speaker:98|>'
    story['characters'][0]['voice_direction'] = 'gentle <|speaker:97|>'
    response = client.post('/api/preview', json=story)
    assert response.status_code == 200, response.text
    data = response.json()
    prompt = data['text']
    assert '<|speaker:99|>' not in prompt
    assert '<|speaker:98|>' not in prompt
    assert '<|speaker:97|>' not in prompt
    assert prompt.count('<|speaker:0|>') == 2
    assert prompt.count('<|speaker:1|>') == 1
    assert 'Who is there?' in prompt
    assert 'angry]' in prompt


def test_direct_uses_text_only_gemini_and_preserves_observed_cast(setup, monkeypatch):
    client, _ = setup
    source = script()
    generated = script()
    generated['characters'][0]['personality'] = 'A suspicious guardian secretly yearning for companionship'
    generated['lines'][0]['text'] = 'Another visitor at midnight? This had better be good.'
    generated['lines'][0]['direction'] = 'under the breath, then suspicious'
    generated['lines'][0]['pause_after_ms'] = 550
    captured = []
    def handler(request):
        assert request.url.host == 'generativelanguage.googleapis.com'
        assert request.url.path == '/v1beta/models/gemini-3.8-flash:generateContent'
        payload = json.loads(request.content)
        captured.append(payload)
        assert request.headers['x-goog-api-key'] == 'unit-gemini-secret'
        assert 'key' not in request.url.params
        assert payload['generationConfig'] == {'responseMimeType': 'application/json'}
        assert 'responseJsonSchema' not in json.dumps(payload)
        assert 'thinkingConfig' not in json.dumps(payload)
        guidance = ' '.join(part.get('text', '') for part in payload.get('systemInstruction', {}).get('parts', []))
        assert 'characters' in guidance and 'properties' in guidance
        assert all('inlineData' not in part for content in payload['contents'] for part in content['parts'])
        return httpx.Response(200, json=gemini_response(generated))
    mock_http(monkeypatch, handler)
    response = client.post('/api/direct', json={'source': source, 'model': 'drama-3', 'genre': 'mystery', 'instructions': 'Make the robot funny.', 'target_seconds': 35})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['generated'] is True
    assert data['tts_model'] == 'drama-3'
    assert [c['id'] for c in data['characters']] == ['c1', 'c2']
    assert data['characters'][0]['observed_traits'] == source['characters'][0]['observed_traits']
    assert data['lines'][0]['text'] != source['lines'][0]['text']
    assert data['lines'][0]['direction']
    assert data['lines'][0]['pause_after_ms'] == 550
    assert len(captured) == 1


def test_direct_rejects_new_character_from_writer(setup, monkeypatch):
    client, _ = setup
    generated = script()
    generated['characters'].append({'id': 'invented', 'name': 'Narrator'})
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=gemini_response(generated)))
    response = client.post('/api/direct', json={'source': script(), 'model': 's2.1-pro-free'})
    assert response.status_code == 502
    assert 'changed the source cast' in response.json()['detail']


def test_direct_restores_dog_and_ambiguous_person_observations(setup, monkeypatch):
    client, _ = setup
    source = script()
    source['characters'][0].update(name='Person', kind='human', description='A simple stick figure', observed_traits='A stick figure with a straight mouth')
    source['characters'][1].update(name='Dog', kind='dog', description='A four-legged dog', observed_traits='Four legs and a tail')
    generated = json.loads(json.dumps(source))
    generated['characters'][0].update(name='Little girl', kind='human', description='A young child', observed_traits='A little girl', personality='Determined and inventive')
    generated['characters'][1].update(name='Boy', kind='human', description='A small boy', observed_traits='Young boy', personality='Affectionate and manipulative')
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=gemini_response(generated)))
    response = client.post('/api/direct', json={'source': source, 'model': 'drama-3'})
    assert response.status_code == 200, response.text
    characters = response.json()['characters']
    for original, directed in zip(source['characters'], characters):
        for field in ('id', 'name', 'kind', 'description', 'observed_traits', 'voice_id'):
            assert directed[field] == original[field]
    assert characters[0]['personality'] == 'Determined and inventive'
    assert characters[1]['kind'] == 'dog'


def test_preview_stable_speaker_maps_follows_cast_not_first_line(setup, monkeypatch):
    client, _ = setup
    monkeypatch.delenv('FISH_API_KEY')
    story = script()
    story['lines'][0]['character_id'] = 'c2'
    response = client.post('/api/preview', json=story)
    assert response.status_code == 200
    data = response.json()
    assert [(speaker['speaker'], speaker['character_id']) for speaker in data['speaker_map']] == [(0, 'c1'), (1, 'c2')]
    assert data['reference_id'] == ['old', 'robot']
    assert data['text'].startswith('<|speaker:1|>')
    assert data['missing_voices'] == []


def test_scene_does_not_invent_per_line_timestamps(setup, monkeypatch):
    client, _ = setup
    story = script()
    story['render_mode'] = 'scene'
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=wav_bytes(4800)))
    response = client.post('/api/render', json=story)
    assert response.status_code == 200
    assert response.json()['segments'] == []
    assert response.json()['timing_available'] is False
    assert response.json()['duration_ms'] == 200


def test_segments_preview_matches_exact_outgoing_performance_requests(setup, monkeypatch):
    client, _ = setup
    story = script()
    story['lines'][0].update(direction='quietly, then sharply', pause_after_ms=550)
    story['lines'][1].update(direction='with playful hesitation', pause_after_ms=900)
    story['lines'][2].update(direction='a relieved gasp', pause_after_ms=0)
    preview = client.post('/api/preview', json=story)
    assert preview.status_code == 200, preview.text
    entries = preview.json()['segment_requests']
    assert len(entries) == 3
    assert [entry['reference_id'] for entry in entries] == ['old', 'robot', 'old']
    assert all(isinstance(entry['reference_id'], str) for entry in entries)
    assert [entry['pause_after_ms'] for entry in entries] == [550, 900, 0]
    assert entries[0]['text'] == '[Gravelly and deliberate; quietly, then sharply; angry] Who is there?'
    assert entries[1]['text'] == '[Bright and precise; with playful hesitation; calm] Just me.'
    assert all('<|speaker:' not in entry['text'] for entry in entries)
    outgoing = []
    def handler(request):
        assert request.url.path == '/v1/tts'
        outgoing.append(json.loads(request.content))
        return httpx.Response(200, content=wav_bytes())
    mock_http(monkeypatch, handler)
    rendered = client.post('/api/render', json=story)
    assert rendered.status_code == 200, rendered.text
    assert [(body['text'], body['reference_id']) for body in outgoing] == [(entry['text'], entry['reference_id']) for entry in entries]
    assert [segment['start_ms'] for segment in rendered.json()['segments']] == [0, 650, 1650]
    assert [segment['end_ms'] for segment in rendered.json()['segments']] == [100, 750, 1750]


@pytest.mark.parametrize('mode', ['scene', 'segments'])
def test_token_only_dialogue_rejected_before_creating_output_folder(setup, monkeypatch, mode):
    client, output = setup
    story = script()
    story['render_mode'] = mode
    story['lines'][0]['text'] = '<|speaker:99|>'
    def handler(request):
        pytest.fail('Empty spoken content must not reach Fish')
    mock_http(monkeypatch, handler)
    response = client.post('/api/render', json=story)
    assert response.status_code == 422, response.text
    assert 'empty after removing control tokens' in response.json()['detail']
    assert not list(output.iterdir())


@pytest.mark.parametrize('endpoint', ['analyze', 'direct'])
def test_missing_gemini_key_is_actionable_and_never_calls_provider(setup, monkeypatch, endpoint):
    client, _ = setup
    monkeypatch.delenv('GEMINI_API_KEY')
    def handler(request):
        pytest.fail('Missing Gemini key must not cause a provider request')
    mock_http(monkeypatch, handler)
    if endpoint == 'analyze':
        response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    else:
        response = client.post('/api/direct', json={'source': script(), 'model': 's2.1-pro-free'})
    assert response.status_code == 503, response.text
    assert 'GEMINI_API_KEY' in response.json()['detail']


@pytest.mark.parametrize('status', [400, 401, 403, 404, 429])
@pytest.mark.parametrize('endpoint', ['analyze', 'direct'])
def test_gemini_http_errors_are_actionable_and_never_leak_key(setup, monkeypatch, status, endpoint):
    client, _ = setup
    calls = []
    def handler(request):
        calls.append(request)
        assert request.url.host == 'generativelanguage.googleapis.com'
        assert request.headers['x-goog-api-key'] == 'unit-gemini-secret'
        return httpx.Response(status, json={'error': {'message': 'unit-gemini-secret sensitive-provider-internals'}})
    mock_http(monkeypatch, handler)
    if endpoint == 'analyze':
        response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    else:
        response = client.post('/api/direct', json={'source': script(), 'model': 's2.1-pro-free'})
    assert response.status_code == 502, response.text
    assert 'Gemini' in response.json()['detail']
    assert 'unit-gemini-secret' not in response.text
    assert 'sensitive-provider-internals' not in response.text
    assert len(calls) == 1


@pytest.mark.parametrize('body,expected', [
    ({'promptFeedback': {'blockReason': 'SAFETY'}}, 'block'),
    ({'candidates': [{'finishReason': 'SAFETY', 'content': {'parts': []}}]}, 'block'),
    ({'candidates': []}, 'empty'),
    ({'candidates': [{'finishReason': 'STOP', 'content': {'parts': []}}]}, 'empty'),
    ({'candidates': [{'finishReason': 'MAX_TOKENS', 'content': {'parts': [{'text': '{"title":"unfinished"'}]}}]}, 'truncat'),
])
def test_gemini_blocked_empty_and_truncated_are_actionable(setup, monkeypatch, body, expected):
    client, _ = setup
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=body))
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 502, response.text
    assert expected in response.json()['detail'].lower(), response.text
    assert 'unit-gemini-secret' not in response.text


def test_gemini_parts_text_is_combined_for_structured_output(setup, monkeypatch):
    client, _ = setup
    text = json.dumps(script())
    midpoint = len(text) // 2
    body = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text[:midpoint]}, {'text': text[midpoint:]}]}}]}
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=body))
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 200, response.text
    assert response.json()['title'] == 'Test play'


def test_health_reports_configuration_without_verifying_secret_or_calling_provider(setup, monkeypatch):
    client, _ = setup
    def handler(request):
        pytest.fail('Health must not pretend a configured key is verified by making generation calls')
    mock_http(monkeypatch, handler)
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.json()
    assert data['gemini_configured'] is True
    assert data['model'] == 'gemini-3.8-flash'
    assert data.get('gemini_verified', False) is False
    assert 'unit-gemini-secret' not in response.text
    assert 'unit-test-secret' not in response.text
    assert 'ollama' not in response.text.lower()
    monkeypatch.delenv('GEMINI_API_KEY')
    assert client.get('/api/health').json()['gemini_configured'] is False


def test_gemini_default_model_is_used_without_ollama_env(setup, monkeypatch):
    client, _ = setup
    monkeypatch.delenv('GEMINI_MODEL')
    monkeypatch.setenv('OLLAMA_MODEL', 'do-not-use-old-config')
    def handler(request):
        assert request.url.host == 'generativelanguage.googleapis.com'
        assert request.url.path == '/v1beta/models/gemini-3.8-flash:generateContent'
        return httpx.Response(200, json=gemini_response(script()))
    mock_http(monkeypatch, handler)
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 200, response.text


def test_gemini_ignores_thought_parts_when_reading_structured_result(setup, monkeypatch):
    client, _ = setup
    body = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'thought': True, 'text': 'Reasoning, not the requested JSON.'}, {'text': json.dumps(script())}]}}]}
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=body))
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 200, response.text
    assert response.json()['title'] == 'Test play'


def test_gemini_truncated_response_rejected_even_if_prefix_is_valid_json(setup, monkeypatch):
    client, _ = setup
    body = gemini_response(script())
    body['candidates'][0]['finishReason'] = 'MAX_TOKENS'
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=body))
    response = client.post('/api/direct', json={'source': script(), 'model': 's2.1-pro-free'})
    assert response.status_code == 502, response.text
    assert any(cue in response.json()['detail'].lower() for cue in ['truncat', 'limit', 'tokens']), response.text


@pytest.mark.parametrize('endpoint', ['analyze', 'direct'])
@pytest.mark.parametrize('error', [
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'Request contains an invalid argument.'},
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'responseMimeType application/json is not supported.'},
])
def test_gemini_invalid_argument_falls_back_once_with_same_input(setup, monkeypatch, endpoint, error):
    client, _ = setup
    calls = []
    def handler(request):
        calls.append(request)
        payload = json.loads(request.content)
        if len(calls) == 1:
            assert payload['generationConfig'] == {'responseMimeType': 'application/json'}
            return httpx.Response(400, json={'error': error})
        assert len(calls) == 2
        primary = json.loads(calls[0].content)
        assert request.url == calls[0].url
        assert request.headers['x-goog-api-key'] == calls[0].headers['x-goog-api-key']
        assert 'generationConfig' not in payload
        assert payload['contents'] == primary['contents']
        assert payload.get('systemInstruction') == primary.get('systemInstruction')
        return httpx.Response(200, json=gemini_response(script()))
    mock_http(monkeypatch, handler)
    if endpoint == 'analyze':
        response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    else:
        response = client.post('/api/direct', json={'source': script(), 'model': 's2.1-pro-free'})
    assert response.status_code == 200, response.text
    assert len(calls) == 2
    assert [c['id'] for c in response.json()['characters']] == ['c1', 'c2']


def test_gemini_fallback_rejection_is_actionable_and_never_retries_third_time(setup, monkeypatch):
    client, _ = setup
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={'error': {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'Request contains an invalid argument.'}})
    mock_http(monkeypatch, handler)
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 502, response.text
    assert len(calls) == 2
    detail = response.json()['detail'].lower()
    assert 'minimal' in detail and 'diagnose.py' in detail, response.text
    assert 'unit-gemini-secret' not in response.text


@pytest.mark.parametrize('error', [
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'API key not valid. Please pass a valid API key.', 'details': [{'reason': 'API_KEY_INVALID'}]},
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'Request contains an invalid argument.', 'details': [{'reason': 'API_KEY_INVALID'}]},
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'Access to this model is denied.'},
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'Billing must be enabled for this project.'},
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'Quota exceeded for the requested model.'},
    {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'Safety policy blocked this request.'},
])
def test_gemini_non_configuration_400_never_retries(setup, monkeypatch, error):
    client, _ = setup
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={'error': error})
    mock_http(monkeypatch, handler)
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 502, response.text
    assert len(calls) == 1
    assert 'unit-gemini-secret' not in response.text


def test_gemini_timeout_never_retries(setup, monkeypatch):
    client, _ = setup
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('timed out', request=request)
    mock_http(monkeypatch, handler)
    response = client.post('/api/analyze', files={'image': ('comic.png', comic_bytes(), 'image/png')})
    assert response.status_code == 502, response.text
    assert 'timed out' in response.json()['detail'].lower()
    assert len(calls) == 1
