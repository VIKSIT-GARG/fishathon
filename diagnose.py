#!/usr/bin/env python3
"""Run small live Gemini checks with your local .env; never print API keys.

Default: text + bundled image (two Gemini calls, plus at most one
compatibility retry per rejected JSON-mode request). --story adds a writer call.
No Fish synthesis or voice search is performed. Provider usage may be billed.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile
from backend import main


async def diagnose(image_path: Path, story: bool) -> int:
    main.gemini_key()
    print(f'Model: {main.gemini_model()}', flush=True)
    print('These are live Gemini checks; provider usage may be billed. No Fish calls.', flush=True)
    print('1. Text + JSON response ...', flush=True)
    content = await main.gemini_generate(
        'Return only this JSON object: {"status":"ok"}.',
        {'type': 'object', 'properties': {'status': {'type': 'string'}}, 'required': ['status']},
    )
    content = content.strip()
    if content.startswith('```'):
        content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content)
    if json.loads(content).get('status') != 'ok':
        raise ValueError('Unexpected text test result')
    print('   PASS: Gemini returned valid JSON.', flush=True)
    print('2. Actual comic extraction endpoint ...', flush=True)
    if not image_path.is_file():
        raise FileNotFoundError('Choose an existing JPEG, PNG or WebP with --image.')
    with image_path.open('rb') as stream:
        data = stream.read(main.MAX_UPLOAD + 1)
    source = await main.analyze(UploadFile(filename=image_path.name, file=io.BytesIO(data)))
    print(f'   PASS: validated {len(source.characters)} characters and {len(source.lines)} source lines.', flush=True)
    if story:
        print('3. Actual story writer endpoint (Fish casting disabled for this check) ...', flush=True)
        fish_key = os.environ.pop('FISH_API_KEY', None)
        try:
            result = await main.direct(main.DirectRequest(source=source, target_seconds=30))
        finally:
            if fish_key is not None:
                os.environ['FISH_API_KEY'] = fish_key
        print(f'   PASS: validated {len(result.lines)} original story lines.', flush=True)
    print('Gemini checks passed. This does not verify Fish audio or every possible image.', flush=True)
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=main.ROOT / 'examples' / 'last-cookie.png')
    parser.add_argument('--story', action='store_true', help='Also make a third Gemini call to test creative direction.')
    args = parser.parse_args()
    try:
        return asyncio.run(diagnose(args.image, args.story))
    except HTTPException as exc:
        print(f'FAIL: {exc.detail}', flush=True)
        return 1
    except (ValueError, OSError, TypeError) as exc:
        # Do not echo a response body, local key, full path or payload on failure.
        print(f'FAIL: {type(exc).__name__}. Check the selected image and retry; no valid result was confirmed.', flush=True)
        return 1
    except KeyboardInterrupt:
        print('Cancelled.', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(cli())
