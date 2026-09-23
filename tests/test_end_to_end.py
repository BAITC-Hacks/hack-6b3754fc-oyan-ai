"""Real catalog -> loader -> filters -> ranking -> cards -> UI integration."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from data_loader import load_contractors
from recommender import recommend

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / 'demo_queries.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', ['dense', 'date_b', 'rare', 'no_match', 'missing_category'])
def test_real_saved_response_and_source_facts(name):
    data = manifest()
    case = next(c for c in data['cases'] if c['name'] == name)
    contractors = load_contractors(ROOT / data['dataset_path'])
    before = deepcopy(contractors)
    result = recommend(case['query'], contractors)
    assert result == case['response']
    assert contractors == before
    assert result['results'] == recommend(case['query'], list(reversed(contractors)))['results']
    assert len(result['results']) <= 3
    stats = result['stats']
    assert stats['city_category_total'] == stats['eligible_count'] + sum(stats['rejected_first_reason'].values())
    for card in result['results']:
        q = case['query']
        assert q['date'] not in card['busy_dates']
        assert card['price_from_kzt'] <= q['budget']
        assert q['event_type'] in card['event_formats']
        assert q['language'] is None or q['language'] in card['languages']
        assert q['duration'] is None or card['max_hours'] is None or q['duration'] <= card['max_hours']
        for fact in card['evidence']:
            if fact['field'] == 'description' and 'start' in fact:
                assert card['description'][fact['start']:fact['end']] == fact['value']


def test_date_pair_change_is_explained_by_calendar():
    cases = {c['name']: c for c in manifest()['cases']}
    a, b = cases['dense'], cases['date_b']
    assert [k for k in a['query'] if a['query'][k] != b['query'][k]] == ['date']
    removed = {c['id'] for c in a['response']['results']} - {c['id'] for c in b['response']['results']}
    assert removed
    busy = {r['id'] for r in b['response']['rejections'] if r['primary_reason'] == 'busy'}
    assert removed <= busy


def test_new_process_repeats_saved_responses():
    run = subprocess.run([sys.executable, str(ROOT / 'run_demo.py'), '--verify'],
                         cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert run.returncode == 0, run.stderr or run.stdout
    assert 'OK: 5 saved responses' in run.stdout


@pytest.mark.parametrize('name', ['dense', 'date_b', 'rare', 'no_match', 'missing_category'])
def test_real_demo_in_streamlit(name):
    case = next(c for c in manifest()['cases'] if c['name'] == name)
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=15)
    assert not at.exception and not at.error
    at.selectbox(key='demo_choice').select(name).run()
    at.button[0].click().run(timeout=15)
    assert not at.exception and not at.error
    result = case['response']
    assert [s.value for s in at.subheader][1:] == [
        f"{i}. {c['anon_name']}" for i, c in enumerate(result['results'], 1)
    ]
    texts = [e.value for e in at.markdown] + [e.value for e in at.success]
    assert result['message'] in texts
    for card in result['results']:
        assert card['explanation'] in texts
