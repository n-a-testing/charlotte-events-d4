"""Fit D3's selected model; never touch D6 test labels. Run from any directory."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata as versions
import json
from pathlib import Path
import platform
import subprocess
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pipeline_def import EventProfileFeatures, FEATURES, build_pool, event_text, make_rows, parse_dt, recommend_bundle

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ndcg(scores, pool, ratings):
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]['start_at'], pool[i]['id']))[:10]
    gains = [ratings.get(pool[i]['id'], 0) for i in order]
    ideal = sorted(ratings.values(), reverse=True)[:10]
    dcg = lambda x: sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(x))
    return dcg(gains) / dcg(ideal) if dcg(ideal) else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'pipeline.joblib')
    args = parser.parse_args()
    d = args.data_dir
    names = ['d3_events.json', 'profiles.json', 'judgments.csv', 'split_manifest.json']
    hashes = {name: digest(d / name) for name in names}
    events = json.loads((d/'d3_events.json').read_text())
    profiles = json.loads((d/'profiles.json').read_text())
    split = json.loads((d/'split_manifest.json').read_text())
    excluded = {r['event_id'] for r in split if r['split'] == 'test'}
    cutoff = parse_dt(profiles[0]['as_of'])
    assert all(parse_dt(p['as_of']) == cutoff for p in profiles)
    ratings = {p['profile_id']: {} for p in profiles}
    with (d/'judgments.csv').open() as f:
        for row in csv.DictReader(f):
            if row['event_id'] in excluded:
                raise ValueError('A reserved test event was found in the ratings file')
            key = row['profile_id']
            value = int(row['rating'])
            if value not in (0, 1, 2) or row['event_id'] in ratings[key]:
                raise ValueError('Invalid or duplicate rating')
            ratings[key][row['event_id']] = value
    catalog = [e for e in events if e['id'] not in excluded]
    history = [e for e in catalog if parse_dt(e['start_at']) < cutoff]
    texts = tuple(event_text(e) for e in history)
    pool = build_pool(catalog, cutoff)
    rows, y, groups = [], [], []
    for p in profiles:
        rows.extend(make_rows(p, pool, catalog))
        y.extend(int(ratings[p['profile_id']].get(e['id'], 0) >= 1) for e in pool)
        groups.extend([p['profile_id']] * len(pool))
    y = np.asarray(y)
    # Historical, label-free corpus is identical across D3 profile folds.
    features = EventProfileFeatures(texts).fit(rows).transform(rows)
    pids = [p['profile_id'] for p in profiles]
    cv = GroupKFold(n_splits=5)
    tuning = []
    for C in (0.01, 0.1, 1.0):
        scores = []
        for train, valid in cv.split(pids, groups=pids):
            train_ids = {pids[i] for i in train}
            mask = np.array([g in train_ids for g in groups])
            m = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=C, max_iter=2000, class_weight='balanced', random_state=0))])
            m.fit(features[mask], y[mask])
            for i in valid:
                ix = np.array([g == pids[i] for g in groups])
                scores.append(ndcg(m.predict_proba(features[ix])[:, 1], pool, ratings[pids[i]]))
        tuning.append({'C': C, 'mean_validation_ndcg_at_10': float(np.mean(scores))})
    C = max(tuning, key=lambda r: r['mean_validation_ndcg_at_10'])['C']
    pipeline = Pipeline([('features', EventProfileFeatures(texts)), ('scale', StandardScaler()),
                         ('model', LogisticRegression(C=C, max_iter=2000, class_weight='balanced', random_state=0))])
    pipeline.fit(rows, y)
    fit_ids = {e['id'] for e in history + pool}
    assert not fit_ids.intersection(excluded)
    version = sklearn.__version__
    artifact_id = hashlib.sha256(json.dumps({'inputs': hashes, 'C': C, 'sklearn': version,
        'pipeline_def': digest(ROOT/'pipeline_def.py')}, sort_keys=True).encode()).hexdigest()[:20]
    metadata = {'steps': [{'name': n, 'class': type(step).__name__} for n, step in pipeline.steps],
                'built_at': datetime.now(timezone.utc).isoformat(), 'sklearn_version': version,
                'python_version': platform.python_version(), 'artifact_id': artifact_id,
                'feature_names': FEATURES, 'C': C, 'tuning': tuning,
                'cutoff': cutoff.isoformat(), 'window_days': 14, 'profile_count': len(profiles),
                'source_event_count': len(events), 'catalog_count': len(catalog),
                'candidate_count': len(pool), 'training_pair_count': len(rows),
                'positive_pair_count': int(y.sum()), 'negative_pair_count': int((1-y).sum()),
                'judgment_count': sum(len(v) for v in ratings.values()),
                'history_document_count': len(history), 'vocabulary_size': len(pipeline['features'].vectorizer_.vocabulary_),
                'reserved_test_ids': len(excluded), 'reserved_test_ids_present': len(excluded & {e['id'] for e in events}),
                'test_ids_used_in_fit': 0, 'input_sha256': hashes,
                'pipeline_def_sha256': digest(ROOT/'pipeline_def.py'),
                'source_repository': 'https://github.com/onlytvisha/charlotte-events-model-eval',
                'source_commit': '078c5a99e0868497c1febdd9c79a5488ac985fa8',
                'label_provenance': 'D3 placeholder profiles and Claude-assigned 0/1/2 relevance ratings; unjudged=0',
                'fit_policy': 'Refit on D3 development window only. Week-later ratings and D6 test labels unused.',
                'time_features': 'Preserve D3 source timestamp hour and weekday (snapshot timestamps are UTC).',
                'snapshot_warning': 'Retrospective D3 snapshot, not an as-of archived feed; availability at cutoff is unverified.',
                'dependencies': {n: versions.version(n) for n in ['scikit-learn', 'numpy', 'scipy', 'joblib', 'fastapi', 'pydantic']}}
    bundle = {'pipeline': pipeline, 'catalog': catalog, 'excluded_ids': sorted(excluded), 'metadata': metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    reloaded = joblib.load(args.output)
    np.testing.assert_allclose(pipeline.predict_proba(rows[:20]), reloaded['pipeline'].predict_proba(rows[:20]), rtol=0, atol=0)
    (ROOT/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (ROOT/'pipeline.sha256').write_text(digest(args.output)+'  pipeline.joblib\n')
    (ROOT/'evidence').mkdir(exist_ok=True)
    audit = {'fit_event_ids': sorted(fit_ids), 'history_ids': [e['id'] for e in history],
             'candidate_ids': [e['id'] for e in pool], 'reserved_test_overlap': sorted(fit_ids & excluded),
             'serialized_predictions_identical': True}
    (ROOT/'evidence/build-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    examples = []
    for p in profiles[:3]:
        request = {k:v for k,v in p.items() if k != 'profile_id'}
        request['as_of'] = parse_dt(p['as_of']).isoformat()
        request['limit'] = 10
        response = recommend_bundle(reloaded, request, 10)
        examples.append({'request': request, 'expected': response})
    (ROOT/'evidence/parity-cases.json').write_text(json.dumps(examples, indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['artifact_id','C','training_pair_count','candidate_count','positive_pair_count','history_document_count','sklearn_version']}, indent=2))

if __name__ == '__main__':
    main()
