"""Serializable D4 feature engineering; no fitting or external I/O at import."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import math
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

FEATURES = ['topic_matches', 'any_match', 'saved_sim', 'tfidf_cos', 'days_ahead',
            'log_desc_len', 'n_topics', 'group_size', 'free_food', 'free_admission',
            'start_hour', 'weekday']
TOPICS = ['technology', 'arts', 'sports', 'wellness', 'career', 'community', 'academic', 'culture']

def parse_dt(value):
    d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

def event_text(event):
    return ' '.join(str(event.get(k) or '') for k in ('title', 'description'))

def tags_of(event):
    return set((event.get('tags') or {}).get('topics') or [])

def group_key(event):
    t = re.sub(r'\([^)]*\)', ' ', (event.get('title') or '').lower())
    t = re.sub(r'\s+at\s+.*$', '', t)
    return ' '.join(re.sub(r'[^a-z ]+', ' ', t).split())

def build_pool(events, as_of, excluded=()):
    end, seen, pool = as_of + timedelta(days=14), set(), []
    for e in sorted(events, key=lambda e: (e['start_at'], e['id'])):
        if e['id'] in excluded or e.get('status') != 'active':
            continue
        if not as_of <= parse_dt(e['start_at']) < end:
            continue
        key = e.get('duplicate_key') or e['id']
        if key not in seen:
            seen.add(key)
            pool.append(e)
    return pool

def make_rows(profile, pool, catalog):
    """Raw profile/event pairs. Context is computed from the complete candidate pool."""
    by_id = {e['id']: e for e in catalog}
    saved = [by_id[i] for i in profile.get('saved_ids', []) if i in by_id]
    counts = Counter(group_key(e) for e in pool)
    return [{'profile': profile, 'event': e, 'saved_events': saved,
             'group_size': counts[group_key(e)]} for e in pool]

class EventProfileFeatures(TransformerMixin, BaseEstimator):
    """Learn historical TF-IDF, then produce D3's 12 profile/event features.

    history_texts must be non-test event text strictly before the build cutoff.
    No ratings or operational timestamps are inspected by this transformer.
    """
    def __init__(self, history_texts=()):
        self.history_texts = history_texts

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 1), min_df=1, stop_words='english')
        self.vectorizer_.fit(self.history_texts)
        self.n_features_out_ = len(FEATURES)
        return self

    def transform(self, X):
        check_is_fitted(self, 'vectorizer_')
        if not X:
            return np.empty((0, len(FEATURES)), dtype=float)
        # Fit is NEVER called here. Sparse row products equal cosine because TF-IDF is L2-normalized.
        queries = [' '.join([' '.join(r['profile'].get('interests', [])),
                            r['profile'].get('free_text') or '',
                            ' '.join(event_text(e) for e in r['saved_events'])]) for r in X]
        q = self.vectorizer_.transform(queries)
        evec = self.vectorizer_.transform([event_text(r['event']) for r in X])
        cosine = np.asarray(q.multiply(evec).sum(axis=1)).ravel()
        values = []
        for r, similarity in zip(X, cosine):
            p, e = r['profile'], r['event']
            topics, interests = tags_of(e), set(p.get('interests', []))
            saved = set().union(*(tags_of(s) for s in r['saved_events']))
            start, cutoff = parse_dt(e['start_at']), parse_dt(p['as_of'])
            tags = e.get('tags') or {}
            values.append([len(topics & interests), float(bool(topics & interests)),
                           len(topics & saved) / max(1, len(topics | saved)), similarity,
                           (start - cutoff).total_seconds() / 86400,
                           math.log1p(len(e.get('description') or '')), len(topics), r['group_size'],
                           float(tags.get('free_food') == 'yes'), float(tags.get('free_admission') == 'yes'),
                           start.hour, start.weekday()])
        return np.asarray(values, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)

def recommend_bundle(bundle, profile, limit=10):
    catalog = bundle['catalog']
    pool = build_pool(catalog, parse_dt(profile['as_of']))
    rows = make_rows(profile, pool, catalog)
    scores = bundle['pipeline'].predict_proba(rows)[:, 1] if rows else []
    saved_ids = set(profile.get('saved_ids', []))
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]['start_at'], pool[i]['id']))
    items = []
    for i in order:
        e = pool[i]
        if e['id'] in saved_ids:
            continue
        items.append({'event_id': e['id'], 'title': e['title'], 'start_at': e['start_at'],
                      'location': e.get('location'), 'url': e.get('url'),
                      'source': e['source_id'], 'topics': sorted(tags_of(e)),
                      'score': float(scores[i])})
        if len(items) == limit:
            break
    return {'model': 'logistic_regression', 'artifact_id': bundle['metadata']['artifact_id'],
            'as_of': profile['as_of'], 'candidate_count': len(pool), 'results': items,
            'warnings': ['Frozen D3 snapshot; this is not a live campus calendar.',
                         'AI-assigned relevance ratings; scores are not calibrated attendance probabilities.',
                         'Reserved test events are withheld. Check official source pages for event details.']}
