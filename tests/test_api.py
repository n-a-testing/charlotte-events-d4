import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.base import clone
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import serve
from pipeline_def import EventProfileFeatures, build_pool, make_rows, parse_dt

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(serve.app)
        cls.cases = json.loads((ROOT/'evidence/parity-cases.json').read_text())
        cls.bundle = serve.BUNDLE

    def test_health_and_metadata(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        info = self.client.get('/pipeline-info').json()
        self.assertEqual(info['sklearn_version'], '1.8.0')
        self.assertEqual([s['class'] for s in info['steps']], ['EventProfileFeatures','StandardScaler','LogisticRegression'])

    def test_serialized_local_api_parity(self):
        for case in self.cases:
            r = self.client.post('/recommendations', json=case['request'])
            self.assertEqual(r.status_code, 200)
            actual = r.json()
            expected = copy.deepcopy(case['expected'])
            actual_scores = [event.pop('score') for event in actual['results']]
            expected_scores = [event.pop('score') for event in expected['results']]
            self.assertEqual(actual, expected)
            np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-10, atol=1e-12)

    def test_invalid_bounds(self):
        bad = [{'limit':0},{'limit':51},{'limit':'10'}, {'limit':True}, {'interests':['invented']},
               {'interests':['arts','arts']}, {'free_text':'a'*1001}, {'free_text':42},
               {'saved_ids':['x']*51}, {'saved_ids':['not-in-catalog']},
               {'as_of':'2026-11-15'}, {'as_of':'bad'}, {'as_of':0},
               {'as_of':'1900-01-01T00:00:00Z'}, {'as_of':'2101-01-01T00:00:00Z'},
               {'rating':2}, {'unexpected':'x'}]
        for payload in bad:
            with self.subTest(payload=str(payload)[:80]):
                self.assertEqual(self.client.post('/recommendations',json=payload).status_code,422)

    def test_empty_pool_and_empty_preferences(self):
        r = self.client.post('/recommendations',json={'as_of':'2099-01-01T00:00:00Z'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['results'],[])
        r = self.client.post('/recommendations',json={'as_of':'2026-11-15T00:00:00Z'})
        self.assertEqual(r.status_code,200)
        self.assertTrue(r.json()['results'])

    def test_timezone_equivalence(self):
        p = self.cases[0]['request'].copy()
        a = self.client.post('/recommendations',json=p).json()
        p['as_of'] = '2026-11-14T19:00:00-05:00'
        self.assertEqual(a,self.client.post('/recommendations',json=p).json())

    def test_no_fit_or_load_on_requests(self):
        with patch('joblib.load',side_effect=AssertionError('load per request')), \
             patch.object(Pipeline,'fit',side_effect=AssertionError('fit during serving')), \
             patch.object(EventProfileFeatures,'fit',side_effect=AssertionError('refit vocabulary')):
            for _ in range(2):
                self.assertEqual(self.client.post('/recommendations',json=self.cases[0]['request']).status_code,200)

    def test_saved_event_and_features(self):
        p = self.cases[0]['request'].copy()
        first = self.cases[0]['expected']['results'][0]['event_id']
        p['saved_ids'] = [first]
        r = self.client.post('/recommendations',json=p)
        self.assertEqual(r.status_code,200)
        self.assertNotIn(first,[x['event_id'] for x in r.json()['results']])
        rows = make_rows(p, build_pool(self.bundle['catalog'],parse_dt(p['as_of'])),self.bundle['catalog'])
        f = self.bundle['pipeline']['features'].transform(rows)
        self.assertTrue(np.isfinite(f).all())
        self.assertTrue((f[:,2] > 0).any())

    def test_test_ids_reserved_and_cloneable(self):
        audit = json.loads((ROOT/'evidence/build-audit.json').read_text())
        self.assertEqual(audit['reserved_test_overlap'],[])
        self.assertFalse(set(audit['fit_event_ids']) & set(self.bundle['excluded_ids']))
        self.assertFalse({e['id'] for e in self.bundle['catalog']} & set(self.bundle['excluded_ids']))
        cloned = clone(self.bundle['pipeline'])
        self.assertFalse(hasattr(cloned['features'],'vectorizer_'))

    def test_forbidden_fields_do_not_change_features(self):
        p = self.cases[0]['request']
        rows = make_rows(p,build_pool(self.bundle['catalog'],parse_dt(p['as_of'])),self.bundle['catalog'])[:4]
        altered = copy.deepcopy(rows)
        for r in altered:
            for key in ('rating','first_seen_at','last_seen_at','updated_at','content_hash'):
                r['event'][key] = 'deliberately changed'
        transform = self.bundle['pipeline']['features'].transform
        np.testing.assert_array_equal(transform(rows),transform(altered))

    def test_cors_and_docs(self):
        r = self.client.options('/recommendations',headers={'Origin':'https://example.vercel.app', 'Access-Control-Request-Method':'POST'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(self.client.get('/docs').status_code,200)
        self.assertIn('/recommendations',self.client.get('/openapi.json').json()['paths'])

    def test_missing_corrupt_wrong_version_unfitted_are_503(self):
        # Fresh interpreter per artifact verifies IMPORT-time handling without damaging live/local bundle.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'corrupt.joblib').write_bytes(b'not a pickle')
            bad = copy.deepcopy(self.bundle)
            bad['metadata']['sklearn_version'] = '0.0.0'
            joblib.dump(bad,directory/'wrong-version.joblib')
            bad = copy.deepcopy(self.bundle)
            bad['pipeline'] = clone(bad['pipeline'])
            joblib.dump(bad,directory/'unfitted.joblib')
            joblib.dump(self.bundle['pipeline'],directory/'bare-pipeline.joblib')
            for name in ['missing.joblib','corrupt.joblib','wrong-version.joblib','unfitted.joblib','bare-pipeline.joblib']:
                code = "from fastapi.testclient import TestClient; import serve; c=TestClient(serve.app); assert c.get('/health').status_code == 503; assert c.get('/pipeline-info').status_code == 503; assert c.post('/recommendations',json={}).status_code == 503; assert c.get('/docs').status_code == 200"
                out = subprocess.run([sys.executable,'-c',code],cwd=ROOT,
                    env={**os.environ,'PIPELINE_PATH':str(directory/name)},capture_output=True,text=True)
                self.assertEqual(out.returncode,0,out.stderr)

if __name__ == '__main__':
    unittest.main(verbosity=2)
