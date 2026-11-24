"""Compare a running API against the local saved bundle; write real evidence only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import httpx
import numpy as np
from fastapi.testclient import TestClient
import serve

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--allow-local', action='store_true')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    url = urlparse(base)
    deployed = url.scheme == 'https' and bool(url.hostname) and url.hostname.endswith('.modal.run')
    if not deployed and not (args.allow_local and url.hostname in ('localhost','127.0.0.1')):
        raise SystemExit('Use a verified https://...modal.run URL, or --allow-local for clearly labeled local evidence.')
    if url.username or url.password or url.query or url.fragment:
        raise SystemExit('Unexpected URL credentials/query/fragment')
    local = TestClient(serve.app)
    cases = json.loads((ROOT/'evidence/parity-cases.json').read_text())
    checks = []
    with httpx.Client(base_url=base, timeout=120, trust_env=deployed) as remote:
        for path in ['/health','/pipeline-info','/docs','/openapi.json']:
            result = remote.get(path)
            assert result.status_code == 200, (path,result.status_code,result.text[:200])
            if path in ('/health','/pipeline-info'):
                assert result.json()['artifact_id'] == serve.BUNDLE['metadata']['artifact_id']
            if path == '/pipeline-info':
                assert result.json() == local.get(path).json(), 'Remote metadata differs'
            checks.append({'path':path,'status':result.status_code,'passed':True})
        for case in cases:
            local_body = local.post('/recommendations',json=case['request']).json()
            r = remote.post('/recommendations',json=case['request'])
            assert r.status_code == 200, r.text
            remote_body = r.json()
            local_results, remote_results = local_body['results'], remote_body['results']
            np.testing.assert_allclose([r['score'] for r in local_results],[r['score'] for r in remote_results],rtol=1e-10,atol=1e-12)
            left, right = json.loads(json.dumps(local_body)), json.loads(json.dumps(remote_body))
            for result in left['results'] + right['results']:
                result.pop('score')
            assert left == right, 'Ranking, metadata or event fields differ'
            delta = max((abs(a['score']-b['score']) for a,b in zip(local_results,remote_results)),default=0)
            checks.append({'path':'/recommendations','status':r.status_code,'request':case['request'],
                           'response':remote_body,'max_score_difference':delta,'passed':True})
        for payload in [{'limit':0},{'interests':['invalid-topic']},{'free_text':'a'*1001},{'as_of':'bad'}]:
            r=remote.post('/recommendations',json=payload)
            assert r.status_code == 422, (payload,r.status_code)
            checks.append({'path':'/recommendations','status':r.status_code,'invalid_case':str(payload)[:100],'passed':True})
    report={'verified_at':datetime.now(timezone.utc).isoformat(),'base_url':base,
            'scope':'deployed Modal' if deployed else 'local HTTP only',
            'artifact_id':serve.BUNDLE['metadata']['artifact_id'],
            'artifact_sha256':hashlib.sha256((ROOT/'pipeline.joblib').read_bytes()).hexdigest(),
            'score_rtol':1e-10,'score_atol':1e-12,'checks':checks,'passed':True}
    path = ROOT/'evidence'/('deployed-verification.json' if deployed else 'local-http-verification.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} checks passed; {report["scope"]}; saved {path.name}')

if __name__ == '__main__':
    main()
