"""Verify Modal first, then configure frontend and Postman with that exact URL."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('url'); a=p.parse_args()
subprocess.run([sys.executable,str(ROOT/'verify_api.py'),'--base-url',a.url],check=True)
url=a.url.rstrip('/')
(ROOT/'frontend/config.json').write_text(json.dumps({'apiBase':url},indent=2)+'\n')
f=ROOT/'postman_collection.json'; collection=json.loads(f.read_text())
collection['variable'][0]['value']=url
f.write_text(json.dumps(collection,indent=2)+'\n')
print('Verified URL saved to frontend/config.json and postman_collection.json')
