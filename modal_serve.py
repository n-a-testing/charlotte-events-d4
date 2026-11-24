"""Separate D4 deployment; leaves the D2 service intact."""
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install_from_requirements(str(ROOT/'requirements-lock.txt'))
         .add_local_file(ROOT/'serve.py', '/root/serve.py')
         .add_local_file(ROOT/'pipeline_def.py', '/root/pipeline_def.py')
         .add_local_file(ROOT/'pipeline.joblib', '/root/pipeline.joblib'))
app = modal.App('charlotte-events-d4')

@app.function(image=image, cpu=1, memory=1024, max_containers=2, timeout=120)
@modal.asgi_app()
def api():
    from serve import app as application
    return application
