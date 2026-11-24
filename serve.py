"""Load the trusted, fitted bundle ONCE at import. No fitting during serving."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
import logging
import os
import warnings
import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, StringConstraints, field_validator
from pipeline_def import EventProfileFeatures, recommend_bundle

ARTIFACT_PATH = Path(os.environ.get('PIPELINE_PATH', Path(__file__).with_name('pipeline.joblib')))
BUNDLE = None
LOAD_ERROR = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        loaded = joblib.load(ARTIFACT_PATH)
    if not isinstance(loaded, dict) or not {'pipeline', 'catalog', 'metadata', 'excluded_ids'} <= loaded.keys():
        raise ValueError('Invalid bundle structure')
    m = loaded['metadata']
    if not {'steps', 'built_at', 'sklearn_version', 'artifact_id'} <= m.keys():
        raise ValueError('Required metadata missing')
    if m['sklearn_version'] != sklearn.__version__:
        raise ValueError('scikit-learn version mismatch')
    p = loaded['pipeline']
    if not isinstance(p, Pipeline) or not isinstance(p['features'], EventProfileFeatures):
        raise ValueError('Expected EventProfileFeatures in a fitted Pipeline')
    check_is_fitted(p['features'], 'vectorizer_')
    check_is_fitted(p['scale'])
    check_is_fitted(p['model'])
    if any(e['id'] in set(loaded['excluded_ids']) for e in loaded['catalog']):
        raise ValueError('Reserved test event in serving catalog')
    BUNDLE = loaded
except Exception as exc:
    LOAD_ERROR = type(exc).__name__
    logging.getLogger(__name__).error('Artifact unavailable (%s)', LOAD_ERROR)

app = FastAPI(title='Charlotte Event Recommendations', version='4.0.0',
              description='Frozen D3 logistic-regression pipeline. Academic prototype with AI-assigned relevance ratings.')
# No credentials, no personal data storage. Restrict ALLOWED_ORIGINS when connecting the D5 frontend.
origins = [s.strip() for s in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=['GET','POST'], allow_headers=['Content-Type'])

Topic = Literal['technology','arts','sports','wellness','career','community','academic','culture']
EventId = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')]

class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'example': {
        'interests':['technology','career'], 'free_text':'coding internships and robotics',
        'saved_ids':[], 'as_of':'2026-11-15T00:00:00Z', 'limit':10}})
    interests: list[Topic] = Field(default_factory=list, max_length=8)
    free_text: str = Field(default='', min_length=0, max_length=1000)
    saved_ids: list[EventId] = Field(default_factory=list, max_length=50)
    as_of: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc),
        ge=datetime(2020,1,1,tzinfo=timezone.utc), lt=datetime(2100,1,1,tzinfo=timezone.utc))
    limit: int = Field(default=10, ge=1, le=50, strict=True)

    @field_validator('interests','saved_ids')
    @classmethod
    def unique_values(cls, values):
        if len(set(values)) != len(values):
            raise ValueError('Duplicate values are not allowed')
        return values

    @field_validator('as_of', mode='before')
    @classmethod
    def iso_timestamp_only(cls, value):
        if not isinstance(value, (str, datetime)):
            raise ValueError('Use an ISO 8601 timestamp with a timezone')
        return value

    @field_validator('saved_ids')
    @classmethod
    def known_ids(cls, values):
        if BUNDLE is not None:
            known = {e['id'] for e in BUNDLE['catalog']}
            if any(v not in known for v in values):
                raise ValueError('Unknown or reserved saved event ID')
        return values

def require_bundle():
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail='Model artifact unavailable; rebuild or restore the matching bundle.')
    return BUNDLE

@app.get('/health', responses={503: {'description': 'Artifact unavailable'}})
def health():
    b = require_bundle()
    return {'status':'ok','artifact_loaded':True,'artifact_id':b['metadata']['artifact_id'],
            'sklearn_version':sklearn.__version__}

@app.get('/pipeline-info', responses={503: {'description': 'Artifact unavailable'}})
def pipeline_info():
    return require_bundle()['metadata']

@app.post('/recommendations', responses={503: {'description': 'Artifact unavailable'}})
def recommendations(request: RecommendationRequest):
    b = require_bundle()
    profile = request.model_dump(mode='json')
    # UTC normalization matches the training snapshot; offset-equivalent requests produce equal scores.
    profile['as_of'] = request.as_of.astimezone(timezone.utc).isoformat()
    return recommend_bundle(b, profile, request.limit)
