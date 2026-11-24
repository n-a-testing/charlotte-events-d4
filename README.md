# D4 Fitted Pipeline and API

Group 13 · DTSC 3602 · Charlotte event recommendations

Group 13 serves the fitted D3 logistic regression model through FastAPI on Modal. The Word report summarizes the pipeline and its limitations. The recorded live verification passed all 11 checks on November 23, 2026.

[API documentation](https://n-a-testing--charlotte-events-d4-api.us-east.modal.run/docs) · [Health check](https://n-a-testing--charlotte-events-d4-api.us-east.modal.run/health) · [Live verification results](evidence/deployed-verification.json)

## What is included

| File | Purpose |
| --- | --- |
| `pipeline_def.py` | Importable `EventProfileFeatures` custom transformer and deterministic ranking helpers |
| `build_pipeline.py` | Development-only tuning, final fit, bundle serialization, provenance and replay cases |
| `pipeline.joblib` | Fitted dictionary bundle: `pipeline`, `catalog`, `excluded_ids`, `metadata` |
| `metadata.json` | Required metadata plus versions, data hashes, feature names, training counts and limitations |
| `serve.py` | Import-time load; `/health`, `/pipeline-info`, `/recommendations`, `/docs` |
| `modal_serve.py` | Separate Modal app `charlotte-events-d4`; ships exactly the three required Python/artifact files |
| `requirements.txt` / `requirements-lock.txt` | Exact direct pins and full runtime dependency closure, including scikit-learn 1.8.0 |
| `verify_api.py` | Checks HTTP behavior and compares local and remote predictions |
| `configure_remote.py` | Verifies the live Modal URL, then writes it into frontend and Postman configuration |
| `postman_collection.json` | Assertions for health, pipeline info, valid 200 and invalid 422 responses |
| `frontend/` | Static Vercel-ready frontend with loading, error and empty states; no fake results |
| `tests/` and `evidence/` | Local tests, local and deployed responses, D3 feature parity and build audit |
| `data/` | Four frozen D3 input files needed for reproducible rebuilding |
| `D4_Explanation.md` | Brief plain-language explanation |

## Run locally

Use Python 3.12. The supplied artifact was built with Python 3.12.14 and scikit-learn 1.8.0.

```bash
cd D4_Group13
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python -m uvicorn serve:app --reload
```

Open http://localhost:8000/docs. Expand `POST /recommendations`, click **Try it out**, and run the supplied historical example. The explicit November 15 date reproduces D3. Omitting `as_of` uses the current date and can legitimately return no events from this frozen snapshot.

In a second terminal with the same virtual environment:

```bash
python verify_api.py --base-url http://127.0.0.1:8000 --allow-local
```

This writes **local-only** evidence. It does not establish Modal parity.

## Deploy and prove it

Use a terminal authenticated to your Modal workspace. Authentication must happen on the machine running `modal deploy`.

```bash
python -m pip install modal==1.5.4
modal setup
modal token info
modal deploy modal_serve.py
```

If already authenticated, skip `modal setup`. The app name differs from D2, so the existing D2 app is not redeployed. Copy the **actual API URL printed by Modal**, without `/docs` or a trailing slash. Then run:

```bash
python configure_remote.py https://n-a-testing--charlotte-events-d4-api.us-east.modal.run
```

The URL shown is the verified D4 deployment. The script refuses non-Modal URLs, verifies the remote artifact and matching predictions, writes `evidence/deployed-verification.json`, and updates both `frontend/config.json` and the Postman `base_url`. It will not record success if deployment or comparison fails.

For Vercel, deploy `frontend/` as a static project (framework **Other**, no build command). If using the Vercel CLI from this folder:

```bash
cd frontend
vercel --prod
```

The frontend calls the configured live Modal API. There is no localhost fallback. Use the deployed page, select interests, and verify that real recommendations appear. The API supports credential-free CORS; no user profile is stored. You can narrow `ALLOWED_ORIGINS` to the frontend origin in the Modal image configuration before redeploying.

Import `postman_collection.json` into Postman and run all five requests against its configured deployed URL. Capture the **valid POST 200** response with passing assertions and the **invalid POST 422** response with passing assertions. Local HTTP evidence is not a substitute for those requested screenshots.

## Rebuild

The fitted bundle is already included. Rebuild only when deliberately producing a new version:

```bash
python build_pipeline.py
python -m unittest discover -s tests -v
```

Do not run the build script during API startup. Rebuilding updates `pipeline.joblib`, `metadata.json`, `pipeline.sha256`, and the build/parity evidence. Redeploy and re-run `configure_remote.py` after a rebuild. `pipeline.sha256` verifies transfer integrity; it is not a trust signature. Load only trusted joblib files.

## Analysis and data lineage

Inputs are copied unchanged from [the D3 repository](https://github.com/onlytvisha/charlotte-events-model-eval). SHA-256 hashes are in metadata. This package does not modify the D2 repository or deployment.

- 1,093 source event records; 915 remain in the serving catalog after test-ID exclusion.
- The frozen manifest contains 179 test IDs; 178 are present in the checked-out D3 snapshot. This measured count differs from the D3 note's statement of 179 present.
- 558 eligible, deduplicated events in the initial 14-day development window; 12 profiles produce 6,696 training pairs.
- 462 explicit relevance judgments. There are 263 positive training pairs; all other 6,433 pairs are treated as negative under D3's `unjudged = 0` convention.
- Seven non-test historical events fit a 282-term TF-IDF vocabulary. No candidate or test text is used to fit that vocabulary.
- Five-fold profile-grouped development CV selects `C=1.0` from D3's grid `{0.01, 0.1, 1.0}`. Then all development profiles are used for the saved fit. This CV is a final tuning step, not a new unbiased estimate of model quality.
- The original D3 leave-one-profile-out nDCG@10 was 0.392 for logistic regression versus 0.063 for chronological ordering and 0.361 for the tree. These are **reported D3 dry-run results**, not D4 or D6 test scores. D3's heuristic and embedding rankers scored higher; logistic regression is the selected fitted classifier, not the best possible recommender.

### Limitations carried forward

Profiles are placeholders and ratings are Claude-assigned, not human preferences. Unjudged events may be relevant. Ranking scores are not calibrated probabilities of attendance or satisfaction. Seven historical documents provide limited text vocabulary coverage. The snapshot was collected after the nominal initial cutoff, so genuine availability of all content at prediction time is not established. D4 preserves D3's feature semantics, including UTC start hour and weekday, rather than silently changing the evaluated model. The frozen manifest covers an earlier snapshot; newly arriving recurring events need an additional group audit before D6. Excluding test IDs does not independently prove that all similar events are separated. Organizer coverage remains unreliable for Campus Events.

No final test metrics or week-later ratings are used in this build. Serving holds reserved IDs out as well. Consequently, this is a reproducible academic demo, not a complete live event catalog. D6 must explicitly address the inherited split and annotation limitations before making real-user performance claims.

## Verification status

- PASS: 11 local unit tests, including 17 invalid-input cases and 5 artifact-failure scenarios.
- PASS: saved/reloaded prediction identity and local HTTP prediction parity for three profiles.
- PASS: all 12 D3 features checked across 6,696 pairs; largest absolute difference `1.6653345369377348e-16` (tolerance `1e-12`).
- PASS: reserved test-ID overlap with fitting corpus and candidate training rows is zero.
- Modal passed all 11 live checks. The top 10 events matched in order across three profiles. The maximum absolute score difference was 2.22 × 10⁻¹⁶.
- PASS: frontend JavaScript syntax check.
- PENDING: frontend browser interaction tests (the workspace blocked launching Chromium), Vercel deployment and real Postman screenshots against the deployed URL.

## AI assistance

OpenAI Codex generated the D4 implementation, documentation and local tests using the supplied project instructions and existing D3 repository. D3's placeholder profiles and Claude-assigned ratings are reused and explicitly disclosed. The group must review and understand the code and accurately document who reviewed, tested, deployed and submitted it.
