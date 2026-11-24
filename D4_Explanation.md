# D4 explanation

D3 selected logistic regression. D4 saves that fitted model and makes it available through FastAPI.

The custom transformer, EventProfileFeatures, learns a TF-IDF vocabulary and converts a student profile and event into the same twelve features used in D3. StandardScaler applies learned scaling, and LogisticRegression returns scores used to rank upcoming events. The full pipeline and event catalog are saved in a dictionary bundle using scikit-learn 1.8.0.

The API loads the artifact once at import. It returns 422 for invalid input and 503 when the model cannot load. All eleven local unit tests and eleven local HTTP checks passed, including matching rankings for three profiles. The custom transformer also matches D3’s feature calculations across all 6,696 pairs.

The profiles and ratings are AI-generated, many unjudged pairs count as negative, and the text vocabulary comes from only seven historical events. The scores are not validated attendance probabilities.

The D4 app runs in the n-a-testing workspace. All 11 live checks passed. Each of the three profiles received the same top ten events in the same order from the saved model and the live API.

[API documentation](https://n-a-testing--charlotte-events-d4-api.us-east.modal.run/docs) · [Health check](https://n-a-testing--charlotte-events-d4-api.us-east.modal.run/health)

[D2 repository](https://github.com/n-a-testing/charlotte-events-data) · [D3 repository](https://github.com/onlytvisha/charlotte-events-model-eval)
