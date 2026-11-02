# SKLearn_Project2: predicting YouTube video popularity

**Status: archived MSc coursework (2023). Not maintained.**

Project 2 of the *Big Data Analytics* module, MSc in Artificial Intelligence & Big Data (Atlantic
Technological University). The task was to load, combine and model a multi-country dataset with
scikit-learn.

- **Data:** the *Trending YouTube Video Statistics* dataset (Kaggle): `CAvideos.csv`, `FRvideos.csv`,
  `INvideos.csv`, `USvideos.csv` plus each country's `*_category_id.json`.
- **First half (top level):** `main.py` loads and combines the country files (`MyDataLoader`,
  `DataCombiner`, `SplitData`, `SaveDataFrame`) and plots summaries (`DataVisualiser`).
  `YoutubeLikesMain.ipynb` is the notebook version.
- **Second half (`MLEnsemble/`):** `main.py` trains regressors on the combined data. Linear
  regression, random forest (with grid search) and PCA were tried and are left commented out; the
  final model is a `GradientBoostingRegressor`.

Run from the folder you want (each `main.py` expects its CSVs in the current directory):

```bash
python main.py              # data loading and combining
cd MLEnsemble && python main.py   # model training
```

Written under coursework deadlines: no tests, no pinned dependencies, and the large CSVs are
committed directly. Kept as a record of my early machine-learning work.
