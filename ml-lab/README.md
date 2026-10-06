# ML Weekly Lab

Five beginner Python notebooks for the Practice tasks in the 21CSC305P Machine Learning assignment.

## Open the website

Double-click **start-lab.cmd**, or run `npm run dev` and open http://127.0.0.1:8000. Node.js is already installed on this computer. You can also use `python -m http.server 8000 --directory dist`.

Choose a unit and press **Run all**. Each code cell has a **Run cell** button, and its output and graphs appear below it. You can edit the code. Run the cells in order because later cells use variables created earlier. The first run needs internet to download Python and its packages; later units reuse the Python worker. **Stop** ends a run. **Reset kernel** clears variables. Changes last until you refresh the page; download your notebook to keep your edits.

The initial outputs are labeled **Reference run**. They were produced by executing the supplied notebooks. Running the code replaces them with fresh results. **Notebook** downloads the current unit; **All 5 notebooks** downloads all the reference notebooks and their CSV files.

## Deploy on Vercel

This website is static. Python runs in the visitor's browser through Pyodide; Vercel does not need a Python server or a notebook kernel. The Python `requirements.txt` is only for running the downloaded notebooks locally.

Commit and push the deployment configuration before deploying. In Vercel's **Settings > Build and Deployment**, set **Framework Preset** to **Other**. Use the repository root (leave **Root Directory** empty); the root `vercel.json` serves `ml-lab/dist`. If the existing project instead uses **Root Directory** `ml-lab`, the configuration in that folder serves `dist`. Both configurations skip dependency installation and the build step because the complete website is already committed.

Deploy the new commit. Redeploying the old commit will not include the fix. The five unit pages, CSV files, notebooks and ZIP are all in the static output directory. No Python entrypoint is required. When regenerating notebooks, run both scripts below and commit the updated `dist` files before deploying again.

## Our dataset: real highway traffic

The [UCI Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) records hourly vehicle counts at a westbound I-94 road sensor between Minneapolis and Saint Paul, USA. The original source contains 48,204 rows, including weather and holiday features from 2012–2018. Citation: Hogue, J. (2019), DOI: 10.24432/C5X60B. License: CC BY 4.0.

The teaching version has **1,008 continuous hourly observations (42 days)**: 13 April 2017 at 10:00 through 25 May 2017 at 09:00, in source local timestamps. Its columns are:

- `date_time`: the recorded hour.
- `traffic_volume`: vehicles counted during that hour.
- `temperature_c`: air temperature in degrees Celsius.

Multiple weather entries can refer to the same hour. We verify that their vehicle counts agree, keep the first entry per timestamp, and sort by time. We select the first six weeks of the longest uninterrupted hourly run before modelling. Temperature is converted from Kelvin by subtracting 273.15. We do not fill missing hours or invent measurements. The complete original compressed CSV is `data/traffic_original.csv.gz`; the small classroom CSV is `data/traffic.csv`. `data/metadata.json` records provenance, selection, units, and the original file hash.

**Faculty explanation:** “I use real highway sensor readings to study traffic patterns and predict the next hour's vehicle count.” High vehicle count means a busy hour; without speed or capacity it does not establish congestion. This is one sensor and one short window, so the scores do not establish performance across roads or seasons.

## What each notebook teaches

1. **Introduction:** read the CSV, inspect rows and missing values, summarize vehicle count and temperature, draw a histogram and scatter plot.
2. **Linear models:** predict the following hour's vehicle count with linear regression; classify the next hour as light/heavy using Bayesian logistic regression and linear SVM. `next_volume = traffic_volume.shift(-1)` creates 1,007 complete pairs. The first 805 pairs train the models and the final 202 test them. Heavy means at least 4,000 vehicles/hour, a preselected classroom cutoff. Only current vehicle count is the input. The Bayesian model uses an intercept/slope grid, Gaussian prior, training likelihood, and posterior predictive averaging.
3. **Clustering and PCA:** K-means, Gaussian mixtures, and Ward agglomerative clustering explore two groups using the same hour's vehicle count and temperature. Standardize both features; inspect cluster means before naming groups. PCA compresses those two features to one. The saved run retains **60.31% of standardized variance**, not 60.31% accuracy. Keeping two components retains all variance but does not reduce dimension.
4. **HMM:** a supervised two-state teaching HMM uses daytime/night-time regime labels from training timestamps (daytime: 06:00–19:59) and light/heavy traffic observations. The first 806 hours train the probability tables and the final 202 are tested. During test filtering, clock-based state labels are withheld and predictions use only previously observed traffic categories. Predictions are recorded before revealing the current category. Add-one smoothing avoids zero probabilities. A practical model would use its available clock; this example illustrates HMM filtering without Baum-Welch or Viterbi. It includes a previous-hour-category baseline, which scores better than the HMM in the saved run.
5. **Combining models:** a depth-two CART tree, 20-tree random forest, and 20-estimator AdaBoost classify the same next-hour target and use the same chronological split as Unit 2.

Each notebook has commented code, real executed outputs, graphs, changes to try, and viva questions. Run cells in order and explain the input, algorithm, and output.

## Open the notebooks in Jupyter

1. Install Python 3.12 or newer.
2. Run `python -m pip install -r requirements.txt` from this project folder.
3. Run `python -m jupyter lab` and open `notebooks`.
4. Choose **Restart Kernel and Run All Cells**.

Keep the `data` folder beside `notebooks`. The simple loading cell supports running from either the project root or the notebooks folder. It does not silently download a different dataset.

## Rebuild and verify

`python scripts/build_notebooks.py` recreates the CSV and five notebooks from the supplied original traffic CSV. It does not need internet. `python scripts/execute_notebooks.py` executes all five and creates `dist/ml-weekly-notebooks.zip`. `node scripts/test-browser.mjs` verifies fresh Python execution in all five website pages, graphs, editing, error recovery, stopping, downloads and mobile layout. The local Python package versions are in `environment-tested.txt`; the browser uses Pyodide 314.0.7 and can have small numerical differences.

The website contains the beginner notebooks. The previous pollution version is preserved only in an ignored local backup under `.sites-runtime`, outside the website and download pack.
