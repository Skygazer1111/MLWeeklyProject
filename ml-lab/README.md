# ML Weekly Lab

Five beginner Python notebooks for the Practice tasks in the 21CSC305P Machine Learning assignment.

## Open the website

Double-click **start-lab.cmd**, or run `npm run dev` and open http://127.0.0.1:8000. Node.js is already installed on this computer. You can also use `python -m http.server 8000 --directory dist`.

Choose a unit and press **Run all**. Each code cell has a **Run cell** button, and its output and graphs appear below it. You can edit the code. Run the cells in order because later cells use variables created earlier. The first run needs internet to download Python and its packages; later units reuse the Python worker. **Stop** ends a run. **Reset kernel** clears variables. Changes last until you refresh the page; download your notebook to keep your edits.

The initial outputs are labeled **Reference run**. They were produced by executing the supplied notebooks. Running the code replaces them with fresh results. **Notebook** downloads the current unit; **All 5 notebooks** downloads all the reference notebooks and their CSV files.

## Our dataset: Old Faithful geyser

A geyser is a hot spring that sometimes shoots water into the air. This small real dataset records **299 eruptions** in their original observation order. It has only two numeric columns, both in minutes:

- `duration`: how long an eruption lasted.
- `waiting`: how long people waited **before that eruption**.

There are no missing values. The only preparation is removing the mirror CSV's row-number column. The original measurements are retained in `data/geyser_original.csv`; the easy-to-read version is `data/geyser.csv`.

Source: [R MASS geyser documentation](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/geyser.html). CSV mirror: [Rdatasets MASS/geyser](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/geyser.csv). Reference: Azzalini, A. and Bowman, A. W. (1990), *A look at some data on the Old Faithful geyser*, Applied Statistics 39, 357-365. Measurements were collected August 1-15, 1985. Some night-time durations were coded as 2, 3 or 4 minutes from descriptions rather than exact timing. This is a stated limitation of the source.

To predict the wait **after** an eruption, the notebooks create `next_wait` using `waiting.shift(-1)`: the next row's waiting time follows the current eruption. This leaves 298 complete pairs. Units 2, 4 and 5 use the first 238 pairs for training and the final 60 pairs for testing. They preserve the recorded order. A wait of at least 70 minutes is called a long wait for this classroom experiment; this cutoff is chosen before testing.

## What each notebook teaches

1. **Introduction:** read a CSV, show rows, use `describe()`, draw a histogram and a scatter plot.
2. **Linear models:** predict next waiting time with linear regression; classify a long wait using Bayesian logistic regression and a linear SVM. The Bayesian example tries an intercept/slope grid, combines a Gaussian prior with the training likelihood, and averages predictions using posterior weights. It is approximate Bayesian inference; it is not ordinary logistic regression renamed as Bayesian.
3. **Clustering and PCA:** K-means, Gaussian mixtures and hierarchical clustering find two groups; PCA compresses two standardized features into one. This is exploratory analysis of all complete pairs.
4. **HMM:** a two-state supervised HMM learns transition and emission tables from short/long eruption labels in training data. Test predictions use forward filtering over previous waiting observations, without revealing test eruption states. It predicts before incorporating the current test observation. Add-one smoothing avoids zero probabilities. This simple version does not use unsupervised Baum-Welch training, backward recursion or Viterbi; the assignment's required HMM prediction experiment is present.
5. **Combining models:** a depth-two CART tree, a 20-tree random forest and 20-estimator AdaBoost classify the same target. Compare actual test scores; more models need not give better accuracy.

Each notebook contains short explanations, code comments, real outputs, graphs, a small change to try, and viva questions. Read and run each cell, then explain its input, model and output in your own words.

## Open the notebooks in Jupyter

1. Install Python 3.12 or newer.
2. Run `python -m pip install -r requirements.txt` from this project folder.
3. Run `python -m jupyter lab` and open `notebooks`.
4. Choose **Restart Kernel and Run All Cells**.

Keep the `data` folder beside `notebooks`. The simple loading cell supports running from either the project root or the notebooks folder. It does not silently download a different dataset.

## Rebuild and verify

`python scripts/build_notebooks.py` recreates the CSV and five notebooks from the supplied original geyser CSV. It does not need internet. `python scripts/execute_notebooks.py` executes all five and creates `dist/ml-weekly-notebooks.zip`. `node scripts/test-browser.mjs` verifies fresh Python execution in all five website pages, graphs, editing, error recovery, stopping, downloads and mobile layout. The local Python package versions are in `environment-tested.txt`; the browser uses Pyodide 314.0.7 and can have small numerical differences.

The website contains the beginner notebooks. The previous pollution version is preserved only in an ignored local backup under `.sites-runtime`, outside the website and download pack.
