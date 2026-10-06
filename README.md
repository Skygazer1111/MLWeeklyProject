# MLWeeklyProject

**Learn machine learning by running real experiments in your browser.**

MLWeeklyProject is a practical learning website for the **21CSC305P Machine Learning** course. The website, **ML Weekly Lab**, brings together five editable Python notebooks, a real highway traffic dataset, executed reference outputs, and visual explanations. Students can explore the data, change the code, train models, and inspect results without installing Python.

Built by **[Priyan](https://portfolio-priyan.vercel.app/)**.

## Contents

- [Features](#features)
- [Course units](#course-units)
- [Technology](#technology)
- [Run the website locally](#run-the-website-locally)
- [Use the notebooks](#use-the-notebooks)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Run with Jupyter](#run-with-jupyter)
- [Regenerate notebooks and downloads](#regenerate-notebooks-and-downloads)
- [Verification](#verification)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Credits and licensing](#credits-and-licensing)

## Features

- A course overview with links to all five units and a preview of the actual dataset.
- Editable Python cells with **Run cell**, **Run all**, and **Shift + Enter** support.
- Python execution inside a browser worker through Pyodide.
- Tables, printed output, graphs, execution counts, and cell timings.
- Previously executed **Reference run** outputs that can be inspected before starting Python.
- **Stop** and **Reset kernel** controls for managing execution.
- Downloads for the edited notebook, all five reference notebooks, and the datasets.
- Direct links to individual units and browser back/forward navigation.
- Responsive cream-and-yellow styling, keyboard focus indicators, and reduced-motion support.
- A static deployment that does not require a backend or hosted notebook server.

## Course units

1. **Introduction** — Import the CSV, inspect rows and missing values, calculate summary statistics, and plot vehicle count and temperature.
2. **Linear models** — Predict the next hour's traffic volume with linear regression and classify light/heavy traffic with Bayesian logistic regression and a linear SVM.
3. **Clustering & PCA** — Explore K-means, Gaussian mixtures, and hierarchical clustering, then reduce standardized features with principal component analysis.
4. **Hidden Markov models** — Use a supervised two-state teaching HMM to explore sequential traffic prediction and compare it with a previous-hour baseline.
5. **Combining models** — Compare CART, random forest, and AdaBoost classifiers using the same next-hour target and chronological split as Unit 2.

The notebooks include commented code, executed results, graphs, experiments to try, and viva questions. Detailed modelling assumptions and notebook explanations are in [the lab documentation](ml-lab/README.md#what-each-notebook-teaches).

## Technology

- **Website:** HTML, CSS, and browser JavaScript modules.
- **Browser Python:** Pyodide and a dedicated Web Worker.
- **Scientific packages:** NumPy, pandas, SciPy, Matplotlib, scikit-learn, and IPython.
- **Local preview:** A small Node.js HTTP server.
- **Notebook generation:** Python, nbformat, and nbclient.
- **Optional local notebook interface:** JupyterLab.
- **Browser verification:** Playwright.
- **Hosting configuration:** Static output prepared for Vercel.

## Run the website locally

### Requirements

- Node.js and npm for the local preview.
- A modern browser that supports JavaScript modules, Web Workers, and WebAssembly.
- Internet access for the first browser Python run, which downloads Pyodide and scientific packages.

Python is only needed if you want to run or regenerate notebooks on your computer.

### Setup

From a terminal:

```bash
git clone https://github.com/Skygazer1111/MLWeeklyProject.git
cd MLWeeklyProject/ml-lab
npm install
npm run dev
```

Open **http://127.0.0.1:8000**. Choose **Start learning** or select a unit from the course cards.

If you already have this repository, start from `MLWeeklyProject` and run:

```bash
cd ml-lab
npm run dev
```

The preview server uses Node.js built-in modules, so installing npm dependencies is only required for browser verification. On Windows, you can also double-click [start-lab.cmd](ml-lab/start-lab.cmd).

**MLWeeklyProject is the main project.** The `ml-lab` folder holds its website, notebooks, dataset, and scripts. Make project commits from the `MLWeeklyProject` repository.

## Use the notebooks

1. Open a unit. The saved reference outputs are visible immediately.
2. Press **Run all** to initialize Python and execute cells from top to bottom. The first run may take a minute.
3. Edit a code cell, then press **Run cell** or **Shift + Enter** to try your change.
4. Inspect the new output beneath the cell. Later cells may depend on variables created by earlier cells.
5. Use **Stop** to terminate a run or **Reset kernel** to clear Python variables.
6. Press **Notebook** to download the current unit with your edits and available outputs.

**Edits live in the current browser session and are lost on refresh.** Download your notebook before leaving if you want to keep changes. The **Get notebooks** link downloads the reference pack, rather than your current edits.

Code runs in your browser. The site has no server-side Python kernel.

## Dataset

The project uses the [UCI Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume), containing hourly readings from an I-94 highway sensor between Minneapolis and Saint Paul, USA.

The supplied original dataset has **48,204 rows**. The classroom subset contains **1,008 consecutive hourly observations over 42 days**, from **13 April 2017 at 10:00** through **25 May 2017 at 09:00**, using source local timestamps.

The teaching CSV contains:

- `date_time` — The observation timestamp.
- `traffic_volume` — Vehicles counted during that hour.
- `temperature_c` — Air temperature in degrees Celsius.

Preparation validates vehicle counts at repeated timestamps, keeps the first weather entry for each hour, sorts the observations, selects a continuous six-week window, and converts Kelvin to Celsius. Missing hours are not interpolated and measurements are not invented.

Files:

- [traffic.csv](ml-lab/data/traffic.csv) — Prepared classroom subset.
- [traffic_original.csv.gz](ml-lab/data/traffic_original.csv.gz) — Original compressed source CSV.
- [metadata.json](ml-lab/data/metadata.json) — Provenance, selection rules, units, source hash, and limitations.

The dataset represents one sensor and one short time window. High vehicle counts alone do not establish congestion, and model scores do not establish performance across roads or seasons. The HMM is a teaching example; its supplied reference run includes a simpler baseline that performs better.

**Citation:** Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. DOI: [10.24432/C5X60B](https://doi.org/10.24432/C5X60B). Dataset license: **CC BY 4.0**.

## Project structure

```text
MLWeeklyProject/
├── README.md                    Project overview and setup
├── machine learning.pdf         Course assignment reference
├── vercel.json                  Serves ml-lab/dist from the repository root
└── ml-lab/
    ├── README.md                Detailed notebook and modelling notes
    ├── package.json             Local preview and browser test commands
    ├── requirements.txt         Local Python dependencies
    ├── start-lab.cmd            Windows preview launcher
    ├── vercel.json              Config for hosting with ml-lab as the root
    ├── data/                   Original and prepared data with metadata
    ├── notebooks/              Five Jupyter notebooks
    ├── dist/                   Complete static website and download assets
    │   ├── index.html          Shared home and notebook layout
    │   ├── styles.css          Responsive website styling
    │   ├── app.js              Navigation, editors, outputs, and downloads
    │   ├── python-worker.js    Browser Python initialization and execution
    │   ├── units.json          Course descriptions and notebook filenames
    │   ├── unit-1/ … unit-5/   Static entry pages for direct unit links
    │   ├── data/               Data served to the browser
    │   ├── notebooks/          Notebook copies served to the browser
    │   └── ml-weekly-notebooks.zip
    ├── scripts/
    │   ├── serve.mjs           Local static preview server
    │   ├── prepare_traffic.py  Dataset preparation
    │   ├── build_notebooks.py  Generate notebooks and synchronize assets
    │   ├── execute_notebooks.py
    │   └── test-browser.mjs    End-to-end verification
    ├── verification.json       Local notebook execution report
    └── browser-verification.json
```

The website's editable source is in `ml-lab/dist`; there is no separate frontend compilation step. Keep the five `dist/unit-*/index.html` files synchronized with `dist/index.html` when changing the shared markup.

## Run with Jupyter

Use **Python 3.12 or newer**. From `MLWeeklyProject/ml-lab`:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies and open JupyterLab:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open a file in `notebooks/` and choose **Restart Kernel and Run All Cells**. Keep `data/` beside `notebooks/` so the CSV loading code can find the supplied dataset.

## Regenerate notebooks and downloads

With the Python dependencies installed, run these commands from `ml-lab`:

```bash
python scripts/build_notebooks.py
python scripts/execute_notebooks.py
```

The first command prepares the dataset, recreates the five notebooks, and synchronizes static assets and unit entry pages. The second executes the notebooks, saves their outputs, copies them into `dist/notebooks`, creates `dist/ml-weekly-notebooks.zip`, and updates `verification.json`.

These scripts overwrite generated notebooks and download assets. Save any manual notebook edits before regeneration. Commit the resulting `dist` changes with the project before deploying.

## Verification

Keep the local preview running in one terminal:

```bash
npm run dev
```

In another terminal, from `ml-lab`, install the browser used by the test script:

```powershell
# Windows PowerShell
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD/.sites-runtime/browsers"
npx playwright install chromium
npm test
```

For macOS/Linux:

```bash
PLAYWRIGHT_BROWSERS_PATH="$PWD/.sites-runtime/browsers" npx playwright install chromium
npm test
```

The test script expects the preview at `http://127.0.0.1:8000` and browsers in `ml-lab/.sites-runtime/browsers`. Internet access is required to test fresh Pyodide execution.

Verification covers home-page navigation, course cards, the dataset preview, browser history, Python execution in all five units, graphs, tables, code editing, error recovery, stopping execution, notebook downloads, direct unit links, and responsive layouts from 320 to 1440 pixels. Screenshots are saved in the ignored `test-results/` folder; the summary is written to `browser-verification.json`. Tested local Python package versions are recorded in `environment-tested.txt`.

## Deployment

The checked-in Vercel configurations serve the complete static website:

- Repository root configuration: `outputDirectory` is **`ml-lab/dist`**.
- Configuration inside `ml-lab`: `outputDirectory` is **`dist`**.
- Both configurations have no framework preset and skip installation and building.

Use **Other** as the framework preset. For the main `MLWeeklyProject` repository, leave the Vercel Root Directory at the repository root so its configuration applies. If an existing hosting project uses `ml-lab` as its Root Directory, the configuration there serves the same website.

Deploy the commit containing your changes. No Python server, Jupyter service, or Python entrypoint is needed. Python dependencies in `requirements.txt` are for local notebook work, while browser Python packages are downloaded by Pyodide.

## Troubleshooting

- **The first run takes time:** Python and scientific packages must download before code executes. Wait for the status to change to ready.
- **Python cannot load:** Check your internet connection and whether the browser or network blocks the Pyodide CDN, then try **Run all** again.
- **A variable is missing:** Run earlier cells or use **Run all** to initialize dependencies in order.
- **Code is still running:** Press **Stop**. A subsequent run starts a fresh worker.
- **Edits disappeared:** Refreshing resets the notebook to the supplied version. Use **Notebook** to save edits before refreshing.
- **The preview cannot start:** Check whether another process already uses port 8000.
- **Playwright cannot find Chromium:** Install Chromium using the browser-path command in [Verification](#verification).
- **A deployment still shows old content:** Confirm the deployment uses the updated commit and the correct static output directory.

## Credits and licensing

- **Developer:** [Priyan](https://portfolio-priyan.vercel.app/).
- **Course:** 21CSC305P Machine Learning.
- **UI inspiration:** [HR management website UI design by Sazidur Rahman](https://dribbble.com/shots/25695355-HR-management-website-UI-design).
- **Dataset:** Hogue (2019), UCI Machine Learning Repository; licensed under CC BY 4.0.
- **Application package metadata:** Declares the MIT license in `ml-lab/package.json`.

© 2026 ML Weekly Lab. Built by [Priyan](https://portfolio-priyan.vercel.app/).
