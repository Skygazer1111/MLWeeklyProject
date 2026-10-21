"""Create five beginner notebooks from real road-traffic sensor data."""
from pathlib import Path
import json
import shutil
import textwrap
import pandas as pd
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
NB = ROOT / 'notebooks'
DIST = ROOT / 'dist'
for folder in [NB, DIST / 'data', DIST / 'notebooks']:
    folder.mkdir(parents=True, exist_ok=True)

from prepare_traffic import prepare
df = prepare(DATA)
for name in ['traffic.csv', 'traffic_original.csv.gz', 'metadata.json']:
    shutil.copy(DATA / name, DIST / 'data' / name)

def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())

def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())

SETUP = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Run from the project folder or from its notebooks folder.
try:
    df = pd.read_csv('data/traffic.csv')
except FileNotFoundError:
    df = pd.read_csv('../data/traffic.csv')

df['date_time'] = pd.to_datetime(df['date_time'])
print('Number of traffic hours:', len(df))
'''
INTRO = '''
## About this transport dataset
This is **real hourly road-sensor data** from westbound Interstate 94 between Minneapolis and Saint Paul, USA. A road sensor counts vehicles passing during each hour. We use **1,008 consecutive hours (42 days), from 13 April 2017 at 10:00 to 25 May 2017 at 09:00**, in the source's local timestamps.

**traffic_volume**: vehicles counted in that hour. **temperature_c**: air temperature in degrees Celsius. **hour**, **weekday**, and **is_weekend** are read off `date_time`. **weather_main** is the weather category (Clear, Clouds, Rain, and so on) and **clouds_pct** is cloud cover. `date_time` identifies the hour and is not itself a model input.

The original dataset has 48,204 rows and includes weather and holidays from 2012–2018. Repeated weather descriptions can share a timestamp; their traffic counts were verified identical before keeping the first entry. The teaching CSV uses the first six weeks of the longest continuous hourly sequence. Temperature was converted from Kelvin by subtracting 273.15. In this window `rain_1h`, `snow_1h`, and `holiday` are all zero or empty, so they are not columns here. No missing hours were filled and no measurements were invented.

Source: [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume). Citation: Hogue, J. (2019), DOI: 10.24432/C5X60B. License: CC BY 4.0. The complete original compressed CSV is included.

**What to tell faculty:** “I am studying traffic patterns using real vehicle counts from a highway sensor.” High vehicle count means a busy hour; without speed or capacity data it does not prove congestion. This small window is a classroom demonstration, not a validation across roads or seasons.

Run cells from top to bottom. Keep the `data` folder beside `notebooks`.
'''
PAIRS = '''
# Each row is exactly one hour after the preceding row.
data = df.copy()
data['next_volume'] = data['traffic_volume'].shift(-1)
data = data.dropna()  # The last hour has no measured following hour.

# Earlier hours teach the model; later hours test it.
split = int(len(data) * 0.8)
train = data.iloc[:split]
test = data.iloc[split:]
print('Training rows:', len(train), '| Testing rows:', len(test))
display(data.head())
'''
CLASSIFY = '''
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# A heavy hour means at least 4000 vehicles. This is a classroom cutoff.
X_train = train[['traffic_volume']]
X_test = test[['traffic_volume']]
y_train = (train['next_volume'] >= 4000).astype(int).to_numpy()
y_test = (test['next_volume'] >= 4000).astype(int).to_numpy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Class 0 = below 4000 vehicles/hour; class 1 = at least 4000 vehicles/hour')
'''
units = []

def add(number, title, aim, practices, cells):
    names = ['introduction', 'linear_models', 'clustering_pca', 'hidden_markov_models', 'combining_models']
    filename = f'unit_{number}_{names[number-1]}.ipynb'
    heading = f'# Unit {number} · {title}\n\nCourse: **21CSC305P — Machine Learning**\n\n## Aim\n{aim}\n\n## Assignment tasks\n'
    heading += '\n'.join(f'{i+1}. {item}' for i, item in enumerate(practices))
    notebook = nbf.v4.new_notebook(cells=[md(heading), md(INTRO), code(SETUP)] + cells,
        metadata={'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python', 'version': '3.12'}, 'unit': number, 'dataset': 'UCI/Metro Interstate Traffic Volume'})
    nbf.write(notebook, NB / filename)
    shutil.copy(NB / filename, DIST / 'notebooks' / filename)
    units.append({'number': number, 'title': title, 'summary': aim, 'practices': practices, 'filename': filename})

add(1, 'Introduction', 'Read real traffic data, understand vehicle count and temperature, and draw simple graphs.', [
    'Devise a program to import, load and view dataset.',
    'Create a program to display the summary and statistics of the dataset.'
], [
md('## Experiment 1 · Read and view\n`head()` shows the first five rows. `shape` gives the number of rows and columns. Each row is one hour. Vehicle count and air temperature are the measurements used in the early units; hour, weekday, weekend, weather, and cloud cover are kept so later units can study the sequence.'),
code('''
display(df.head())
print('Rows and columns:', df.shape)
print('Missing values:')
print(df.isna().sum())
'''),
md('## Experiment 2 · Summary statistics\n`describe()` gives count, mean, standard deviation, minimum, quartiles and maximum. The mean is the average; the median is the middle value.'),
code('''
display(df.describe().round(2))
print('Average traffic volume:', round(df['traffic_volume'].mean(), 2), 'vehicles/hour')
print('Median air temperature:', round(df['temperature_c'].median(), 2), 'degrees C')

plt.hist(df['traffic_volume'], bins=12, edgecolor='white')
plt.xlabel('Current volume (vehicles/hour)')
plt.ylabel('Number of hours')
plt.title('How busy are the recorded traffic hours?')
plt.show()
'''),
code('''
plt.scatter(df['temperature_c'], df['traffic_volume'], s=20)
plt.xlabel('Air temperature (degrees C)')
plt.ylabel('Current volume (vehicles/hour)')
plt.title('Temperature and traffic volume')
plt.show()
'''),
md('## What to explain\nA histogram counts observations in ranges. A scatter plot draws one dot for each row. Read the graphs before making a claim about the relationship.\n\n**Try:** Change `bins=12` to `bins=6`.\n\n**Viva:** What does one row represent? How do mean and median differ?')])

add(2, 'Linear models', 'Use current traffic volume to predict the next hour, then classify light or heavy traffic.', [
    'Implement linear regression to perform prediction.',
    'Implement Bayesian logistic regression and SVM for classification.'
], [
md('## Prepare the prediction problem\nUse this hour’s vehicle count to predict the following hour’s count. `shift(-1)` creates the target, `next_volume`. The last row is removed because its next hour is outside the window. Only current `traffic_volume` is used as input; no future count is passed to the model.'),
code(PAIRS),
md('## Experiment 1 · Linear regression\nFit a straight line: next traffic volume = slope × current traffic volume + intercept. `fit()` learns from training rows; `predict()` gives answers for new rows.'),
code('''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()
model.fit(train[['traffic_volume']], train['next_volume'])
predicted_volume = model.predict(test[['traffic_volume']])
print('Slope:', round(model.coef_[0], 2))
print('Intercept:', round(model.intercept_, 2))
print('Mean absolute error:', round(mean_absolute_error(test['next_volume'], predicted_volume), 2), 'vehicles/hour')
display(pd.DataFrame({'actual': test['next_volume'].to_numpy()[:5], 'predicted': predicted_volume[:5]}).round(2))

plt.scatter(test['traffic_volume'], test['next_volume'], label='Actual test rows')
plt.plot(test['traffic_volume'], predicted_volume, color='orange', label='Predicted line')
plt.xlabel('Current volume (vehicles/hour)')
plt.ylabel('Next volume (vehicles/hour)')
plt.legend()
plt.show()
'''),
md('## Prepare classification\nWe turn the target into two categories: **0 = below 4000 vehicles/hour**, **1 = at least 4000 vehicles/hour**. Scaling centers the input around zero. Learn the scaler from training rows only.'),
code(CLASSIFY),
md('## Experiment 2a · Bayesian logistic regression\nA logistic curve turns a number into a probability. We try a small grid of intercepts and slopes, give each a Gaussian prior, and calculate how well it explains the training labels. We then average the curves using their posterior weights. This is a simple **grid approximation to Bayesian inference**, using just one input and two parameters. The grid is finite, so the answer is approximate.'),
code('''
x_train = X_train_scaled[:, 0]
x_test = X_test_scaled[:, 0]
parameters = []
scores = []

# Try different intercepts and slopes for the logistic curve.
for intercept in np.linspace(-6, 6, 41):
    for slope in np.linspace(-6, 6, 41):
        probability = 1 / (1 + np.exp(-(intercept + slope * x_train)))
        likelihood = np.sum(y_train * np.log(probability + 1e-12)
                            + (1 - y_train) * np.log(1 - probability + 1e-12))
        prior = -(intercept**2 + slope**2) / 8  # Gaussian prior: variance 4
        parameters.append([intercept, slope])
        scores.append(likelihood + prior)

# Convert scores into posterior weights that add up to one.
weights = np.exp(np.array(scores) - np.max(scores))
weights = weights / weights.sum()
bayes_probability = np.zeros(len(x_test))
for weight, (intercept, slope) in zip(weights, parameters):
    probability = 1 / (1 + np.exp(-(intercept + slope * x_test)))
    bayes_probability += weight * probability

bayes_prediction = (bayes_probability >= 0.5).astype(int)
print('Bayesian logistic accuracy:', round(accuracy_score(y_test, bayes_prediction), 3))
'''),
code('''
order = np.argsort(test['traffic_volume'].to_numpy())
plt.scatter(test['traffic_volume'], y_test, label='Actual class', alpha=0.5)
plt.plot(test['traffic_volume'].to_numpy()[order], bayes_probability[order], color='orange', label='Probability of heavy traffic')
plt.xlabel('Current volume (vehicles/hour)')
plt.ylabel('Class / probability')
plt.legend()
plt.show()
'''),
md('## Experiment 2b · SVM\nAn SVM finds a boundary between classes. With one input and a linear kernel, the boundary is a cutoff along the current-volume axis.'),
code('''
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
svm_prediction = svm.predict(X_test_scaled)
print('SVM accuracy:', round(accuracy_score(y_test, svm_prediction), 3))
ConfusionMatrixDisplay.from_predictions(y_test, svm_prediction, display_labels=['Light traffic', 'Heavy traffic'], colorbar=False)
plt.title('SVM: actual and predicted classes')
plt.show()
'''),
md('## What to explain\nRegression predicts a number; classification predicts a category. MAE of 500 vehicles/hour means the average absolute error is 500 vehicles/hour. The Bayesian model combines a prior with training evidence and averages possible curves. The 4000 cutoff was chosen before testing and is not an official congestion limit.\n\n**Try:** Predict the next count with `model.predict(pd.DataFrame({"traffic_volume": [3000]}))`.\n\n**Viva:** What are input and target? Why preserve time order? What is a prior? Why does high volume not necessarily mean congestion?')])

add(3, 'Clustering & PCA', 'Group similar traffic hours and reduce vehicle count and temperature to one new feature.', [
    'Implement K-means clustering, mixtures of Gaussians and Hierarchical clustering algorithm to categorize data.',
    'Create a program to perform PCA.'
], [
md('## Prepare the two features\nUse measured vehicle count and air temperature for the same hour. Standardize because vehicles/hour and degrees Celsius have different scales. This unit explores all 1008 hours and does not evaluate a forecast. Do not include the timestamp or a traffic class label in clustering.'),
code('''
from sklearn.preprocessing import StandardScaler

data = df.copy()
features = data[['traffic_volume', 'temperature_c']]
scaled = StandardScaler().fit_transform(features)
display(features.head())
'''),
md('## Experiment 1 · Three ways to cluster\nK-means assigns points to the nearest center. Gaussian mixtures use probabilities. Hierarchical clustering repeatedly joins nearby groups. We choose two groups to inspect operating patterns. Inspect group means before naming them; they are not guaranteed to be light/heavy traffic classes.'),
code('''
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled)
display(features.assign(cluster=kmeans_labels).groupby('cluster').mean().round(2))
mixture = GaussianMixture(n_components=2, random_state=42)
mixture_labels = mixture.fit_predict(scaled)
hierarchy = AgglomerativeClustering(n_clusters=2)
hierarchy_labels = hierarchy.fit_predict(scaled)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].scatter(data['traffic_volume'], data['temperature_c'], c=kmeans_labels, s=15)
axes[0].set_title('K-means')
axes[1].scatter(data['traffic_volume'], data['temperature_c'], c=mixture_labels, s=15)
axes[1].set_title('Gaussian mixture')
axes[2].scatter(data['traffic_volume'], data['temperature_c'], c=hierarchy_labels, s=15)
axes[2].set_title('Hierarchical clustering')
for ax in axes:
    ax.set_xlabel('Traffic volume (vehicles/hour)')
    ax.set_ylabel('Air temperature (degrees C)')
plt.tight_layout()
plt.show()
'''),
md('## Experiment 2 · PCA\nPCA creates new features from the original ones. Here we compress two standardized features into one principal component and report how much variation it keeps.'),
code('''
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
one_feature = pca.fit_transform(scaled)
print('Original shape:', scaled.shape)
print('New shape:', one_feature.shape)
print('Variance kept:', round(pca.explained_variance_ratio_[0] * 100, 2), '%')

plt.scatter(one_feature[:, 0], np.zeros(len(one_feature)), c=kmeans_labels, s=15)
plt.xlabel('Principal component 1')
plt.yticks([])
plt.title('The two features compressed to one')
plt.show()
'''),
md('## What to explain\nClustering does not need answer labels. Cluster 0 and cluster 1 are just names and can swap between methods. PCA reduces the number of features; it does not create a class label.\n\n**Try:** Change `n_components=1` to `2` and compare the new shape.\n\n**Viva:** Why scale vehicle count and temperature? What is `n_init`? How does GMM differ from K-means? What is Ward linkage? Is variance kept an accuracy score? How is clustering different from classification?')])

add(4, 'Hidden Markov models', 'Follow one traffic sequence through Markov chains, hidden regimes, forward–backward, Viterbi, and a Kalman filter.', [
    'Implement HMM to predict the sequential data.'
], [
md('''
## How to present this unit
Every section is one syllabus idea, one picture, and one sentence you can say out loud. The sequence stays in hourly order. The first 80% of hours are for learning. The last 20% are the test. Grey bands and clock labels are the answer key: the test algorithms do not receive them.

**Say this:** “I am not treating the rows as a bag of independent hours. The road has memory, so the model is a chain.”
'''),
code('''
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

INK, CREAM, ACCENT = '#1c1c18', '#fffbea', '#ffda38'
OBS_COLORS = ['#6f93c2', '#e0b12a', '#e07a3d']
REGIME_COLORS = ['#2f3b56', '#7fbfb0', '#e07a3d']
WEATHER_COLORS = {
    'Clear': '#ffda38', 'Clouds': '#8aa0b4', 'Rain': '#3d6ea8', 'Mist': '#b7c4c4',
    'Drizzle': '#6f92b8', 'Thunderstorm': '#5c4d7a', 'Haze': '#c4b49a', 'Snow': '#dfe7ee', 'Fog': '#9aa0a0'
}
LEVELS = ['Low', 'Medium', 'High']
REGIMES = ['Night', 'Off-peak', 'Rush']
plt.rcParams.update({
    'figure.facecolor': CREAM, 'axes.facecolor': CREAM, 'savefig.facecolor': CREAM,
    'axes.edgecolor': '#e4e0d4', 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'text.color': INK, 'font.size': 10
})
heat = LinearSegmentedColormap.from_list('lab', ['#fffbea', '#ffda38', '#c47b12', '#1c1c18'])

data = df.copy()
split = int(len(data) * 0.8)
hour = data['hour'].to_numpy()
volume = data['traffic_volume'].to_numpy(dtype=float)
low_cut, high_cut = np.quantile(volume[:split], [1 / 3, 2 / 3])
observations = np.digitize(volume, [low_cut, high_cut])
# Night: before 06:00 or from 21:00. Rush: 07:00-09:00 and 16:00-18:00. Everything else is off-peak.
states = np.ones(len(hour), dtype=int)
states[(hour < 6) | (hour >= 21)] = 0
states[((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18))] = 2
day0 = int(np.where((hour == 0) & (np.arange(len(hour)) >= split))[0][0])
assert day0 + 72 < len(data)
print('Training hours:', split, '| Test hours:', len(data) - split)
print('Low below', round(low_cut), '| medium until', round(high_cut), '| high from', round(high_cut), 'vehicles/hour')
print('Cuts use the training hours only. Test pictures start at', data['date_time'].iloc[day0])
display(data.head())
'''),
md('''
## 1 · Sequential data
A row only makes sense next to the rows around it. Traffic repeats every day, weekdays differ from weekends, and weather sits on top of that rhythm.

**Say this:** “Before any model, the pictures show daily memory, a weekly pattern, and a weather effect.”
'''),
code('''
fig, ax = plt.subplots(figsize=(8.4, 3.6), layout='constrained')
for flag, name, color in [(0, 'Weekday', INK), (1, 'Weekend', '#c47b12')]:
    means = data[data['is_weekend'] == flag].groupby('hour')['traffic_volume'].mean()
    ax.plot(means.index, means.to_numpy(), marker='o', ms=3.5, color=color, label=name)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('Hour of day')
ax.set_ylabel('Average vehicles/hour')
ax.set_title('The same clock hour is a different road on the weekend')
ax.legend(frameon=False)
plt.show()
'''),
code('''
fig, ax = plt.subplots(figsize=(10.4, 3.8), layout='constrained')
times = data['date_time']
night = ((hour < 6) | (hour >= 21)).astype(int)
edges = np.diff(np.concatenate([[0], night, [0]]))
for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
    ax.axvspan(times.iloc[start], times.iloc[end] + pd.Timedelta(hours=1), color='#e4e0d4', lw=0)
ax.plot(times, volume, color=INK, lw=0.45, alpha=0.45, zorder=2)
for name, color in WEATHER_COLORS.items():
    mask = data['weather_main'].to_numpy() == name
    if mask.any():
        ax.scatter(times[mask], volume[mask], s=8, c=color, label=name, zorder=3)
ax.set_ylabel('Vehicles/hour')
ax.set_title('42 days in order — grey bands are night')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, frameon=False, fontsize=7, loc='outside lower center')
plt.show()
'''),
code('''
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot = data.pivot_table(index='hour', columns='weekday', values='traffic_volume', aggfunc='mean')[order]
fig, ax = plt.subplots(figsize=(9.2, 3.5), layout='constrained')
image = ax.imshow(pivot.to_numpy().T, aspect='auto', cmap=heat)
ax.set_yticks(range(7), order)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('Hour of day')
ax.set_title('Average traffic: hour against weekday')
fig.colorbar(image, ax=ax, label='Vehicles/hour', fraction=0.03)
plt.show()

centered = volume - volume.mean()
acf = np.correlate(centered, centered, mode='full')[len(centered) - 1:len(centered) + 168]
acf = acf / acf[0]
fig, ax = plt.subplots(figsize=(8.6, 3.4), layout='constrained')
ax.bar(range(len(acf)), acf, color=ACCENT, edgecolor=INK, width=1)
for lag, name in [(24, '24 h'), (168, '168 h')]:
    ax.axvline(lag, color='#c47b12', lw=1.2)
    ax.text(lag + 2, 0.92, name, color='#c47b12')
ax.set_ylim(-0.4, 1.05)
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Memory peaks one day later, and again one week later')
plt.show()
'''),
code('''
weather_order = data.groupby('weather_main')['traffic_volume'].median().sort_values(ascending=False).index
groups = [data.loc[data['weather_main'] == name, 'traffic_volume'] for name in weather_order]
fig, ax = plt.subplots(figsize=(8.6, 3.6), layout='constrained')
ax.boxplot(groups, tick_labels=list(weather_order), patch_artist=True,
           boxprops={'facecolor': ACCENT, 'edgecolor': INK}, medianprops={'color': INK})
plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
ax.set_ylabel('Vehicles/hour')
ax.set_title('Weather category and how busy the hour is')
plt.show()

fig, ax = plt.subplots(figsize=(8.2, 3.6), layout='constrained')
ax.scatter(data['clouds_pct'], volume, c=hour, cmap=heat, s=12)
ax.set_xlabel('Cloud cover (%)')
ax.set_ylabel('Vehicles/hour')
ax.set_title('Cloud cover, coloured by hour of day')
fig.colorbar(ax.collections[0], ax=ax, label='Hour')
plt.show()
'''),
md('''
**Say this:** “The 24-hour spike is why a Markov model can beat a model that ignores order. Rain and cloud cover change the level, but the clock is the stronger pattern. `rain_1h`, `snow_1h`, and `holiday` are empty in this window, so I use weather category and cloud cover.”

## 2 · Markov models
Collapse the count into Low, Medium, and High using the training tertiles. A Markov chain says the next level depends on the current level only: the past matters through the latest observation.

**Say this:** “The matrix is the whole model. The diagonal is the chance that the road stays in the same level.”
'''),
code('''
A_obs = np.ones((3, 3))
for i in range(split - 1):
    A_obs[observations[i], observations[i + 1]] += 1
A_obs = A_obs / A_obs.sum(axis=1, keepdims=True)
display(pd.DataFrame(A_obs, index=LEVELS, columns=['Next ' + name for name in LEVELS]).round(3))

fig, ax = plt.subplots(figsize=(5.4, 4.2), layout='constrained')
image = ax.imshow(A_obs, vmin=0, vmax=1, cmap=heat)
ax.set_xticks(range(3), ['Next ' + name for name in LEVELS])
ax.set_yticks(range(3), LEVELS)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{A_obs[i, j]:.2f}', ha='center', va='center', color='white' if A_obs[i, j] > 0.62 else INK)
fig.colorbar(image, ax=ax, fraction=0.046, label='Probability')
ax.set_title('Observed Markov transition matrix')
plt.show()
'''),
code('''
pos = np.array([[0.0, 0.0], [2.3, 0.0], [1.15, 1.85]])
fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
ax.set_xlim(-0.8, 3.1)
ax.set_ylim(-0.9, 2.7)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Read the chain as a map, not only a table')
for i, name in enumerate(LEVELS):
    ax.add_patch(Circle(pos[i], 0.42, facecolor=OBS_COLORS[i], edgecolor=INK, lw=1.2, zorder=3))
    ax.text(pos[i, 0], pos[i, 1] + 0.08, name, ha='center', va='center', fontsize=8, zorder=4)
    ax.text(pos[i, 0], pos[i, 1] - 0.14, f'stay {A_obs[i, i]:.2f}', ha='center', va='center', fontsize=7, zorder=4)
for i in range(3):
    for j in range(3):
        if i == j:
            continue
        delta = pos[j] - pos[i]
        unit = delta / np.linalg.norm(delta)
        normal = np.array([-unit[1], unit[0]])
        ax.add_patch(FancyArrowPatch(pos[i] + unit * 0.5, pos[j] - unit * 0.5, arrowstyle='-|>',
            mutation_scale=11, connectionstyle='arc3,rad=0.16', color=INK, lw=0.5 + 2.4 * A_obs[i, j], zorder=2))
        ax.text(*((pos[i] + pos[j]) / 2 + normal * 0.2), f'{A_obs[i, j]:.2f}', ha='center', va='center', fontsize=8)
plt.show()
'''),
code('''
rng = np.random.default_rng(0)
simulated = np.zeros(72, dtype=int)
simulated[0] = observations[split - 1]
for t in range(1, 72):
    simulated[t] = rng.choice(3, p=A_obs[simulated[t - 1]])
real = observations[split:split + 72]
fig, axes = plt.subplots(2, 1, figsize=(9.4, 4.0), sharex=True, sharey=True, layout='constrained')
axes[0].step(range(72), real, where='mid', color=INK)
axes[1].step(range(72), simulated, where='mid', color='#c47b12')
axes[0].set_title('Real test levels')
axes[1].set_title('A chain simulated from the same matrix')
for ax in axes:
    ax.set_yticks([0, 1, 2], LEVELS)
axes[1].set_xlabel('Hour into the test window')
plt.show()

stationary = np.ones(3) / 3
for _ in range(40):
    stationary = stationary @ A_obs
empirical = np.bincount(observations[:split], minlength=3) / split
fig, ax = plt.subplots(figsize=(6.4, 3.5), layout='constrained')
x = np.arange(3)
ax.bar(x - 0.16, empirical, width=0.32, color=ACCENT, edgecolor=INK, label='Seen in training')
ax.bar(x + 0.16, stationary, width=0.32, color='#8aa0b4', edgecolor=INK, label='Stationary distribution')
ax.set_xticks(x, LEVELS)
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')
ax.set_title('After many steps the chain forgets where it started')
ax.legend(frameon=False)
plt.show()
'''),
md('''
**Say this:** “The simulated strip has the same kind of runs as the real road, because staying put is likely. The two bars match: the long-run mix of levels is already visible in the data.”

## 3 · The hidden Markov idea
A Markov chain models the level we see. An HMM adds a hidden regime. The regime follows its own Markov chain, and the traffic level is emitted by the current regime. We see the vehicles. We do not see the regime during the test.

**Say this:** “Daytime is not the same thing as heavy traffic. The regime causes the observation; it is not the observation.”
'''),
code('''
fig, ax = plt.subplots(figsize=(9.2, 3.3), layout='constrained')
ax.set_xlim(-0.5, 4.6)
ax.set_ylim(-0.7, 1.8)
ax.axis('off')
ax.set_title('Hidden chain on top, observations underneath')
for t in range(5):
    ax.add_patch(Circle((t, 1), 0.24, facecolor=ACCENT, edgecolor=INK, lw=1.2, zorder=3))
    ax.text(t, 1, 'z' + str(t + 1), ha='center', va='center', zorder=4)
    ax.add_patch(Circle((t, 0), 0.24, facecolor='white', edgecolor=INK, lw=1.2, zorder=3))
    ax.text(t, 0, 'x' + str(t + 1), ha='center', va='center', zorder=4)
    ax.annotate('', xy=(t, 0.28), xytext=(t, 0.74), arrowprops={'arrowstyle': '->', 'color': INK})
    if t:
        ax.annotate('', xy=(t - 0.28, 1), xytext=(t - 0.72, 1), arrowprops={'arrowstyle': '->', 'color': INK})
ax.text(2, 1.55, 'Hidden regime  z', ha='center')
ax.text(2, -0.5, 'Observed traffic level  x', ha='center')
plt.show()
'''),
md('''
## 4 · Maximum likelihood for the HMM
Two ways to learn the tables.

**Supervised MLE.** On the training hours the clock tells us the regime: Night, Off-peak, or Rush. Count pairs and add one so no probability is zero. `A[i, j]` is how often regime i is followed by regime j. `B[i, j]` is how often regime i produces level j.

**Unsupervised MLE is Baum–Welch, an EM algorithm.** Hide the clock. The E-step runs forward–backward to get gamma, the responsibility for one state, and xi, the responsibility for a pair. The M-step re-estimates the tables from those responsibilities. Repeat. The training log-likelihood should rise.

**Say this:** “Counting is maximum likelihood when the regimes are labelled. EM is maximum likelihood when they are not. I sort the learned states by average volume afterwards; the algorithm itself never saw those names.”
'''),
code('''
def hmm_pass(obs, pi, A, B):
    # Scaled forward-backward. c_t keeps each row of alpha at a sum of 1.
    obs = np.asarray(obs)
    T, K = len(obs), len(pi)
    alpha = np.zeros((T, K))
    c = np.zeros(T)
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]
    beta = np.zeros((T, K))
    beta[-1] = 1
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
        beta[t] /= c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    emit_beta = B[:, obs[1:]].T * beta[1:]
    numer = alpha[:-1, :, None] * A[None, :, :] * emit_beta[:, None, :]
    xi = numer / numer.sum(axis=(1, 2), keepdims=True)
    return alpha, beta, gamma, xi, c, float(np.sum(np.log(c)))

def viterbi(obs, pi, A, B):
    obs = np.asarray(obs)
    T, K = len(obs), len(pi)
    delta = np.empty((T, K))
    psi = np.empty((T, K), dtype=int)
    delta[0] = np.log(pi) + np.log(B[:, obs[0]])
    log_A = np.log(A)
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores.max(axis=0) + np.log(B[:, obs[t]])
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, delta

def baum_welch(obs, n_iter=20, seed=7):
    rng = np.random.default_rng(seed)
    obs = np.asarray(obs)
    pi = np.full(3, 1 / 3)
    A = rng.random((3, 3))
    A /= A.sum(1, keepdims=True)
    B = rng.random((3, 3))
    B /= B.sum(1, keepdims=True)
    history = []
    for _ in range(n_iter):
        _, _, gamma, xi, _, ll = hmm_pass(obs, pi, A, B)
        history.append(ll)
        pi = gamma[0] / gamma[0].sum()
        A = xi.sum(0) / np.maximum(gamma[:-1].sum(0)[:, None], 1e-12)
        A /= A.sum(1, keepdims=True)
        B = np.vstack([np.array([gamma[obs == m, k].sum() for m in range(3)]) for k in range(3)])
        B = np.maximum(B, 1e-12)
        B /= B.sum(1, keepdims=True)
    return pi, A, B, np.array(history)

pi_sup = np.ones(3)
pi_sup[states[0]] += 1
pi_sup = pi_sup / pi_sup.sum()
A_sup = np.ones((3, 3))
B_sup = np.ones((3, 3))
for i in range(split - 1):
    A_sup[states[i], states[i + 1]] += 1
for i in range(split):
    B_sup[states[i], observations[i]] += 1
A_sup /= A_sup.sum(1, keepdims=True)
B_sup /= B_sup.sum(1, keepdims=True)
display(pd.DataFrame(A_sup, index=REGIMES, columns=['Next ' + name for name in REGIMES]).round(3))
display(pd.DataFrame(B_sup, index=REGIMES, columns=LEVELS).round(3))

for matrix, ylabels, xlabels, title, ylabel in [
    (A_sup, REGIMES, ['Next ' + name for name in REGIMES], 'Supervised transitions A', 'Current regime'),
    (B_sup, REGIMES, LEVELS, 'Supervised emissions B', 'Hidden regime')]:
    fig, ax = plt.subplots(figsize=(5.6, 4.1), layout='constrained')
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap=heat)
    ax.set_xticks(range(3), xlabels)
    ax.set_yticks(range(3), ylabels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white' if matrix[i, j] > 0.62 else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, label='Probability')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.show()
'''),
code('''
pi_em, A_em, B_em, history = baum_welch(observations[:split])
print('Baum-Welch log-likelihood, first to last:', round(history[0], 2), '->', round(history[-1], 2))
assert history[-1] >= history[0] - 1e-3
fig, ax = plt.subplots(figsize=(7.2, 3.4), layout='constrained')
ax.plot(np.arange(1, len(history) + 1), history, marker='o', color=INK)
ax.set_xlabel('EM iteration')
ax.set_ylabel('Training log-likelihood')
ax.set_title('Each M-step fits the responsibilities from the E-step')
plt.show()

train_volume = volume[:split]
path, _ = viterbi(observations[:split], pi_em, A_em, B_em)
means = np.array([train_volume[path == k].mean() if np.any(path == k) else 0 for k in range(3)])
order_states = np.argsort(means)
pi_em = pi_em[order_states]
A_em = A_em[np.ix_(order_states, order_states)]
B_em = B_em[order_states]
state_names = ['Quieter', 'Middle', 'Busier']
rank = np.empty_like(order_states)
rank[order_states] = np.arange(3)
path = rank[path]

fig, ax = plt.subplots(figsize=(6.6, 3.6), layout='constrained')
bottom = np.zeros(3)
for level, color in enumerate(OBS_COLORS):
    ax.bar(state_names, B_em[:, level], bottom=bottom, color=color, edgecolor=INK, label=LEVELS[level])
    bottom += B_em[:, level]
ax.set_ylim(0, 1)
ax.set_ylabel('Emission probability')
ax.set_title('What each learned hidden state tends to emit')
ax.legend(frameon=False, ncol=3)
plt.show()

hour_state = np.zeros((24, 3))
for h, state in zip(hour[:split], path):
    hour_state[h, state] += 1
hour_state /= np.maximum(hour_state.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(9.2, 3.2), layout='constrained')
image = ax.imshow(hour_state.T, aspect='auto', cmap=heat, vmin=0, vmax=1)
ax.set_yticks(range(3), state_names)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('Hour of day')
ax.set_title('No clock was given, but the hidden states rediscover the day')
fig.colorbar(image, ax=ax, fraction=0.03, label='Share of that hour')
plt.show()
'''),
md('''
## 5 · Forward and backward
Run this on the test week. The start belief is carried out of training, then the clock is hidden.

Alpha is the filtered belief: which regime is likely given the past and the present. Beta is the backward message: how well the future fits each regime. Gamma is proportional to alpha times beta, and it is the smoothed belief given the whole displayed week.

**Say this:** “Filtering answers where am I now? Smoothing answers where was I, now that I have seen the rest of the week? Smoothing is smoother because it is allowed to look ahead.”
'''),
code('''
alpha_train, _, _, _, _, _ = hmm_pass(observations[:day0], pi_sup, A_sup, B_sup)
pi_window = alpha_train[-1] @ A_sup
window = observations[day0:day0 + 72]
alpha, beta, gamma, _, _, _ = hmm_pass(window, pi_window, A_sup, B_sup)
fig, axes = plt.subplots(3, 1, figsize=(9.6, 5.4), sharex=True, layout='constrained')
for ax, matrix, title in zip(axes, [alpha, beta, gamma], [
        'Forward alpha — filtered belief',
        'Backward beta — message from the future',
        'Smoothed gamma — proportional to alpha times beta']):
    image = ax.imshow(matrix.T, aspect='auto', cmap=heat, vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(3), REGIMES)
    ax.set_title(title)
axes[-1].set_xlabel('Hour into the test week')
fig.colorbar(image, ax=list(axes), fraction=0.02, label='Weight')
plt.show()

fig, ax = plt.subplots(figsize=(9.2, 3.5), layout='constrained')
clock = states[day0:day0 + 72] == 2
ax.fill_between(range(72), 0, clock, color='#e07a3d', alpha=0.18, label='Clock says Rush')
ax.plot(alpha[:, 2], color=INK, lw=1.8, label='Filtered P(Rush)')
ax.plot(gamma[:, 2], color='#c47b12', lw=1.8, label='Smoothed P(Rush)')
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Hour into the test week')
ax.set_ylabel('Probability')
ax.set_title('Smoothing uses the whole week; filtering cannot')
ax.legend(frameon=False, ncol=3)
plt.show()
'''),
md('''
## 6 · The sum-product algorithm
On a chain, forward–backward is the sum-product algorithm. Alpha is the message from the past. Beta is the message from the future. Their product is the marginal. For the first 8 test hours the notebook also sums the brute-force probability of all 3^8 = 6561 hidden paths. The dots should sit on the lines.

Xi is the pairwise marginal, the expected share of each hidden transition. That is the quantity EM uses to re-estimate A.

**Say this:** “Sum-product does not list the paths. It passes messages along the edges and gets the same marginals.”
'''),
code('''
short = observations[day0:day0 + 8]
_, _, gamma_short, xi_short, _, _ = hmm_pass(short, pi_window, A_sup, B_sup)
codes = np.arange(3 ** 8)
paths = np.stack([(codes // 3 ** t) % 3 for t in range(8)], axis=1)
logp = np.log(pi_window[paths[:, 0]]) + np.log(B_sup[paths[:, 0], short[0]])
for t in range(1, 8):
    logp += np.log(A_sup[paths[:, t - 1], paths[:, t]]) + np.log(B_sup[paths[:, t], short[t]])
weights = np.exp(logp - logp.max())
weights /= weights.sum()
brute = np.zeros((8, 3))
for t in range(8):
    for k in range(3):
        brute[t, k] = weights[paths[:, t] == k].sum()
gap = np.max(np.abs(brute - gamma_short))
print('Largest gap between all 6561 paths and sum-product:', f'{gap:.2e}')
assert gap < 1e-6

fig, ax = plt.subplots(figsize=(8.4, 3.6), layout='constrained')
for k, name in enumerate(REGIMES):
    ax.plot(range(8), gamma_short[:, k], color=REGIME_COLORS[k], lw=2, label='Sum-product ' + name)
    ax.scatter(range(8), brute[:, k], color=REGIME_COLORS[k], zorder=3)
ax.set_xticks(range(8))
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Hour in the 8-hour check')
ax.set_ylabel('P(regime)')
ax.set_title('Dots are brute force; lines are message passing')
ax.legend(frameon=False, ncol=3)
plt.show()

pair = xi_short.mean(axis=0)
fig, ax = plt.subplots(figsize=(5.3, 4.1), layout='constrained')
image = ax.imshow(pair, vmin=0, vmax=pair.max(), cmap=heat)
ax.set_xticks(range(3), ['To ' + name for name in REGIMES])
ax.set_yticks(range(3), ['From ' + name for name in REGIMES])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{pair[i, j]:.2f}', ha='center', va='center', color='white' if pair[i, j] > pair.max() * 0.62 else INK)
fig.colorbar(image, ax=ax, fraction=0.046, label='Average xi')
ax.set_title('Pairwise messages over these 8 hours')
plt.show()
'''),
md('''
## 7 · Scaling factors
Multiplying a long run of probabilities underflows toward zero in ordinary floating point. At each step, `c_t` is the sum of the unnormalised forward values, and we divide by it. The scaled alpha stays a probability. The log-likelihood is the sum of `log c_t`, and it matches a separate log-space forward pass.

**Say this:** “Scaling does not change the answer. It stops the computer from rounding the entire belief to zero.”
'''),
code('''
_, _, _, _, c_all, ll_scaled = hmm_pass(observations, pi_sup, A_sup, B_sup)
log_alpha = np.empty((len(observations), 3))
log_alpha[0] = np.log(pi_sup) + np.log(B_sup[:, observations[0]])
log_A = np.log(A_sup)
for t in range(1, len(observations)):
    log_alpha[t] = np.logaddexp.reduce(log_alpha[t - 1][:, None] + log_A, axis=0) + np.log(B_sup[:, observations[t]])
ll_logspace = float(np.logaddexp.reduce(log_alpha[-1]))
print('sum of log c =', round(ll_scaled, 4))
print('log-space forward =', round(ll_logspace, 4))
assert abs(ll_scaled - ll_logspace) < 1e-4

raw = np.empty(len(observations))
acc = pi_sup * B_sup[:, observations[0]]
raw[0] = acc.sum()
for t in range(1, len(observations)):
    acc = (acc @ A_sup) * B_sup[:, observations[t]]
    raw[t] = acc.sum()
if np.any(raw == 0):
    print('Float64 unscaled mass hits exact 0 at hour', int(np.argmax(raw == 0)))
else:
    print('Float64 mass stays positive. The log10 curve still shows the collapse toward the float64 limit.')

fig, ax = plt.subplots(figsize=(9.2, 3.5), layout='constrained')
ax.plot(log_alpha.max(axis=1) / np.log(10), color=INK, label='log10 of unscaled forward peak')
ax.axhline(-308, color='#c47b12', ls='--', label='Approximate float64 limit')
ax.set_xlabel('Hour in the full sequence')
ax.set_ylabel('log10 mass')
ax.set_title('Without scaling the forward mass falls by hundreds of orders')
ax.legend(frameon=False)
plt.show()

fig, ax = plt.subplots(figsize=(9.2, 3.4), layout='constrained')
ax.plot(c_all, color=INK, lw=0.8)
ax.set_xlabel('Hour')
ax.set_ylabel('Scaling factor c')
ax.set_title('c_t is the probability of the new observation given the past')
plt.show()
'''),
md('''
## 8 · Viterbi
Forward–backward sums over paths. Viterbi keeps the single best path. In logs, the score of a node is its emission plus the best incoming transition. Pointers remember which predecessor won, and the end of the chain walks back from the best final state.

**Say this:** “The colour is the winning score at that node. The white path is the decoded story of the day. Picking the best state at each hour separately can jump; Viterbi picks one story that holds together.”
'''),
code('''
path24, delta = viterbi(observations[day0:day0 + 24], pi_window, A_sup, B_sup)
fig, ax = plt.subplots(figsize=(9.4, 3.6), layout='constrained')
image = ax.imshow(delta.T, aspect='auto', cmap=heat, interpolation='nearest')
ax.plot(range(24), path24, marker='o', color='white', markeredgecolor=INK, lw=2, label='Viterbi path')
ax.set_yticks(range(3), REGIMES)
ax.set_xticks(range(0, 24, 3), hour[day0:day0 + 24][::3])
ax.set_xlabel('Hour of the first test day')
ax.set_title('Trellis for one day, started from the training belief')
fig.colorbar(image, ax=ax, fraction=0.03, label='Log score')
ax.legend(frameon=False, loc='upper right')
plt.show()

decoded, _ = viterbi(window, pi_window, A_sup, B_sup)
posterior = np.argmax(gamma, axis=1)
strips = [window, decoded, posterior, states[day0:day0 + 72]]
names = ['Observed level', 'Viterbi path', 'Best state each hour', 'Clock regime']
palettes = [OBS_COLORS, REGIME_COLORS, REGIME_COLORS, REGIME_COLORS]
fig, axes = plt.subplots(4, 1, figsize=(9.6, 3.6), sharex=True, layout='constrained')
for ax, row, name, palette in zip(axes, strips, names, palettes):
    ax.imshow(np.asarray(row)[None, :], aspect='auto', interpolation='nearest',
              cmap=ListedColormap(palette), vmin=-0.5, vmax=2.5)
    ax.set_yticks([0])
    ax.set_yticklabels([name])
axes[-1].set_xlabel('Hour into the test week')
fig.suptitle('Four readings of the same 72 hours')
plt.show()
'''),
md('''
## 9 · Predict the next hour
This is the practice task. From the current belief, step the hidden chain forward with A, then emit with B. Record the most likely next level before seeing it. Two HMMs are scored: the supervised regime model, and the Baum–Welch model that never saw the clock. The previous hour and the training majority are the honest baselines.

**Say this:** “A busy hour usually stays busy, so copying the previous level is hard to beat. The HMM is still the model that explains why, through a hidden regime.”
'''),
code('''
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def forecast(pi, A, B):
    belief = pi.astype(float).copy()
    preds, probs = [], []
    for i, obs in enumerate(observations):
        nxt = belief if i == 0 else belief @ A
        emitted = nxt @ B
        if i >= split:
            preds.append(int(np.argmax(emitted)))
            probs.append(emitted)
        belief = nxt * B[:, obs]
        belief /= belief.sum()
    return np.array(preds), np.array(probs)

pred_sup, prob_sup = forecast(pi_sup, A_sup, B_sup)
pred_em, _ = forecast(pi_em, A_em, B_em)
actual = observations[split:]
previous = observations[split - 1:-1]
majority = int(np.bincount(observations[:split]).argmax())
scores = {
    'Supervised HMM': accuracy_score(actual, pred_sup),
    'HMM without clock': accuracy_score(actual, pred_em),
    'Previous hour': accuracy_score(actual, previous),
    'Always majority': accuracy_score(actual, np.full_like(actual, majority)),
}
for name, score in scores.items():
    print(name + ':', round(score, 3))

fig, ax = plt.subplots(figsize=(9.2, 3.4), layout='constrained')
ax.step(range(48), actual[:48], where='mid', color=INK, label='Actual level')
ax.step(range(48), pred_sup[:48], where='mid', color='#c47b12', label='Supervised HMM')
ax.set_yticks([0, 1, 2], LEVELS)
ax.set_xlabel('Test hour')
ax.set_title('Forecast made before the hour is revealed')
ax.legend(frameon=False)
plt.show()

ConfusionMatrixDisplay.from_predictions(actual, pred_sup, display_labels=LEVELS, cmap=heat, colorbar=False)
plt.title('Supervised HMM: rows are actual, columns are predicted')
plt.show()

fig, ax = plt.subplots(figsize=(7.4, 3.5), layout='constrained')
names = list(scores)
values = list(scores.values())
ax.bar(names, values, color=[ACCENT, '#e07a3d', '#8aa0b4', '#d9d6cc'], edgecolor=INK)
ax.set_ylim(0, 1)
for i, value in enumerate(values):
    ax.text(i, value + 0.03, f'{value:.2f}', ha='center')
plt.setp(ax.get_xticklabels(), rotation=12, ha='right')
ax.set_ylabel('Test accuracy')
ax.set_title('HMM forecasts against the baselines')
plt.show()

fig, ax = plt.subplots(figsize=(9.4, 3.5), layout='constrained')
ax.stackplot(range(len(prob_sup)), prob_sup.T, colors=OBS_COLORS, labels=LEVELS, alpha=0.95)
ax.set_ylim(0, 1)
ax.set_xlabel('Test hour')
ax.set_ylabel('Forecast probability')
ax.set_title('The ribbon is the full forecast, not only the winning level')
ax.legend(frameon=False, ncol=3, loc='upper right')
plt.show()
'''),
md('''
## 10 · Linear dynamical systems
An HMM with a continuous hidden state is a linear dynamical system. `x_t` is the latent traffic level and `y_t` is the counted vehicles.

The level follows the hour-of-week average, plus a fraction of the previous deviation, plus noise. The count is that latent level plus measurement noise. The hour-of-week average is learned from training only. The Kalman filter updates the latent level one hour at a time. The RTS smoother then walks backward, which is the continuous version of forward–backward. The Kalman gain is the weight given to the new measurement.

**Say this:** “When the gain is high, I trust this hour’s count. When it is low, I trust the prediction. The band is the filter’s uncertainty, not a promise about next week.”
'''),
code('''
how = data['date_time'].dt.dayofweek.to_numpy() * 24 + hour
train_how = how[:split]
train_volume = volume[:split]
profile = np.array([
    train_volume[train_how == slot].mean() if np.any(train_how == slot) else train_volume.mean()
    for slot in range(168)
])
mu = profile[how]
residual = train_volume - mu[:split]
phi = float(np.dot(residual[:-1], residual[1:]) / np.dot(residual[:-1], residual[:-1]))
innovation = residual[1:] - phi * residual[:-1]
q = float(np.var(innovation) * 0.5)
R = float(np.var(innovation) * 0.5)
print('AR coefficient phi =', round(phi, 3))
print('Process variance', round(q), '| measurement variance', round(R))

T = len(volume)
x_pred = np.zeros(T)
P_pred = np.zeros(T)
x_filt = np.zeros(T)
P_filt = np.zeros(T)
gain = np.zeros(T)
x_pred[0] = mu[0]
P_pred[0] = float(np.var(residual))
for t in range(T):
    gain[t] = P_pred[t] / (P_pred[t] + R)
    x_filt[t] = x_pred[t] + gain[t] * (volume[t] - x_pred[t])
    P_filt[t] = (1 - gain[t]) * P_pred[t]
    if t + 1 < T:
        x_pred[t + 1] = mu[t + 1] + phi * (x_filt[t] - mu[t])
        P_pred[t + 1] = phi ** 2 * P_filt[t] + q
x_smooth = x_filt.copy()
P_smooth = P_filt.copy()
for t in range(T - 2, -1, -1):
    C = P_filt[t] * phi / P_pred[t + 1]
    x_smooth[t] = x_filt[t] + C * (x_smooth[t + 1] - x_pred[t + 1])
    P_smooth[t] = P_filt[t] + C ** 2 * (P_smooth[t + 1] - P_pred[t + 1])

idx = np.arange(split, T)
band = 2 * np.sqrt(P_filt[idx])
fig, ax = plt.subplots(figsize=(9.8, 3.8), layout='constrained')
ax.fill_between(data['date_time'].iloc[idx], x_filt[idx] - band, x_filt[idx] + band, color=ACCENT, alpha=0.55, label='Filter ± 2 sigma')
ax.plot(data['date_time'].iloc[idx], volume[idx], color=INK, lw=1, label='Counted vehicles')
ax.plot(data['date_time'].iloc[idx], x_filt[idx], color='#c47b12', lw=1.6, label='Kalman filter')
ax.plot(data['date_time'].iloc[idx], x_smooth[idx], color='#2f3b56', lw=1.4, label='RTS smoother')
ax.set_ylabel('Vehicles/hour')
ax.set_title('Continuous hidden state on the test hours')
ax.legend(frameon=False, ncol=2)
plt.show()

fig, ax = plt.subplots(figsize=(8.6, 3.3), layout='constrained')
ax.plot(gain, color=INK)
ax.axvline(split, color='#c47b12', ls='--', label='Test starts')
ax.set_ylabel('Kalman gain')
ax.set_xlabel('Hour')
ax.set_title('Gain is the weight on the new vehicle count')
ax.legend(frameon=False)
plt.show()

naive = np.abs(volume[idx] - volume[idx - 1]).mean()
kalman_mae = np.abs(volume[idx] - x_pred[idx]).mean()
print('One-step MAE — Kalman', round(kalman_mae), 'vehicles/hour; previous hour', round(naive))
fig, ax = plt.subplots(figsize=(5.6, 3.4), layout='constrained')
ax.bar(['Kalman one-step', 'Previous hour'], [kalman_mae, naive], color=[ACCENT, '#8aa0b4'], edgecolor=INK)
ax.set_ylabel('Mean absolute error')
ax.set_title('Continuous forecast on the test hours')
plt.show()
'''),
md('''
## 11 · What to say, and the viva
Walk the pictures in this order. Do not claim the HMM beats copying the previous hour unless the bar chart says so.

1. Timeline, weekend curves, and the 24-hour autocorrelation: the road has memory.
2. Markov matrix and state map: the next level depends on this level.
3. Hidden-chain diagram: the regime is hidden and the vehicle level is observed.
4. Counted tables, then the rising EM curve and the hour-by-state heatmap.
5. Forward, backward, and the filtered-versus-smoothed Rush curve.
6. The 8-hour check where brute force lands on the sum-product lines.
7. The log10 collapse, then Viterbi’s white path.
8. The accuracy bars and the probability ribbon.
9. The Kalman band, the gain, and the continuous error.
'''),
code('''
fig, ax = plt.subplots(figsize=(10.2, 4.6), layout='constrained')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis('off')
ax.set_title('One sequence, eight questions')

def box(x, y, w, h, text, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                facecolor=face, edgecolor=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.5)

box(0.2, 4.5, 3.0, 1.3, 'Sequential data' + chr(10) + 'order and daily memory', '#fff8db')
box(3.5, 4.5, 3.0, 1.3, 'Markov chain' + chr(10) + 'next level from this level', ACCENT)
box(6.8, 4.5, 3.0, 1.3, 'Hidden Markov model' + chr(10) + 'regime emits the level', '#f3d27a')
for x in (3.15, 6.45):
    ax.annotate('', xy=(x + 0.3, 5.15), xytext=(x, 5.15), arrowprops={'arrowstyle': '->', 'color': INK})
for x, text, face in [
    (0.15, 'Forward-' + chr(10) + 'backward', '#fff8db'),
    (2.15, 'Sum-product' + chr(10) + 'messages', '#fff8db'),
    (4.15, 'Scaling' + chr(10) + 'factors', '#fff8db'),
    (6.15, 'Viterbi' + chr(10) + 'best path', ACCENT),
    (8.15, 'Kalman' + chr(10) + 'filter', '#f3d27a')]:
    box(x, 0.5, 1.75, 1.7, text, face)
ax.annotate('', xy=(5, 2.25), xytext=(8.3, 4.45), arrowprops={'arrowstyle': '->', 'color': INK})
plt.show()

display(pd.DataFrame({
    'Question a faculty member may ask': [
        'Why not shuffle the rows?',
        'What is a state, an observation, and an emission?',
        'Why divide the forward vector by its sum?',
        'How is smoothing different from filtering?',
        'Why did brute force match the lines?',
        'What does Viterbi return that forward-backward does not?',
        'Why can EM rediscover the clock?',
        'What is the Kalman gain?',
    ],
    'Answer from these pictures': [
        'Autocorrelation peaks at 24 h and 168 h, so order is the signal.',
        'Night, off-peak, and rush are hidden. Low, medium, and high are observed. B links them.',
        'That is the scaling factor. It keeps a belief and builds the log-likelihood.',
        'Filtering uses the past. Smoothing multiplies by the backward message.',
        'Sum-product is the efficient way to total every hidden path.',
        'One jointly best path, not a separate winner at each hour.',
        'The emissions and transitions prefer different parts of the day.',
        'How much the new vehicle count moves the latent level.',
    ],
}))
''')])
add(5, 'Combining models', 'Compare a small decision tree with random forest and AdaBoost classifiers.', [
    'Implement CART learning algorithms to perform categorization.',
    'Implement Ensemble learning models to perform classification.'
], [
md('## Prepare the same prediction task\nUse the same chronological split as Unit 2. The input is the current hourly count; the target is whether the following hour reaches 4000 vehicles. Tree models can use vehicle counts directly without standardization.'),
code(PAIRS),
code('''
from sklearn.metrics import accuracy_score

X_train = train[['traffic_volume']]
X_test = test[['traffic_volume']]
y_train = (train['next_volume'] >= 4000).astype(int)
y_test = (test['next_volume'] >= 4000).astype(int)
'''),
md('## Experiment 1 · CART decision tree\nCART makes yes/no splits. Limit the depth to two so we can read the whole tree.'),
code('''
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X_train, y_train)
tree_prediction = tree.predict(X_test)
print('CART accuracy:', round(accuracy_score(y_test, tree_prediction), 3))

plt.figure(figsize=(9, 4))
plot_tree(tree, feature_names=['Current volume (vehicles/hour)'], class_names=['Light traffic', 'Heavy traffic'], filled=True, fontsize=9)
plt.title('A small CART tree')
plt.show()
'''),
md('## Experiment 2 · Two ensemble models\nRandom forest combines many trees. AdaBoost combines small models while paying more attention to earlier mistakes. Both use the same training and testing rows as CART.'),
code('''
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

forest = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
forest.fit(X_train, y_train)
forest_prediction = forest.predict(X_test)

boost = AdaBoostClassifier(n_estimators=20, random_state=42)
boost.fit(X_train, y_train)
boost_prediction = boost.predict(X_test)

results = pd.DataFrame({
    'Model': ['CART', 'Random forest', 'AdaBoost'],
    'Accuracy': [accuracy_score(y_test, tree_prediction),
                 accuracy_score(y_test, forest_prediction),
                 accuracy_score(y_test, boost_prediction)]
})
display(results.round(3))
plt.bar(results['Model'], results['Accuracy'])
plt.ylim(0, 1)
plt.ylabel('Test accuracy')
plt.title('Compare the three classifiers')
plt.show()
'''),
md('## What to explain\nA tree is one model; an ensemble combines several models. Compare the measured test accuracies instead of assuming more trees always win. Two ensemble methods cover the assignment without a long list of models.\n\n**Try:** Change `n_estimators=20` to `10`.\n\n**Viva:** What does tree depth mean? How are random forest and boosting different?')])

(DIST / 'units.json').write_text(json.dumps(units, indent=2), encoding='utf-8')
for number in range(1, 6):
    route = DIST / f'unit-{number}'
    route.mkdir(exist_ok=True)
    shutil.copy(DIST / 'index.html', route / 'index.html')
print('Created five beginner notebooks using', len(df), 'traffic hours.')
print('Code lines by unit:', {u['number']: sum(len(c.source.splitlines()) for c in nbf.read(NB / u['filename'], as_version=4).cells if c.cell_type == 'code') for u in units})
