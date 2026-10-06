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

**traffic_volume**: vehicles counted in that hour. **temperature_c**: air temperature in degrees Celsius. **date_time** identifies the hour and is not a numerical model feature.

The original dataset has 48,204 rows and includes weather and holidays from 2012–2018. Repeated weather descriptions can share a timestamp; their traffic counts were verified identical before keeping the first entry. The teaching CSV uses the first six weeks of the longest continuous hourly sequence. Temperature was converted from Kelvin by subtracting 273.15. No missing hours were filled and no measurements were invented.

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
md('## Experiment 1 · Read and view\n`head()` shows the first five rows. `shape` gives the number of rows and columns. Each row is one hour; numeric columns measure vehicle count and air temperature.'),
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

add(4, 'Hidden Markov models', 'Use a two-state HMM to predict whether the next observed traffic hour is busy.', [
    'Implement HMM to predict the sequential data.'
], [
md('## Understand the sequence\nKeep the continuous hourly order. For this supervised teaching HMM, the hidden regime is **night-time or daytime**, labelled in training using `6 <= hour < 20`. The observation is **light or heavy traffic**, with heavy meaning at least 4000 vehicles/hour. During testing, hide regime labels and use only previously observed traffic categories. We deliberately withhold the clock to demonstrate filtering; a practical traffic model would use the available hour-of-day directly. We learn tables by counting labelled training data, not by unsupervised Baum-Welch.'),
code('''
data = df.copy()
split = int(len(data) * 0.8)
hour = data['date_time'].dt.hour
states = ((hour >= 6) & (hour < 20)).astype(int).to_numpy()
observations = (data['traffic_volume'] >= 4000).astype(int).to_numpy()
print('State 0 = night-time regime; state 1 = daytime regime')
print('Observation 0 = light traffic; observation 1 = heavy traffic')
display(data.head())
'''),
md('## Step 1 · Learn two probability tables\n`transition[i, j]` tells us how often state i is followed by state j. `emission[i, j]` tells us how often state i produces observation j. Starting counts at one prevents zero probabilities (add-one smoothing). Only the first 80% of rows contribute to these tables.'),
code('''
transition = np.ones((2, 2))
emission = np.ones((2, 2))
initial = np.ones(2)
initial[states[0]] += 1
initial = initial / initial.sum()

for i in range(split - 1):
    transition[states[i], states[i + 1]] += 1
for i in range(split):
    emission[states[i], observations[i]] += 1

transition = transition / transition.sum(axis=1, keepdims=True)
emission = emission / emission.sum(axis=1, keepdims=True)
display(pd.DataFrame(transition, index=['Night-time', 'Daytime'], columns=['Next night-time', 'Next daytime']).round(3))
display(pd.DataFrame(emission, index=['Night-time', 'Daytime'], columns=['Light traffic', 'Heavy traffic']).round(3))

plt.imshow(transition, vmin=0, vmax=1, cmap='Blues')
plt.xticks([0, 1], ['Next night-time', 'Next daytime'])
plt.yticks([0, 1], ['Night-time', 'Daytime'])
plt.colorbar(label='Probability')
plt.title('Hidden-state transition probabilities')
plt.show()
'''),
md('## Step 2 · Predict, then update\n`belief` is our probability for each hidden state. Matrix multiplication predicts the next state, then the next observation. After making that prediction, we reveal the actual traffic category and update the belief. Dividing by the sum normalizes the forward filter and avoids extremely small accumulated probabilities.'),
code('''
predictions = []
probabilities = []
belief = initial.copy()

for i, observation in enumerate(observations):
    if i == 0:
        next_state = belief
    else:
        next_state = belief @ transition
    next_observation = next_state @ emission

    # Record the test prediction BEFORE revealing this observation.
    if i >= split:
        predictions.append(int(np.argmax(next_observation)))
        probabilities.append(next_observation[1])

    belief = next_state * emission[:, observation]
    belief = belief / belief.sum()

from sklearn.metrics import accuracy_score
actual = observations[split:]
print('HMM test accuracy:', round(accuracy_score(actual, predictions), 3))
print('Previous-hour category baseline:', round(accuracy_score(actual, observations[split - 1:-1]), 3))
print('Probability the following traffic hour is heavy:', round(float((belief @ transition @ emission)[1]), 3))

plt.plot(actual[:30], 'o-', label='Actual category', alpha=0.6)
plt.plot(predictions[:30], 'x--', label='Predicted category')
plt.yticks([0, 1], ['Light traffic', 'Heavy traffic'])
plt.xlabel('Test observation number')
plt.title('Sequential HMM predictions')
plt.legend()
plt.show()
'''),
md('## What to explain\nThe hidden regime is separate from the observed traffic category: daytime does not always mean heavy traffic. We reveal time-based regime labels only for training, then infer regime beliefs from past observed categories. The HMM predicts the next traffic category before seeing it. It approximates day/night durations using transition probabilities rather than an exact clock.\n\n**Try:** Inspect the strongest transition.\n\n**Viva:** What are the states and observations? What is an emission? Why normalize the belief? Why must we predict before updating?')])

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
