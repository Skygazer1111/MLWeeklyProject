"""Create five beginner notebooks from the two-column Old Faithful dataset."""
from pathlib import Path
import hashlib
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

# Only remove the CSV mirror's row number. Do not sort or change measurements.
df = pd.read_csv(DATA / 'geyser_original.csv')[['duration', 'waiting']]
assert len(df) == 299 and df.isna().sum().sum() == 0
df.to_csv(DATA / 'geyser.csv', index=False)
metadata = {
    'name': 'Old Faithful geyser (MASS version)', 'rows': 299,
    'columns': {'duration': 'Eruption duration in minutes', 'waiting': 'Waiting time BEFORE this eruption, in minutes'},
    'collection': 'Continuous observations, August 1-15, 1985',
    'source': 'https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/geyser.html',
    'csv_source': 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/geyser.csv',
    'citation': 'Azzalini, A. and Bowman, A. W. (1990). A look at some data on the Old Faithful geyser. Applied Statistics 39, 357-365.',
    'preparation': 'Drop rownames; preserve all 299 measurements and their sequence order.',
    'limitations': 'Some night-time durations were recorded as 2, 3 or 4 minutes from short/medium/long descriptions.',
    'sha256_original': hashlib.sha256((DATA / 'geyser_original.csv').read_bytes()).hexdigest(),
}
(DATA / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
for name in ['geyser.csv', 'geyser_original.csv', 'metadata.json']:
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
    df = pd.read_csv('data/geyser.csv')
except FileNotFoundError:
    df = pd.read_csv('../data/geyser.csv')

print('Number of eruptions:', len(df))
'''
INTRO = '''
## About this small dataset
Old Faithful is a geyser: a natural hot spring that sometimes shoots water into the air. This dataset has **299 eruptions and only two measurements**. Each row is one eruption, in the original recorded sequence.

**duration**: how long this eruption lasted, in minutes. **waiting**: how long people waited **before this eruption**, in minutes. There are no missing values, so we do not need to fill or remove them.

Source: [R MASS dataset documentation](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/geyser.html). Observations were collected August 1-15, 1985. Reference: Azzalini and Bowman (1990), *A look at some data on the Old Faithful geyser*. The supplied original CSV is from the Rdatasets mirror. We removed only its row-number column. Some night-time durations were recorded as 2, 3 or 4 minutes from short/medium/long descriptions.

Run the cells from top to bottom. Keep the `data` folder beside the `notebooks` folder. Learn each step by changing a value and rerunning the cell. These are small classroom demonstrations, not a complete prediction system.
'''
PAIRS = '''
# The next row's waiting time is the wait AFTER the current eruption.
data = df.copy()
data['next_wait'] = data['waiting'].shift(-1)
data = data.dropna()  # The last eruption has no recorded next wait.

# Keep the first 80% for learning and the final 20% for testing.
split = int(len(data) * 0.8)
train = data.iloc[:split]
test = data.iloc[split:]
print('Training rows:', len(train), '| Testing rows:', len(test))
display(data.head())
'''
CLASSIFY = '''
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# A long wait means 70 minutes or more. This is our classroom rule.
X_train = train[['duration']]
X_test = test[['duration']]
y_train = (train['next_wait'] >= 70).astype(int).to_numpy()
y_test = (test['next_wait'] >= 70).astype(int).to_numpy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Class 0 = wait below 70 minutes; class 1 = wait at least 70 minutes')
'''
units = []

def add(number, title, aim, practices, cells):
    names = ['introduction', 'linear_models', 'clustering_pca', 'hidden_markov_models', 'combining_models']
    filename = f'unit_{number}_{names[number-1]}.ipynb'
    heading = f'# Unit {number} · {title}\n\nCourse: **21CSC305P — Machine Learning**\n\n## Aim\n{aim}\n\n## Assignment tasks\n'
    heading += '\n'.join(f'{i+1}. {item}' for i, item in enumerate(practices))
    notebook = nbf.v4.new_notebook(cells=[md(heading), md(INTRO), code(SETUP)] + cells,
        metadata={'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python', 'version': '3.12'}, 'unit': number, 'dataset': 'MASS/geyser'})
    nbf.write(notebook, NB / filename)
    shutil.copy(NB / filename, DIST / 'notebooks' / filename)
    units.append({'number': number, 'title': title, 'summary': aim, 'practices': practices, 'filename': filename})

add(1, 'Introduction', 'Read a small CSV, understand its two columns, and draw simple graphs.', [
    'Devise a program to import, load and view dataset.',
    'Create a program to display the summary and statistics of the dataset.'
], [
md('## Experiment 1 · Read and view\n`head()` shows the first five rows. `shape` gives the number of rows and columns. Each number is a measurement in minutes.'),
code('''
display(df.head())
print('Rows and columns:', df.shape)
print('Missing values:')
print(df.isna().sum())
'''),
md('## Experiment 2 · Summary statistics\n`describe()` gives count, mean, standard deviation, minimum, quartiles and maximum. The mean is the average; the median is the middle value.'),
code('''
display(df.describe().round(2))
print('Average eruption duration:', round(df['duration'].mean(), 2), 'minutes')
print('Median waiting time:', df['waiting'].median(), 'minutes')

plt.hist(df['duration'], bins=12, edgecolor='white')
plt.xlabel('Eruption duration (minutes)')
plt.ylabel('Number of eruptions')
plt.title('How long do eruptions last?')
plt.show()
'''),
code('''
plt.scatter(df['waiting'], df['duration'], s=20)
plt.xlabel('Waiting time before eruption (minutes)')
plt.ylabel('Eruption duration (minutes)')
plt.title('The two measurements')
plt.show()
'''),
md('## What to explain\nA histogram counts observations in ranges. A scatter plot draws one dot for each row. Read the graphs before making a claim about the relationship.\n\n**Try:** Change `bins=12` to `bins=6`.\n\n**Viva:** What does one row represent? How do mean and median differ?')])

add(2, 'Linear models', 'Use eruption duration to predict the next waiting time, then classify a short or long wait.', [
    'Implement linear regression to perform prediction.',
    'Implement Bayesian logistic regression and SVM for classification.'
], [
md('## Prepare the prediction problem\nThe original `waiting` column belongs **before** its eruption. `shift(-1)` takes the following row’s waiting time, so `next_wait` belongs **after** the current eruption. The last row is removed because its next wait is unknown. Only duration is used as the input; `next_wait` is the answer we want to predict.'),
code(PAIRS),
md('## Experiment 1 · Linear regression\nFit a straight line: next waiting time = slope × duration + intercept. `fit()` learns from training rows; `predict()` gives answers for new rows.'),
code('''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()
model.fit(train[['duration']], train['next_wait'])
predicted_wait = model.predict(test[['duration']])
print('Slope:', round(model.coef_[0], 2))
print('Intercept:', round(model.intercept_, 2))
print('Mean absolute error:', round(mean_absolute_error(test['next_wait'], predicted_wait), 2), 'minutes')
display(pd.DataFrame({'actual': test['next_wait'].to_numpy()[:5], 'predicted': predicted_wait[:5]}).round(2))

plt.scatter(test['duration'], test['next_wait'], label='Actual test rows')
plt.plot(test['duration'], predicted_wait, color='orange', label='Predicted line')
plt.xlabel('Eruption duration (minutes)')
plt.ylabel('Next waiting time (minutes)')
plt.legend()
plt.show()
'''),
md('## Prepare classification\nWe turn the target into two categories: **0 = short wait**, **1 = long wait**. Scaling centers the input around zero. Learn the scaler from training rows only.'),
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
order = np.argsort(test['duration'].to_numpy())
plt.scatter(test['duration'], y_test, label='Actual class', alpha=0.5)
plt.plot(test['duration'].to_numpy()[order], bayes_probability[order], color='orange', label='Probability of a long wait')
plt.xlabel('Eruption duration (minutes)')
plt.ylabel('Class / probability')
plt.legend()
plt.show()
'''),
md('## Experiment 2b · SVM\nAn SVM finds a boundary between classes. With one input and a linear kernel, the boundary is a cutoff along the duration axis.'),
code('''
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
svm_prediction = svm.predict(X_test_scaled)
print('SVM accuracy:', round(accuracy_score(y_test, svm_prediction), 3))
ConfusionMatrixDisplay.from_predictions(y_test, svm_prediction, display_labels=['Short wait', 'Long wait'], colorbar=False)
plt.title('SVM: actual and predicted classes')
plt.show()
'''),
md('## What to explain\nRegression predicts a number; classification predicts a category. An error of 5 minutes means a numerical prediction was 5 minutes away from its actual value. The Bayesian model uses a prior and averages several possible curves. We do not adjust model settings after seeing test results.\n\n**Try:** Predict the next waiting time for a 2-minute eruption using `model.predict(pd.DataFrame({"duration": [2]}))`.\n\n**Viva:** What are input and target? What is a prior? What does accuracy measure?')])

add(3, 'Clustering & PCA', 'Find two groups of eruptions and reduce two measurements to one new feature.', [
    'Implement K-means clustering, mixtures of Gaussians and Hierarchical clustering algorithm to categorize data.',
    'Create a program to perform PCA.'
], [
md('## Prepare the two features\nUse duration and the following waiting time together. Standardization helps because duration is only a few minutes, while waiting is many more minutes. This unit explores the whole dataset; it is not a held-out prediction test.'),
code('''
from sklearn.preprocessing import StandardScaler

data = df.copy()
data['next_wait'] = data['waiting'].shift(-1)
data = data.dropna()
features = data[['duration', 'next_wait']]
scaled = StandardScaler().fit_transform(features)
display(features.head())
'''),
md('## Experiment 1 · Three ways to cluster\nK-means assigns points to the nearest center. Gaussian mixtures use probabilities. Hierarchical clustering repeatedly joins nearby groups. We choose two groups to study short/long eruption patterns.'),
code('''
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled)
mixture = GaussianMixture(n_components=2, random_state=42)
mixture_labels = mixture.fit_predict(scaled)
hierarchy = AgglomerativeClustering(n_clusters=2)
hierarchy_labels = hierarchy.fit_predict(scaled)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].scatter(data['duration'], data['next_wait'], c=kmeans_labels, s=15)
axes[0].set_title('K-means')
axes[1].scatter(data['duration'], data['next_wait'], c=mixture_labels, s=15)
axes[1].set_title('Gaussian mixture')
axes[2].scatter(data['duration'], data['next_wait'], c=hierarchy_labels, s=15)
axes[2].set_title('Hierarchical clustering')
for ax in axes:
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Next wait (minutes)')
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
md('## What to explain\nClustering does not need answer labels. Cluster 0 and cluster 1 are just names and can swap between methods. PCA reduces the number of features; it does not create a class label.\n\n**Try:** Change `n_components=1` to `2` and compare the new shape.\n\n**Viva:** Why scale the two columns? How is clustering different from classification?')])

add(4, 'Hidden Markov models', 'Use a small two-state HMM to predict the next short or long waiting interval.', [
    'Implement HMM to predict the sequential data.'
], [
md('## Understand the sequence\nKeep the recorded order. A hidden state is a **short or long eruption** (`duration >= 3` means long). An observation is the **short or long wait after it** (`next_wait >= 70` means long). For training we reveal the eruption states and count transitions and emissions. During test predictions, we use only previously observed waits; upcoming durations and waits are hidden. This is a **supervised HMM demonstration**, which keeps training much simpler than Baum-Welch EM.'),
code('''
data = df.copy()
data['next_wait'] = data['waiting'].shift(-1)
data = data.dropna()
split = int(len(data) * 0.8)
states = (data['duration'] >= 3).astype(int).to_numpy()
observations = (data['next_wait'] >= 70).astype(int).to_numpy()
print('State 0 = short eruption; state 1 = long eruption')
print('Observation 0 = short wait; observation 1 = long wait')
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
display(pd.DataFrame(transition, index=['Short eruption', 'Long eruption'], columns=['Next short', 'Next long']).round(3))
display(pd.DataFrame(emission, index=['Short eruption', 'Long eruption'], columns=['Short wait', 'Long wait']).round(3))

plt.imshow(transition, vmin=0, vmax=1, cmap='Blues')
plt.xticks([0, 1], ['Next short', 'Next long'])
plt.yticks([0, 1], ['Short eruption', 'Long eruption'])
plt.colorbar(label='Probability')
plt.title('Hidden-state transition probabilities')
plt.show()
'''),
md('## Step 2 · Predict, then update\n`belief` is our probability for each hidden state. Matrix multiplication predicts the next state, then the next observation. After making that prediction, we reveal the actual waiting category and update the belief. Dividing by the sum normalizes the forward filter and avoids extremely small accumulated probabilities.'),
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
print('Previous-wait baseline:', round(accuracy_score(actual, observations[split - 1:-1]), 3))
print('Probability the following wait is long:', round(float((belief @ transition @ emission)[1]), 3))

plt.plot(actual[:30], 'o-', label='Actual category', alpha=0.6)
plt.plot(predictions[:30], 'x--', label='Predicted category')
plt.yticks([0, 1], ['Short wait', 'Long wait'])
plt.xlabel('Test observation number')
plt.title('Sequential HMM predictions')
plt.legend()
plt.show()
'''),
md('## What to explain\nAn ordinary Markov model would treat the observed wait category as the state. This HMM has a separate eruption state and a table connecting it to the observed wait. We know state labels during training, then hide them during filtering and prediction. We do not claim to learn unlabelled hidden states with EM.\n\n**Try:** Inspect which transition has the largest probability.\n\n**Viva:** What are the two states? What is an emission? Why do we predict before updating with the actual observation?')])

add(5, 'Combining models', 'Compare a small decision tree with random forest and AdaBoost classifiers.', [
    'Implement CART learning algorithms to perform categorization.',
    'Implement Ensemble learning models to perform classification.'
], [
md('## Prepare the same prediction task\nUse the same split as Unit 2. The input is eruption duration, and the target is whether the following wait reaches 70 minutes. Tree models can use the original minutes without scaling.'),
code(PAIRS),
code('''
from sklearn.metrics import accuracy_score

X_train = train[['duration']]
X_test = test[['duration']]
y_train = (train['next_wait'] >= 70).astype(int)
y_test = (test['next_wait'] >= 70).astype(int)
'''),
md('## Experiment 1 · CART decision tree\nCART makes yes/no splits. Limit the depth to two so we can read the whole tree.'),
code('''
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X_train, y_train)
tree_prediction = tree.predict(X_test)
print('CART accuracy:', round(accuracy_score(y_test, tree_prediction), 3))

plt.figure(figsize=(9, 4))
plot_tree(tree, feature_names=['Duration (minutes)'], class_names=['Short wait', 'Long wait'], filled=True, fontsize=9)
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
print('Created five beginner notebooks using', len(df), 'geyser records.')
print('Code lines by unit:', {u['number']: sum(len(c.source.splitlines()) for c in nbf.read(NB / u['filename'], as_version=4).cells if c.cell_type == 'code') for u in units})
