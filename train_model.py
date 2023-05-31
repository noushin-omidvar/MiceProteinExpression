import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utility import report, plot_confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold

# Load data
data_file = 'Data/Processed_Data/processed_data.xlsx'
data_file_path = os.path.join(os.getcwd(), data_file)
data = pd.read_excel(data_file_path, header=0)
data_size = data.shape

# Get column names and class labels
names = list(data.columns.values)[1:-4]
prot_names = [x.encode('UTF8')[:-2] for x in names]
classes = list(set(data['class']))

# Select features and target variable
X_complete = data.iloc[:, 1:78]
feature_names = X_complete.columns
y = data['class']

# Data splitting and scaling
X_train, X_test, y_train, y_test = train_test_split(X_complete, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize random forest classifier
classifier = RandomForestClassifier()

# Grid search over random forest parameters
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10, None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"], 
              "max_features": ["sqrt", "log2"]}

grid_search = GridSearchCV(classifier, param_grid=param_grid)

# Perform grid search to find best estimator
start = time.time()
grid_search.fit(X_train_scaled, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
classifier = grid_search.best_estimator_

# Evaluation of the classifier without feature selection
fitted = classifier.fit(X_train_scaled, y_train)
scores = cross_val_score(classifier, X_train_scaled, y_train, cv=10)
print('After cross validation, Scores are:', scores * 100)
print('Mean:', scores.mean(), 'and Standard Deviation:', scores.std())

predicted = fitted.predict(X_test_scaled)
print(classification_report(y_test, predicted, target_names=classes))

# Plot confusion matrix for the classifier without feature selection
conf_matrix = confusion_matrix(y_test, predicted)
np.set_printoptions(precision=2)
plt.figure('Figure 1')
plot_confusion_matrix(conf_matrix,
                      classes=classes,
                      title='Random-Forest without feature selection',
                      normalize=True)
plt.savefig('visualizations/No_feature_selection.png', bbox_inches='tight')

# Get feature importances and plot feature importance graph
importances = classifier.feature_importances_
importances = pd.Series(importances, index=feature_names)

indices = np.argsort(importances)[::-1]
sorted_feature_names = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(18, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=60)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Plot')
plt.tight_layout()
plt.show()

# Perform recursive feature elimination with cross-validation (RFECV)
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X_train_scaled, y_train)

selected_indices = rfecv.get_support(indices=True)
selected_feature_names_RF = [feature_names[i] for i in selected_indices]

print("Selected Features:")
print(selected_feature_names_RF)

X_train_selected = rfecv.transform(X_train_scaled)
X_test_selected = rfecv.transform(X_test_scaled)
print(X_test_scaled.shape)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Accuracy of the model based on selected features is', rfecv.grid_scores_[rfecv.n_features_])

# Evaluate the classifier with selected features
fitted = classifier.fit(X_train_selected, y_train)
scores = cross_val_score(classifier, X_train_selected, y_train, cv=10)
print('After cross validation, Scores are:', scores * 100)
print('Mean:', scores.mean(), 'and Standard Deviation:', scores.std())

predicted = fitted.predict(X_test_selected)
print(classification_report(y_test, predicted, target_names=classes))

# Plot confusion matrix for the classifier with feature selection
conf_matrix = confusion_matrix(y_test, predicted)
np.set_printoptions(precision=2)
plt.figure('Figure 2')
plot_confusion_matrix(conf_matrix,
                      classes=classes,
                      title='Random-Forest with feature selection',
                      normalize=True)
plt.savefig('visualizations/feature_selection.png', bbox_inches='tight')

# Plot RFECV curve
plt.figure(figsize=(10, 6))
plt.title('Recursive Feature Elimination (RFE) Curve')
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.plot(rfecv.grid_scores_)

# Plot feature importance for selected features
importances = classifier.feature_importances_
importances = pd.Series(importances, index=selected_feature_names_RF)

indices = np.argsort(importances)[::-1]
sorted_feature_names = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(18, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=60)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Plot')
plt.tight_layout()
plt.show()

# Create correlation matrix heatmap for selected features
df_selected = pd.DataFrame(X_train_scaled[:, selected_indices], columns=selected_feature_names_RF)
corr_matrix = df_selected.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', square=True)
plt.title("Correlation Matrix Heatmap for Selected Features")
plt.tight_layout()
plt.show()
