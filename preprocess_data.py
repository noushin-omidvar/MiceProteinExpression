import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Load the data
data_file = 'Data/Raw_Data/Data_Cortex_Nuclear.xls'
data_file_path = os.path.join(os.getcwd(), data_file)
raw_data = pd.read_excel(data_file_path, header=0)
data_size = raw_data.shape

#---------------------------------
# Exploratory Data Analysis (EDA)
#----------------------------------
# Data Summary
print("Data Summary:")
print(raw_data.head())

# Violin Plot
plt.figure(figsize=(10, 8))
sns.violinplot(data=raw_data)
plt.savefig('violinPlot.png')
plt.show()

# Data Information
print("\nData Information:")
print(raw_data.info())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(raw_data.describe())

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(raw_data.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('corr.png')
plt.show()

# Missing Data Matrix
msno.matrix(raw_data)
plt.savefig('missing_matrix.png')
plt.show()

#---------------------
# Data Preprocessing
#---------------------
# Handling Missing Values
# Excluding samples with a majority of missing values
missing_values_count = raw_data.isnull().sum().sum()
print('Number of missing values:', missing_values_count)

df = raw_data[raw_data.isnull().sum(axis=1) < 40]
df_size = df.shape
excluded_samples_count = data_size[0] - df_size[0]
print(excluded_samples_count, 'samples had more than 40 missing values. These instances were excluded.')

missing_values_count_after_exclusion = df.isnull().sum().sum()
print('Number of missing values after exclusion:', missing_values_count_after_exclusion)

# Imputation using k-Nearest Neighbors
# Completing missing values using mean value from nearest neighbors
imputer = KNNImputer(n_neighbors=2)
data_complete = raw_data.copy()
data_complete.iloc[:, 1:78] = imputer.fit_transform(data_complete.iloc[:, 1:78])

# Save the processed data
data_file_new = 'Data/Processed_Data/processed_data.xlsx'
data_file_new_path = os.path.join(os.getcwd(), data_file_new)
data_complete.to_excel(data_file_new_path, engine='xlsxwriter')
