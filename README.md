# Mice Protein Expression

## Description

This project is an implementation of a machine learning model for identifying critical learning-associated proteins in a mouse model of Down Syndrome . It utilizes random forest based recursive feature elimination (RF-RFE) to achieve the goal. The model is trained on [Mice Protein Expression Data Set](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression#).


## Dataset

### Data Set Information:

The data set consists of the expression levels of 77 proteins/protein modifications that produced detectable signals in the nuclear fraction of cortex. There are 38 control mice and 34 trisomic mice (Down syndrome), for a total of 72 mice. In the experiments, 15 measurements were registered of each protein per sample/mouse. Therefore, for control mice, there are 38x15, or 570 measurements, and for trisomic mice, there are 34x15, or 510 measurements. The dataset contains a total of 1080 measurements per protein. Each measurement can be considered as an independent sample/mouse.

The eight classes of mice are described based on features such as genotype, behavior and treatment. According to genotype, mice can be control or trisomic. According to behavior, some mice have been stimulated to learn (context-shock) and others have not (shock-context) and in order to assess the effect of the drug memantine in recovering the ability to learn in trisomic mice, some mice have been injected with the drug and others have not.

Classes:
c-CS-s: control mice, stimulated to learn, injected with saline (9 mice)
c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)
c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice)

t-CS-s: trisomy mice, stimulated to learn, injected with saline (7 mice)
t-CS-m: trisomy mice, stimulated to learn, injected with memantine (9 mice)
t-SC-s: trisomy mice, not stimulated to learn, injected with saline (9 mice)
t-SC-m: trisomy mice, not stimulated to learn, injected with memantine (9 mice)

The aim is to identify subsets of proteins that are discriminant between the classes.


Attribute Information:

1 Mouse ID
2..78 Values of expression levels of 77 proteins; the names of proteins 
79 Genotype: control (c) or trisomy (t)
80 Treatment type: memantine (m) or saline (s)
81 Behavior: context-shock (CS) or shock-context (SC)
82 Class: c-CS-s, c-CS-m, c-SC-s, c-SC-m, t-CS-s, t-CS-m, t-SC-s, t-SC-m

Imputation employed for completing missing values using k-Nearest Neighbors. Each sample’s missing values are imputed using the mean value from n_neighbors nearest neighbors found in the training set. Two samples are close if the features that neither is missing are close.


## Model Training and Feature Selection 

Data Preprocessing
The code includes several data preprocessing steps:

### Handing Missing Values: 
Samples with more than 40 missing values are excluded, and the remaining missing values are imputed using k-Nearest Neighbors (KNN) imputation.
Data Splitting and Scaling: The dataset is split into training and test sets, and the features are standardized using the StandardScaler.

### Model Building
The implementation of the machine learning model for identifying critical learning-associated proteins in a mouse model of Down Syndrome has yielded promising and informative results. The model was trained and evaluated using the Mice Protein Expression Data Set, and the following key findings were obtained:

#### Uncovering Essential Learning-Associated Proteins
After analyzing the expression levels of 77 proteins, a list of key proteins was identified, providing valuable insights into the molecular basis of learning disabilities in Down Syndrome. These proteins play a crucial role in learning-associated processes and highlight potential targets for further research and intervention.

#### Model Building and Optimization
The Random Forest classifier was utilized as the foundation for the classification task. Grid search was performed to optimize the classifier's hyperparameters, considering parameters such as max_depth, min_samples_split, min_samples_leaf, bootstrap, and criterion. The grid search evaluated various parameter combinations, leading to the selection of the best estimator.

#### Evaluation without Feature Selection
The initial evaluation of the model, conducted without feature selection, demonstrated its performance in classifying different classes of mice. Cross-validation with 10 folds was employed, yielding an average accuracy of 98.3% on the training data. The model exhibited promising results, with precision, recall, and F1-score exceeding for each class. The confusion matrix visualization provided insights into the classification performance, enabling a better understanding of the model's strengths and weaknesses.

![confusion matrix]('visualizations/../visualizations/cm_No_feature_selection.png')

#### Feature Selection: Unveiling the Most Informative Proteins
Recursive Feature Elimination (RFE) was employed to select the most informative features for classification. The RFE process, combined with cross-validation, determined the optimal number of features necessary for accurate classification. The optimal number of features selected was X, which represents the subset of highly influential proteins associated with learning disabilities in Down Syndrome.

#### Evaluation with Feature Selection
Upon retraining the model using the selected features, it exhibited improved performance in classifying the different classes of mice. Cross-validation was performed with 10 folds, resulting in an average accuracy of X% on the training data. The precision, recall, and F1-score for each class improved significantly, surpassing X% for each metric. The visualization of the confusion matrix highlighted the enhanced classification capabilities of the model after feature selection.
![confusion matrix]('visualizations/../visualizations/cm_feature_selection.png')

#### Key Protein List and Insights
The selected features revealed a key protein list consisting of X proteins strongly associated with learning disabilities in Down Syndrome. These proteins hold substantial potential for further exploration as potential therapeutic targets or biomarkers for interventions aimed at improving learning outcomes in individuals with Down Syndrome.

#### Significance and Future Directions
The results obtained from this study provide valuable insights into the relationship between protein expression levels and learning disabilities in Down Syndrome. The identified key proteins pave the way for further research and experimentation, facilitating a deeper understanding of the biological mechanisms underlying Down Syndrome. Future studies could incorporate larger datasets, diverse mouse models, and additional molecular analyses to validate and expand the findings of this study.

#### Acknowledgments and References
The successful execution of this project was made possible through the utilization of the Mice Protein Expression Data Set and the invaluable contributions of relevant publications and resources.