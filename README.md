# Iris Classification with Machine Learning

## Project Overview:
    This project applies machine learning algorithms to classify Iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements.The pipeline includes data preprocessing, model training, evaluation, cross-validation, and hyperparameter tuning.

## Workflow:
  1.Load Dataset -> sklearn.datasets.load_iris()
  2.Preprocessing -> StandardScaler used for feature scaling.
  3.Train-Test Split -> 80% training, 20% testing.
  4.Models Used:
     i)  Decision Tree Classifier 
     ii) Support Vector Machine (SVM) 
     iii)Random Forest Classifier 
     iv) Ensemble Voting Classifier ->soft voting
## Evaluation Metrics:
      i)  Accuracy Score
      ii) Confusion Matrix
      iii)Classification Report (Precision, Recall, F1-score)
## Cross Validation 
      5-fold CV for each model.
## Hyperparameter Tuning -> GridSearchCV applied to optimize:
       i)  SVM → C, kernel
       ii) Random Forest > n_estimators, max_depth, min_samples_split
       iii)Decision Tree -> criterion, max_depth, min_samples_split, min_samples_leaf

## Results:
      1.All models achieved high accuracy on the Iris dataset.
      2.GridSearchCV identified the best hyperparameters for SVM, Random Forest, and Decision Tree, improving performance.
      3.The Voting Classifier (Ensemble) combined multiple models for robust results.
## Project Structure
      irisclassification/
      │── Iris_Classification.ipynb # Jupyter notebook with code
      │── README.md # Project documentation
      │── requirements.txt # Dependencies

