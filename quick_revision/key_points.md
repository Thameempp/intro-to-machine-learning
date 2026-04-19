# Machine Learning  Key Points

---

## 1. Types of Machine Learning
- Supervised Learning → labeled data (regression, classification)
- Unsupervised Learning → no labels (clustering, dimensionality reduction)
- Semi-supervised → small labeled + large unlabeled
- Reinforcement Learning → reward-based learning

---

## 2. Supervised Learning Essentials
- Goal: learn mapping \( X \rightarrow y \)
- Regression → continuous output
- Classification → discrete output
- Model learns by minimizing loss function

---

## 3. Important Concepts
- Feature (X) → input variable
- Label (y) → output/target
- Prediction (\(\hat{y}\)) → model output
- Parameters (w, b) → learned values
- Loss function → error measure
- Optimization → minimize loss

---

## 4. Train-Test Split
- Training set → learn model
- Validation set → tune hyperparameters
- Test set → final evaluation
- Never train on test data

---

## 5. Overfitting vs Underfitting
- Overfitting → memorizes data, poor generalization
- Underfitting → fails to learn patterns
- Solution:
  - Overfitting → regularization, more data
  - Underfitting → more complex model

---

## 6. Bias-Variance Tradeoff
- High bias → underfitting
- High variance → overfitting
- Goal → balance both

---

## 7. Gradient Descent
- Used to minimize loss
- Learning rate (\(\alpha\)) controls step size
- Types:
  - Batch
  - Stochastic (SGD)
  - Mini-batch

---

## 8. Feature Engineering
- Scaling (very important)
- Encoding categorical variables
- Handling missing values
- Feature selection

---

## 9. Feature Scaling
- Standardization → mean 0, std 1
- Normalization → range [0,1]
- Important for:
  - Gradient-based models
  - Distance-based models (KNN, SVM)

---

## 10. Regularization
- L1 (Lasso) → feature selection
- L2 (Ridge) → weight shrinkage
- Prevents overfitting

---

## 11. Evaluation Metrics

### Classification
- Accuracy → overall correctness
- Precision → correctness of positive predictions
- Recall → coverage of actual positives
- F1-score → balance of precision & recall

### Regression
- MSE → squared error
- RMSE → interpretable error
- MAE → absolute error

---

## 12. Cross Validation
- K-Fold → robust evaluation
- Helps avoid overfitting
- Used for hyperparameter tuning

---

## 13. Common Algorithms

### Regression
- Linear Regression
- Ridge / Lasso

### Classification
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest

### Clustering
- K-Means
- Hierarchical

---

## 14. KNN Key Idea
- Based on distance
- No training phase
- Sensitive to scaling

---

## 15. Decision Trees
- Split data based on features
- Prone to overfitting
- Random Forest reduces variance

---

## 16. SVM
- Finds maximum margin boundary
- Works well in high dimensions
- Kernel trick for non-linear data

---

## 17. Naive Bayes
- Based on probability
- Assumes feature independence
- Works well for text data

---

## 18. PCA (Dimensionality Reduction)
- Reduces number of features
- Keeps maximum variance
- Improves performance & visualization

---

## 19. Pipeline (Real Workflow)
1. Data Collection
2. Data Cleaning
3. EDA (Exploratory Data Analysis)
4. Feature Engineering
5. Model Training
6. Evaluation
7. Tuning
8. Deployment

---

## 20. Hyperparameters vs Parameters
- Parameters → learned (w, b)
- Hyperparameters → set manually (k, alpha, depth)

---

## 21. Learning Curves
- Diagnose:
  - Overfitting
  - Underfitting
- Plot: training vs validation error

---

## 22. Important Practices
- Always visualize data
- Check data leakage
- Use baseline model
- Start simple, then improve
- Use proper metrics

---

## 23. When to Use What

- Linear Regression → linear relationship
- Logistic Regression → binary classification
- KNN → small dataset
- SVM → high dimensional data
- Tree/Forest → non-linear data
- PCA → high dimensional features

---

## 24. Data Issues
- Missing values → impute/remove
- Outliers → detect & handle
- Imbalanced data → resampling, weights

---

## 25. Final Quick Reminders
- Garbage in → garbage out
- Data preprocessing is critical
- Simpler models often perform well
- Always validate properly
- Understand data before model

---