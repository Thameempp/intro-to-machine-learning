# Machine Learning – Detailed Comparison Tables (Deep Revision)

---

## 1. Supervised vs Unsupervised vs Reinforcement Learning

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|-------|--------------------|----------------------|------------------------|
| Data Type | Labeled (X, y) | Unlabeled (X only) | Interaction data (state, action, reward) |
| Goal | Learn mapping \( f(X) \rightarrow y \) | Discover hidden patterns | Learn optimal policy |
| Output | Prediction (class/value) | Clusters, embeddings | Actions/strategy |
| Feedback | Direct (loss) | No direct feedback | Delayed reward |
| Algorithms | Linear Reg, Logistic, SVM | K-Means, PCA, DBSCAN | Q-Learning, DQN |
| Example | Spam detection | Customer segmentation | Game playing |
| When to Use | When labels available | When labels unavailable | Sequential decision problems |

---

## 2. Regression vs Classification

| Aspect | Regression | Classification |
|--------|-----------|---------------|
| Output Type | Continuous | Discrete (classes) |
| Hypothesis | \( \hat{y} \in \mathbb{R} \) | \( \hat{y} \in \{0,1,...\} \) |
| Loss Functions | MSE, MAE | Cross-Entropy, Hinge |
| Evaluation | RMSE, MAE, \(R^2\) | Accuracy, Precision, Recall |
| Examples | Price prediction | Disease detection |
| Boundary | No boundary | Decision boundary |
| Complexity | Often simpler | Can be complex |

---

## 3. L1 vs L2 vs Elastic Net (Regularization)


## 3. L1 vs L2 vs Elastic Net (Regularization)

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|------------|------------|-------------|
| Penalty | \( \lambda \sum_{j=1}^{n} |w_j| \) | \( \lambda \sum_{j=1}^{n} w_j^2 \) | \( \lambda_1 \sum_{j=1}^{n} |w_j| + \lambda_2 \sum_{j=1}^{n} w_j^2 \) |
| Effect on Weights | Sparse (many \( w_j = 0 \)) | Shrinks weights (small but non-zero) | Combination of both (sparse + shrinkage) |
| Feature Selection | Yes | No | Yes (controlled) |
| Handles Multicollinearity | Poor | Good | Best |
| Geometry | Diamond (L1 ball) | Circle/Sphere (L2 ball) | Hybrid (between L1 and L2) |
| When to Use | High-dimensional sparse data | Correlated features | General-purpose (mix of both) |
---

## 4. Gradient Descent Variants

| Type | Update Rule | Pros | Cons | Use Case |
|------|------------|------|------|----------|
| Batch GD | Full dataset | Stable convergence | Slow | Small datasets |
| SGD | One sample | Fast, escapes local minima | Noisy | Online learning |
| Mini-batch | Small batch | Balanced | Needs tuning | Deep learning |

---

## 5. Classification Metrics (Detailed)

| Metric | Formula | Focus | Use Case |
|--------|--------|------|----------|
| Accuracy | \( \frac{TP+TN}{Total} \) | Overall correctness | Balanced data |
| Precision | \( \frac{TP}{TP+FP} \) | False positives | Spam detection |
| Recall | \( \frac{TP}{TP+FN} \) | False negatives | Medical diagnosis |
| F1 Score | Harmonic mean | Balance | Imbalanced datasets |
| ROC-AUC | Area under curve | Ranking quality | Model comparison |

---

## 6. Regression Metrics (Detailed)

| Metric | Formula | Interpretation | Sensitivity |
|--------|--------|---------------|-------------|
| MSE | Avg squared error | Penalizes large errors | High |
| RMSE | \( \sqrt{MSE} \) | Same unit as y | High |
| MAE | Avg absolute error | Robust | Low |
| \(R^2\) | Variance explained | Model fit quality | Medium |

---

## 7. KNN vs SVM vs Decision Tree (Deep Comparison)

| Aspect | KNN | SVM | Decision Tree |
|--------|-----|-----|---------------|
| Learning Type | Lazy | Eager | Eager |
| Training Time | None | Medium | Fast |
| Prediction Time | High | Low | Low |
| Decision Boundary | Non-linear | Linear/non-linear | Axis-aligned |
| Scaling Required | Yes | Yes | No |
| Interpretability | Low | Medium | High |
| Overfitting | Low (large k) | Low | High |
| Memory | High | Medium | Low |

---

## 8. Decision Tree vs Random Forest vs Gradient Boosting

| Aspect | Decision Tree | Random Forest | Gradient Boosting |
|--------|--------------|---------------|-------------------|
| Model Type | Single tree | Bagging ensemble | Boosting ensemble |
| Overfitting | High | Low | Medium |
| Bias | Low | Medium | Low |
| Variance | High | Low | Medium |
| Training | Fast | Medium | Slow |
| Accuracy | Moderate | High | Very High |
| Examples | CART | RF | XGBoost |

---

## 9. Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|--------|----------|
| Training | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias |
| Weighting | Equal | Adaptive |
| Sensitivity to Noise | Low | High |
| Example | Random Forest | AdaBoost, XGBoost |

---

## 10. K-Means vs Hierarchical vs DBSCAN

| Aspect | K-Means | Hierarchical | DBSCAN |
|--------|--------|--------------|--------|
| Need K | Yes | No | No |
| Shape | Spherical | Any | Arbitrary |
| Noise Handling | Poor | Poor | Good |
| Speed | Fast | Slow | Medium |
| Output | Clusters | Dendrogram | Clusters + noise |
| Use Case | Simple clustering | Data hierarchy | Outlier detection |

---

## 11. PCA vs Feature Selection

| Aspect | PCA | Feature Selection |
|--------|-----|-------------------|
| Method | Linear transformation | Subset selection |
| Output | New features | Original features |
| Variance | Maximized | Not guaranteed |
| Interpretability | Low | High |
| Use Case | Compression | Simplicity |

---

## 12. Bias vs Variance Tradeoff

| Aspect | Bias | Variance |
|--------|------|----------|
| Cause | Simplifying assumptions | Sensitivity to data |
| Effect | Underfitting | Overfitting |
| Error Type | Systematic | Random |
| Fix | Increase model complexity | Regularization, more data |

---

## 13. Hyperparameters vs Parameters

| Aspect | Parameters | Hyperparameters |
|--------|-----------|----------------|
| Learned | Yes | No |
| Examples | Weights (w), bias (b) | Learning rate, k |
| Optimization | Gradient descent | Grid/Random search |

---

## 14. Train vs Validation vs Test

| Dataset | Role | Important Note |
|--------|------|----------------|
| Train | Fit model | Largest portion |
| Validation | Tune model | Used multiple times |
| Test | Final evaluation | Use once only |

---

## 15. Normalization vs Standardization

| Aspect | Normalization | Standardization |
|--------|--------------|-----------------|
| Formula | Min-Max scaling | Z-score |
| Range | [0,1] | Unbounded |
| Sensitive to Outliers | Yes | Less |
| Use Case | Neural nets, images | General ML |

---

## 16. Generative vs Discriminative Models

| Aspect | Generative | Discriminative |
|--------|-----------|----------------|
| Learn | \( P(X, Y) \) | \( P(Y \mid X) \) |
| Output | Joint distribution | Decision boundary |
| Examples | Naive Bayes | Logistic Regression |
| Advantage | Handles missing data | Better accuracy |

---

## 17. Online vs Batch Learning

| Aspect | Online | Batch |
|--------|--------|-------|
| Data Flow | Streaming | Static |
| Update | Incremental | Full retrain |
| Memory | Low | High |
| Use Case | Real-time systems | Offline training |

---

## 18. Distance Metrics

| Metric | Formula | Best For |
|--------|--------|----------|
| Euclidean | \( \sqrt{\sum (x_i - y_i)^2} \) | Continuous data |
| Manhattan | \( \sum |x_i - y_i| \) | High dimensions |
| Cosine | \( \frac{x \cdot y}{||x|| ||y||} \) | Text/sparse data |

---

## 19. Activation Functions

| Function | Formula | Range | Use Case |
|----------|--------|-------|----------|
| Sigmoid | \( \frac{1}{1+e^{-x}} \) | (0,1) | Binary output |
| ReLU | \( \max(0,x) \) | [0,∞) | Deep learning |
| Tanh | \( \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | (-1,1) | Centered data |

---

## 20. Key Summary (High-Yield)

| Concept | Key Insight | Practical Tip |
|--------|------------|---------------|
| Overfitting | Memorization | Use regularization |
| Underfitting | Too simple | Increase complexity |
| Scaling | Essential | Always preprocess |
| Cross-validation | Reliable metric | Use K-Fold |
| Feature Engineering | Most important | Spend time here |
| Simplicity | Often better | Start simple |

---