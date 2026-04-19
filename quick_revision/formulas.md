# Machine Learning – Quick Revision Formulas Cheat Sheet

---

## 1. Linear Regression

### Hypothesis
\[
\hat{y}^{(i)} = w^T x^{(i)} + b
\]

### Cost Function (MSE)
\[
L = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
\]

### Gradient Descent Updates
\[
w := w - \alpha \frac{\partial L}{\partial w}
\]
\[
b := b - \alpha \frac{\partial L}{\partial b}
\]

#### Terms
- \( x^{(i)} \): input features
- \( y^{(i)} \): true value
- \( \hat{y}^{(i)} \): predicted value
- \( w \): weights
- \( b \): bias
- \( m \): number of samples
- \( \alpha \): learning rate

#### Usage
- Continuous output prediction

---

## 2. Logistic Regression

### Hypothesis (Sigmoid)
\[
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b
\]

### Cost Function (Binary Cross Entropy)
\[
L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
\]

#### Usage
- Binary classification

---

## 3. Softmax (Multiclass Classification)

\[
P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
\]

### Cross Entropy Loss
\[
L = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
\]

#### Usage
- Multi-class classification

---

## 4. Regularization

### Ridge (L2)
\[
L = \text{MSE} + \lambda \sum_{j=1}^{n} w_j^2
\]

### Lasso (L1)
\[
L = \text{MSE} + \lambda \sum_{j=1}^{n} |w_j|
\]

### Elastic Net
\[
L = \text{MSE} + \lambda_1 \sum |w_j| + \lambda_2 \sum w_j^2
\]

#### Terms
- \( \lambda \): regularization strength

#### Usage
- Prevent overfitting
- Feature selection (Lasso)

---

## 5. Gradient Descent Variants

### Batch GD
\[
w := w - \alpha \nabla L(w)
\]

### Stochastic GD (SGD)
- Update per sample

### Mini-batch GD
- Update per batch

---

## 6. Bias-Variance Tradeoff

\[
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
\]

#### Usage
- Model selection and tuning

---

## 7. Evaluation Metrics

### Accuracy
\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

### Precision
\[
\text{Precision} = \frac{TP}{TP + FP}
\]

### Recall
\[
\text{Recall} = \frac{TP}{TP + FN}
\]

### F1 Score
\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

---

## 8. Confusion Matrix

| Actual \ Predicted | Positive | Negative |
|-------------------|----------|----------|
| Positive          | TP       | FN       |
| Negative          | FP       | TN       |

---

## 9. K-Nearest Neighbors (KNN)

### Distance (Euclidean)
\[
d(x, x') = \sqrt{\sum_{j=1}^{n} (x_j - x'_j)^2}
\]

#### Usage
- Instance-based learning

---

## 10. Naive Bayes

### Bayes Theorem
\[
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
\]

### Naive Assumption
\[
P(x_1, x_2, ..., x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)
\]

#### Usage
- Text classification

---

## 11. Support Vector Machine (SVM)

### Decision Boundary
\[
w^T x + b = 0
\]

### Hinge Loss
\[
L = \max(0, 1 - y(w^T x + b))
\]

#### Usage
- Classification with margin maximization

---

## 12. Decision Tree (Split Criteria)

### Gini Impurity
\[
G = 1 - \sum_{k=1}^{K} p_k^2
\]

### Entropy
\[
H = -\sum_{k=1}^{K} p_k \log p_k
\]

#### Usage
- Tree-based models

---

## 13. Principal Component Analysis (PCA)

### Covariance Matrix
\[
\Sigma = \frac{1}{m} X^T X
\]

### Projection
\[
Z = X W
\]

#### Usage
- Dimensionality reduction

---

## 14. Cross Validation

### K-Fold CV
\[
\text{Error} = \frac{1}{K} \sum_{i=1}^{K} L_i
\]

#### Usage
- Model evaluation

---

## 15. Learning Rate Decay

### Exponential Decay
\[
\alpha_t = \alpha_0 e^{-kt}
\]

---

## 16. Neural Networks

### Forward Pass
\[
a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})
\]

### Loss (MSE)
\[
L = \frac{1}{m} \sum (y - \hat{y})^2
\]

### Backpropagation
\[
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial W}
\]

---

## 17. Activation Functions

### Sigmoid
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### ReLU
\[
f(x) = \max(0, x)
\]

### Tanh
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

---

## 18. Overfitting vs Underfitting

- Overfitting: low bias, high variance
- Underfitting: high bias, low variance

---

## 19. Standardization

\[
z = \frac{x - \mu}{\sigma}
\]

#### Terms
- \( \mu \): mean
- \( \sigma \): standard deviation

---

## 20. Normalization (Min-Max)

\[
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]

---

## Key Reminders

- Always scale data before training
- Use cross-validation for reliable evaluation
- Regularization helps generalization
- Choose metric based on problem type
- Watch learning rate carefully

---