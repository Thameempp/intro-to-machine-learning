# Anomaly Detection Overview

Anomaly detection identifies unusual or rare data points that deviate significantly from normal patterns.

## Key Concepts

- **Outlier/Anomaly**: Data point that differs significantly from others
- **Normal Behavior**: Typical pattern in the data
- **Deviation**: Measure of how different a point is from the normal pattern

## Algorithms Covered

1. **Isolation Forest**: Builds isolated trees to separate anomalies
2. **Local Outlier Factor (LOF)**: Density-based method comparing local densities
3. **One-Class SVM**: Support Vector Machine for identifying outliers

## Characteristics

- **Unsupervised**: Works without labeled anomaly data
- **Scalability**: Different algorithms for different data sizes
- **Interpretability**: Understanding why something is anomalous

## Use Cases

- Fraud detection
- Network intrusion detection
- Equipment failure prediction
- Disease outbreak detection
- Manufacturing quality control

## When to Use Each

- **Isolation Forest**: Fast, efficient for high-dimensional data
- **LOF**: Good for local density-based anomalies
- **One-Class SVM**: Flexible, works with various kernels
