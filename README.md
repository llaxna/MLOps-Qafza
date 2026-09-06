# Task 02 — End-to-End Machine Learning Pipeline

## Overview

This task implements an end-to-end machine learning workflow for predicting whether an Olist e-commerce order will be delivered late.

The workflow follows a structured ML development process:

1. Read and understand the source data
2. Build an order-level machine learning dataset
3. Create the target label
4. Split the data into training, validation, and test sets
5. Perform exploratory data analysis
6. Engineer and preprocess features
7. Train, tune, and evaluate classification models

The project uses the **Brazilian E-Commerce Public Dataset by Olist** and focuses on predicting the `Late` delivery class.

The main objective is not only to train a model, but also to demonstrate good machine learning practices such as:

* avoiding data leakage
* handling one-to-many relationships correctly
* handling class imbalance
* separating training, validation, and test data
* fitting preprocessing transformations only on training data
* comparing models against a simple baseline
* saving trained artifacts for later reuse

---

# 1. Project Objective

The goal of this task is to build a classification model that predicts whether an order will be:

* **On Time**
* **Late**

The prediction target is based on the actual customer delivery date compared with the estimated delivery date.

An order is considered **Late** when:

```text
actual delivery date > estimated delivery date
```

Otherwise, it is classified as **On Time**.

Because late orders represent a minority of the dataset, the project treats this as an **imbalanced classification problem**.

Therefore, accuracy is not used as the only evaluation metric. Particular attention is given to:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

The **F1-score for the Late class** is used as the primary model-selection metric.

---

# 2. Dataset

The project uses the Olist Brazilian E-Commerce dataset.

The source data contains multiple related tables, including:

* `olist_orders`
* `olist_order_items`
* `olist_order_payments`
* `olist_order_reviews`
* `olist_products`
* `olist_sellers`
* `olist_customers`
* `olist_geolocation`
* `product_category_name_translation`

The tables have different grains and relationships.

For example, an order can contain multiple:

* order items
* payments
* products
* sellers

Therefore, these one-to-many tables are aggregated to the order level before being joined to the main order table.

The final machine learning dataset contains **one row per order**.

---

# 3. Notebook Structure

The task is divided into six notebooks:

```text
01_read_and_join.ipynb
02_create_label.ipynb
03_train_validation_test_split.ipynb
04_eda.ipynb
05_feature_engineering.ipynb
06_train_models.ipynb
```

Each notebook has a specific responsibility in the machine learning workflow.

---

# 4. Notebook 1 — Read and Join

## Objective

The first notebook focuses on understanding the source tables and creating a clean order-level machine learning dataset.

## Main activities

### Read the source tables

Each source table is loaded and inspected individually.

The following were checked:

* number of rows
* columns
* data types
* keys
* duplicates
* table grain
* relationships between tables

### Understand table grain

The main table is:

```text
olist_orders
```

because it contains one row per order.

Other tables contain multiple records per order.

For example:

```text
olist_order_items
    multiple rows → one order

olist_order_payments
    multiple rows → one order
```

These tables are aggregated to order level before joining them to `olist_orders`.

### Create order-level features

The following order-level aggregates were created:

* `item_count`
* `product_count`
* `seller_count`
* `total_price`
* `total_freight`
* `payment_count`
* `payment_value`
* `max_installments`

Customer information was also joined at the order level.

## Important design decision

The target label was **not** created in this notebook.

The purpose of Notebook 1 is only to build the clean machine learning table.

## Artifact

```text
artifacts/ml_table.parquet
```

The resulting table contains one row per order.

---

# 5. Notebook 2 — Create Label

## Objective

The second notebook creates the target variable for the classification problem.

## Label definition

The label is based on the actual customer delivery date and estimated delivery date.

```text
If actual delivery date > estimated delivery date:
    Late

Otherwise:
    On Time
```

The resulting target column is:

```text
label
```

with two possible values:

```text
On Time
Late
```

## Validation

The label logic was validated against actual orders to ensure that the generated label correctly represents the comparison between delivery dates.

## Class distribution

The resulting dataset contains:

```text
On Time: 91.89%
Late:     8.11%
```

The imbalance ratio is approximately:

```text
11.33 : 1
```

This means there are approximately 11.33 On Time orders for every Late order.

## Imbalance decision

The target is considered significantly imbalanced.

Therefore, later modeling should not rely on accuracy alone.

The following metrics are considered more appropriate:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

Particular attention is given to the ability to identify the minority `Late` class.

## Artifact

```text
artifacts/labeled_table.parquet
```

---

# 6. Notebook 3 — Train / Validation / Test Split

## Objective

The third notebook creates separate datasets for:

* training
* validation
* testing

The split is performed before detailed exploratory analysis and model development to reduce the risk of using test information when making modeling decisions.

## Split strategy

A **time-based split** was selected instead of a random split.

This decision was made because the real-world objective is to predict delivery performance for future orders.

A chronological split better represents this scenario:

```text
Past orders → Training
Later orders → Validation
Most recent orders → Test
```

The data was divided approximately as:

```text
70% Training
15% Validation
15% Test
```

## Resulting datasets

### Training

```text
Rows: 67,533

Date range:
2016-09-15 → 2018-04-15

On Time: 90.97%
Late:     9.03%
```

### Validation

```text
Rows: 14,471

Date range:
2018-04-15 → 2018-06-21

On Time: 94.66%
Late:     5.34%
```

### Test

```text
Rows: 14,472

Date range:
2018-06-21 → 2018-08-29

On Time: 93.39%
Late:     6.61%
```

The label distribution changes over time, so the same class ratio was not artificially forced across the splits.

This is appropriate for a chronological split because the changing distribution may represent real temporal variation.

## Validation checks

The split was checked to ensure:

* chronological ordering
* no overlapping order IDs
* correct row counts
* correct date ranges
* label distribution in each split

## Artifacts

```text
artifacts/train.parquet
artifacts/validation.parquet
artifacts/test.parquet
```

---

# 7. Notebook 4 — Exploratory Data Analysis

## Objective

The fourth notebook performs detailed exploratory data analysis using **only the training split**.

The validation and test sets are not used for exploratory decisions.

The purpose of the EDA is to understand:

* data quality
* missing values
* numerical distributions
* categorical variables
* relationships with the target
* temporal patterns
* geographic patterns
* potential useful features
* potential leakage

---

## Dataset structure

The training data contains:

```text
67,533 orders
```

Memory usage was approximately:

```text
20.76 MB
```

The dataset contains numerical, categorical, date, and identifier variables.

---

## Missing values

Missing values were found in only a small number of fields.

The main missing-value counts were:

| Column                         | Missing |
| ------------------------------ | ------: |
| `order_approved_at`            |      14 |
| `order_delivered_carrier_date` |       1 |
| `payment_count`                |       1 |
| `payment_value`                |       1 |
| `max_installments`             |       1 |

Missingness is therefore minimal.

Missing values are handled during feature engineering using appropriate imputation strategies.

---

# 8. Numerical Analysis

The main numerical variables include:

* `item_count`
* `product_count`
* `seller_count`
* `total_price`
* `total_freight`
* `payment_count`
* `payment_value`
* `max_installments`

The variables are generally right-skewed.

Some examples of extreme values include:

```text
total_price:     13,440
payment_value:   13,664.08
total_freight:    1,002.29
```

These values were not automatically removed because they may represent legitimate large orders.

No negative values were found in the main numerical variables.

---

# 9. Late vs On Time Analysis

The EDA showed that Late orders have somewhat higher average values for some financial variables.

For example:

| Feature         | Late Mean | On Time Mean |
| --------------- | --------: | -----------: |
| `total_price`   |    149.77 |       134.51 |
| `total_freight` |     25.20 |        21.95 |
| `payment_value` |    175.01 |       156.50 |

However, the median values are much closer.

This suggests that skewed distributions and high-value orders contribute to some of the observed differences.

---

# 10. Correlation Analysis

Several relationships were identified between numerical variables.

The strongest correlation was:

```text
total_price ↔ payment_value
Correlation ≈ 0.996
```

This indicates that these two variables contain highly similar information.

Other moderate relationships include:

```text
product_count ↔ seller_count
≈ 0.563

item_count ↔ total_freight
≈ 0.475

total_price ↔ total_freight
≈ 0.415
```

The strong correlation between `total_price` and `payment_value` should be considered during future feature engineering and model interpretation.

---

# 11. Geographic Analysis

Customer state showed meaningful differences in late-delivery rates.

Examples include:

```text
AL ≈ 27.85%
MA ≈ 21.53%
CE ≈ 18.25%
RJ ≈ 16.58%
```

while some states had lower rates, such as:

```text
RO ≈ 3.78%
AC ≈ 4.55%
AM ≈ 4.76%
SP ≈ 5.59%
```

These differences suggest that customer geography may contain predictive information.

However, states with small numbers of observations should be interpreted carefully because their estimated late rates can be unstable.

The current ML table contains customer geography but does not contain sufficient seller geography or coordinates to calculate reliable physical delivery distance.

Therefore, physical distance was not included.

---

# 12. Temporal Analysis

Purchase weekday was analyzed against the target.

Examples:

```text
Monday:    10.02% Late
Friday:     9.47% Late
Thursday:   8.29% Late
Sunday:     8.38% Late
```

The differences are relatively small.

Nevertheless, time-related features may provide useful information when combined with other predictors.

---

# 13. Delivery Duration and Leakage

A strong relationship was found between actual delivery duration and the target.

Late orders had an average delivery duration of approximately:

```text
34.51 days
```

while On Time orders had an average of:

```text
11.77 days
```

However, actual delivery duration was **not used as a predictive feature**.

The reason is that actual delivery duration depends on the actual delivery date, which is only known after the delivery has occurred.

Using it would therefore introduce **target leakage**.

This is an important modeling decision.

---

# 14. EDA Conclusions

The EDA suggested the following feature groups:

### Order characteristics

* item count
* product count
* seller count

### Financial characteristics

* total price
* total freight
* payment value
* installments

### Temporal characteristics

* purchase hour
* purchase day
* day of week
* month
* week of year
* weekend indicator

### Geographic characteristics

* customer state

### Delivery planning information

* estimated delivery window

The analysis also highlighted:

* strong class imbalance
* highly skewed numerical variables
* redundancy between `total_price` and `payment_value`
* potentially useful geographic information
* importance of avoiding delivery-date leakage

## Artifacts

EDA charts were saved under:

```text
artifacts/eda_charts/
```

---

# 15. Notebook 5 — Feature Engineering

## Objective

The fifth notebook transforms the raw order-level features into a model-ready feature matrix.

The most important requirement is that the model must only use information available at prediction time.

---

# 16. Time Features

Features were derived from:

```text
order_purchase_timestamp
```

including:

* `purchase_hour`
* `purchase_dayofweek`
* `purchase_month`
* `purchase_day`
* `purchase_weekofyear`
* `is_weekend`

These features allow the model to capture possible temporal patterns.

---

# 17. Estimated Delivery Feature

The following feature was created:

```text
estimated_delivery_days
```

It represents the estimated delivery window from the purchase timestamp to the estimated delivery date.

This feature can be used because the estimated delivery date is available before the actual delivery occurs.

---

# 18. Leakage Prevention

The following future information was excluded from the predictive features:

```text
order_delivered_customer_date
order_delivered_carrier_date
```

These columns contain information that would only become available after the delivery process has progressed.

They could therefore leak information about the target.

Identifiers were also excluded:

```text
order_id
customer_id
customer_unique_id
```

because they do not represent useful generalizable predictive characteristics.

---

# 19. Categorical Encoding

The following categorical variable was used:

```text
customer_state
```

It was transformed using one-hot encoding.

Unknown categories are handled using:

```text
handle_unknown="ignore"
```

This allows the same preprocessing pipeline to safely process new categories during inference.

---

# 20. Missing Value Handling

Numerical features use:

```text
Median Imputation
```

Categorical features use:

```text
Most-Frequent Imputation
```

This ensures that the final feature matrix contains no missing values.

---

# 21. Numerical Scaling

Numerical features were standardized using:

```text
StandardScaler
```

The scaler was fitted using the **training data only**.

The same fitted transformation was then applied to:

* validation
* test

This prevents validation and test information from influencing preprocessing.

---

# 22. Final Feature Set

The final feature matrix contains:

```text
42 features
```

The resulting datasets are:

```text
Train:
67,533 rows × 43 columns

Validation:
14,471 rows × 43 columns

Test:
14,472 rows × 43 columns
```

The additional column is the target label.

No missing values remain in the final feature tables.

---

# 23. Saved Preprocessing Artifacts

The fitted preprocessing pipeline was saved as:

```text
artifacts/features/preprocessor.joblib
```

The final feature names were saved as:

```text
artifacts/features/feature_list.csv
```

The transformed datasets were saved as:

```text
artifacts/features/train_features.parquet
artifacts/features/validation_features.parquet
artifacts/features/test_features.parquet
```

Saving the fitted preprocessing pipeline is important because production inference must use the **same transformations** used during model training.

The preprocessing pipeline should not be refitted on new production data.

---

# 24. Notebook 6 — Train, Tune, Evaluate

## Objective

The sixth notebook trains classification models, compares them against a simple baseline, tunes the strongest candidate, and evaluates the final model.

The main challenge is the severe class imbalance.

---

# 25. Baseline Model

A majority-class baseline was created using:

```text
DummyClassifier(strategy="most_frequent")
```

The baseline predicts every order as:

```text
On Time
```

This results in high accuracy but zero ability to detect Late orders.

Validation results:

```text
Accuracy:  94.66%
Precision: 0.00
Recall:    0.00
F1:        0.00
ROC-AUC:   0.50
PR-AUC:    0.0534
```

This demonstrates why accuracy alone is misleading for this problem.

---

# 26. Logistic Regression

A Logistic Regression model was trained using:

```text
class_weight="balanced"
```

This gives additional importance to the minority Late class.

The initial validation performance was:

```text
Accuracy:  68.49%
Precision: 10.68%
Recall:    66.49%
F1:        18.40%
ROC-AUC:   0.749
PR-AUC:    0.137
```

The model was substantially better at identifying Late orders than the majority baseline.

---

# 27. Random Forest

A Random Forest classifier was also evaluated.

The initial model achieved:

```text
Accuracy: 94.38%
Precision: 9.80%
Recall:    0.65%
F1:        1.21%
ROC-AUC:   0.642
PR-AUC:    0.082
```

Although its accuracy was high, it performed poorly at identifying the minority Late class.

---

# 28. Model Comparison

The validation results showed that Logistic Regression performed substantially better than Random Forest for the primary metric.

The Random Forest was therefore not the preferred model for this problem.

The main reason is that:

```text
High accuracy ≠ good minority-class detection
```

The Logistic Regression model achieved much stronger recall and F1-score for Late orders.

---

# 29. Logistic Regression Tuning

The Logistic Regression regularization parameter `C` was tuned using the validation set.

The tested values were:

```text
C = 0.01
C = 0.10
C = 1.00
C = 10.00
```

All configurations used:

```text
class_weight = balanced
```

Validation results:

|         C |  Precision |     Recall |         F1 | ROC-AUC | PR-AUC |
| --------: | ---------: | ---------: | ---------: | ------: | -----: |
|      0.01 |     0.1045 |     0.6727 |     0.1809 |  0.7480 | 0.1351 |
|      0.10 |     0.1052 |     0.6727 |     0.1820 |  0.7509 | 0.1380 |
|      1.00 |     0.1068 |     0.6649 |     0.1840 |  0.7490 | 0.1373 |
| **10.00** | **0.1073** | **0.6468** | **0.1841** |  0.7450 | 0.1353 |

The best configuration according to the selected primary metric, F1-score, was:

```text
C = 10.0
class_weight = balanced
```

---

# 30. Final Model

The final selected model is:

```text
Logistic Regression
```

with:

```text
C = 10.0
class_weight = balanced
max_iter = 1000
```

The trained model was saved as:

```text
artifacts/models/final_model.joblib
```

---

# 31. Validation Performance

The final Logistic Regression achieved the following validation performance:

```text
Accuracy:  69.38%
Precision: 10.73%
Recall:    64.68%
F1:        18.41%
ROC-AUC:   0.7450
PR-AUC:    0.1353
```

For the Late class:

```text
Precision: 10.7%
Recall:    64.7%
F1-score:  18.4%
```

The model therefore identifies a substantial proportion of Late orders, but produces a relatively high number of false positives.

---

# 32. Final Test Performance

The final model was evaluated on the held-out test data.

Test results:

```text
Accuracy:  44.15%
Precision: 9.48%
Recall:    87.04%
F1:        17.09%
ROC-AUC:   0.6733
PR-AUC:    0.1156
```

For the Late class:

```text
Precision: 9.5%
Recall:    87.0%
F1-score:  17.1%
```

The model successfully identifies most Late orders, but its precision is low.

This indicates a significant trade-off between:

```text
Detecting Late orders
        vs.
Avoiding false Late predictions
```

---

# 33. Interpretation of the Final Model

The final model should not be judged primarily by its accuracy.

The test accuracy is only:

```text
44.15%
```

but the model achieves:

```text
87.04% recall for Late orders
```

This means that the model identifies most of the actual Late orders.

However, the precision of:

```text
9.48%
```

means that many orders predicted as Late are actually On Time.

Therefore, the current model is better interpreted as a **high-recall baseline for detecting potentially late orders**, rather than a highly precise late-delivery predictor.

---

# 34. Model Artifacts

The main artifacts generated throughout Task 02 are:

```text
artifacts/
│
├── ml_table.parquet
├── labeled_table.parquet
├── train.parquet
├── validation.parquet
├── test.parquet
│
├── eda_charts/
│
├── features/
│   ├── train_features.parquet
│   ├── validation_features.parquet
│   ├── test_features.parquet
│   ├── preprocessor.joblib
│   └── feature_list.csv
│
├── models/
│   └── final_model.joblib
│
└── model_results/
    ├── results_summary.csv
    └── final_test_results.csv
```

---

# 35. Machine Learning Workflow

The complete workflow can be summarized as:

```text
Raw Olist Tables
       │
       ▼
01 — Read & Join
       │
       ▼
Order-Level ML Table
       │
       ▼
02 — Create Label
       │
       ▼
Labeled Dataset
       │
       ▼
03 — Train / Validation / Test Split
       │
       ├──────────────┐
       ▼              ▼
   Train          Validation
       │              │
       └──────┬───────┘
              ▼
04 — EDA
              │
              ▼
05 — Feature Engineering
              │
              ▼
42 Model Features
              │
              ▼
06 — Train & Tune
              │
       ┌──────┴──────┐
       ▼             ▼
   Baseline      ML Models
                     │
                     ▼
              Model Selection
                     │
                     ▼
             Final Logistic Regression
                     │
                     ▼
              Final Test Evaluation
```

---

# 36. Key ML Practices Demonstrated

## Avoiding target leakage

Actual delivery information was excluded from the predictive features.

## Correct handling of one-to-many relationships

Order items and payments were aggregated before joining to the order-level table.

## Time-based splitting

The dataset was split chronologically to better represent future prediction.

## Training-only preprocessing

Transformers were fitted only on training data.

## Class imbalance handling

`class_weight="balanced"` was used for the classification models.

## Appropriate evaluation metrics

F1, precision, recall, ROC-AUC, and PR-AUC were used instead of relying only on accuracy.

## Baseline comparison

A majority-class baseline was established before evaluating machine learning models.

## Artifact persistence

The preprocessing pipeline and trained model were saved so they can be reused later.

---

# 37. Final Findings

The main findings from Task 02 are:

1. The dataset has a significant class imbalance, with Late orders representing a minority of observations.

2. A majority-class model can achieve high accuracy while completely failing to detect Late orders.

3. Logistic Regression performed substantially better than Random Forest for detecting the Late class.

4. Customer geography, order characteristics, financial variables, and temporal features appear to contain useful predictive information.

5. `total_price` and `payment_value` are highly correlated and contain redundant information.

6. Numerical variables are generally right-skewed and contain some extreme but potentially valid observations.

7. Actual delivery duration is strongly related to the target but cannot be used as a predictive feature because it introduces target leakage.

8. The final Logistic Regression model achieves high recall for Late orders but relatively low precision.

9. The test F1-score of 0.1709 is close to the validation F1-score of 0.1841, although the precision/recall balance changes between the two datasets.

10. The current model should be considered a baseline that can be improved through additional feature engineering, alternative models, and decision-threshold optimization.

---

# 38. Future Improvements

Potential improvements for future iterations include:

* additional time-based features
* richer seller/customer geographic features
* seller-to-customer distance
* estimated delivery slack
* order approval delay, when available at prediction time
* seller performance history
* customer historical behavior
* seasonal features
* alternative classification algorithms
* threshold optimization
* cross-validation within the training data
* more systematic hyperparameter optimization
* calibration of predicted probabilities
* additional imbalance-handling techniques

Any future feature should still be checked carefully to ensure that it was available at the intended prediction time.

---

# 39. Conclusion

Task 02 successfully implements a complete machine learning workflow from raw relational data through final model evaluation.

The project demonstrates the importance of understanding data grain, preventing leakage, separating training and evaluation data, handling class imbalance, and using appropriate metrics.

The final model is a balanced Logistic Regression classifier selected based on validation F1-score.

While its precision is relatively low, its high recall demonstrates that it can identify a large proportion of potentially late orders. The results provide a useful baseline for future iterations of the project.

The saved preprocessing pipeline and trained model also provide the foundation for the next stages of the MLOps workflow, where the model can be packaged, served, monitored, and eventually deployed.
