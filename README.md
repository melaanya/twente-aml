# Final project 

## Description

A recently appeared library **CatBoost** for gradient boosting on decision trees is claimed to outperform existing implementations by special processing of categorical variables and improved process of gradient estimation. 
It is alleged to surpass existing state-of-the-art models on several benchmarks for classification tasks with no regards to any regression problems. 

This project investigates the applicability of CatBoost  in the regression problem of housing price prediction. 
We compare the CatBoost performance with the performance of two other state-of-the-art gradient boosting libraries such as **sklearn** and **XGBoost** by training and evaluating them on three datasets with different properties. We also analyze categorical feature importance and their influence on performance.

## Results

### [Sberbank Dataset](https://www.kaggle.com/c/sberbank-russian-housing-market/data)

| **Model**   | **Default**   |   **Tuned** |  **Max\_depth, n\_estimators, learning rate** |  **Training time**|
| -------- | --------- | ------- |  --------- |  --------|
| CatBoost  |  0.5161 |      0.5102         |             3, 100, 0.01   |            765.91 |
| XGBoost     |        **0.5098** |      **0.5081**          |        2, 500, 0.05   |    158.67    |
|  sklearn GB   |   0.5102  |   0.5091     |          1, 1000, 0.1   |      **148.41**   |


### [Ames Dataset](https://www.kaggle.com/c/housing-data/data)

| **Model**   | **Default**   |   **Tuned** |  **Max\_depth, n\_estimators, learning rate** |  **Training time**|
| -------- | --------- | ------- |  --------- |  --------|
| CatBoost  |  0.1789 |      0.1435         |             1, 1000, 0.1   |            58.82 |
| XGBoost     |        0.1315 |      0.1246          |                          2, 500, 0.1   |         **8.72** |
|  sklearn GB   |   **0.1285**  | **0.1241**          |                         1, 1000, 0.1   |           10.10  |


### [Boston Dataset](https://www.kaggle.com/c/boston-housing/data)

| **Model**   | **Default**   |   **Tuned** |  **Max\_depth, n\_estimators, learning rate** |  **Training time**|
| -------- | --------- | ------- |  --------- |  --------|
| CatBoost  |  **0.1584** |  **0.1423**              |                    1, 1000, 0.01       |        56.33 |
| XGBoost     |        0.1952   |    0.1753           |                       2, 1000, 0.01    |        **0.75**  |
|  sklearn GB   |   0.1993    |   0.1721   |                                  2, 1000, 0.01    |         1.07     |


## Conclusions

CatBoost outperforms two models only on Boston dataset with the smallest number of both features and training instances while still conceding to other models in more realistic Ames and Sberbank datasets. Further analysis shows CatBoost's apparent preference towards the categorical variables when building the estimators which does not always result in score improvements.
