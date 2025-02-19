
# HEART DISEASE PREDICTION WITH ADA BOOST AND RANDOM FOREST

This group project aims to compare the performance of two popular ensemble learning algorithms—AdaBoost and Random Forest—on a specific dataset under defined conditions. In addition to the algorithm comparison, we incorporate a feature selection process to evaluate and quantify its impact on model performance. The project is implemented in Python, leveraging libraries such as scikit-learn, pandas, and numpy.


## Data Source

 https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction

 `Dataset Details`
    
    1. Age: Patient's age in years.
    2. Sex: Patient's gender.
        1: Male
        2: Female
    3. Chest Pain Type:
        1: Typical angina
        2: Atypical angina
        3: Non-anginal pain
        4: Asymptomatic
    4. BP: Blood pressure at rest (mm Hg).
    5. Cholesterol: Total cholesterol in the blood vessels (mg/dL).
    6. FBS over 120: Fasting blood sugar level over 120 mg/dL.
        1: True
        2: False
    7. EKG Results: Resting electrocardiographic results.
        0: Normal
        1: Abnormal ST-T wave (T wave inversion and/or ST segment elevation or depression > 0.05 mV)
        2: Indicates probable or definite left ventricular hypertrophy based on Estes' criteria
    8. Max HR: Maximum heart rate.
    9. Exercise Angina: Angina induced by exercise.
        1: True
        0: False
    10. ST Depression: ST depression induced by exercise relative to rest.
    11. Slope of ST: Slope of the peak exercise ST segment.
        1: Upsloping
        2: Flat
        3: Downsloping
    12. Number of Vessels: Number of major vessels (0-3) colored by fluoroscopy.
    13. Thallium:
        3: Normal
        6: Fixed defect
        7: Reversible defect
    14. Heart Disease: Classification result.
        present: Heart/cardiovascular disease detected
        absent: No heart/cardiovascular disease detected




## Project Goal
- **Compare Two Algorithms** : Evaluate AdaBoost and Random Forest to determine which algorithm best suits the given conditions and dataset.
- **Assess Feature Selection Impact** : Analyze how feature selection affects model accuracy and other performance metrics by comparing models built with the full feature set against those using a reduced feature set.




## Tech Stack

- Python
- Google Colaboratory    
- Pandas
- Numpy
- Sklearn



## Evaluation Process

`Evaluation Model`

```python
    def evaluate(model, X_validation, y_validation):
        start_time = time.time() 
        process = psutil.Process()

        y_prediction = model.predict(X_validation)
        y_proba = model.predict_proba(X_validation)[:, 1]

        end_time = time.time() 

        accuracy = accuracy_score(y_validation, y_prediction) 
        precision = precision_score(y_validation, y_prediction, pos_label=1)
        recall = recall_score(y_validation, y_prediction, pos_label=1) 
        f1 = f1_score(y_validation, y_prediction, pos_label=1) 
        auc = roc_auc_score(y_validation, y_proba) 
        execution_time = end_time - start_time 

        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-Score: ", f1)
        print("AUC: ", auc)
        print("Execution Time (s): ", execution_time)
```
`Feature Selection Model (Top 5)`

```python
    correlations = data.corr()['Heart Disease'].sort_values(ascending=False).reset_index()
    correlations = correlations[correlations['index'] != 'Heart Disease']

    top_features = correlations['index'].head(5)
    print(top_features)
```    

| Features Rank  | Features Name  | 
| :-------------- | :------- | 
| `1`       | `Thallium`| 
| `2` | `Chest pain type` | 
| `3`       | `Number of vessels fluro` | 
| `4` | `ST depression` | 
| `5`       | `Exercise angina` |

## Project Output
- RF-FF = Random Forest Full Feature
- AB-FF = AdaBoost Full Feature
- RF-FS = Random Forest Feature Selection
- AB-FS = AdaBoost Feature Selection

| Evaluation   | RF-FF  | AB-FF  | RF-FS | AB-FS |
| :-------------- | :------- | :----------  | :--------- | :----------- |
| `Accuracy`  | `0.909`| `0.818`| `0.863`| `0.909`|
| `Precision` | `0.941` | `0.923`| `0.933`        | `0.941`        |
| `Recall`    | `0.842` | `0.631`| `0.736`        | `0.842`        |
| `F1-Score` | `0.888` | `0.749`| `0.823`        | `0.888`        |
| `AUC`   | `0.946` | `0.919`| `0.958` | `0.863`        |
| `Execution Time (s)`| `0.024` | `0.015`| `0.013`        | `0.012 `        |


`For Full Discussion, Please Read :` [Paper (Indonesia Language)](https://drive.google.com/file/d/1BkcmqolLDl5qX2z1hmjNkqe59oRaY8-O/view)
