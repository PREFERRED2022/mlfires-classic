# Predicting next day fire risk with ensemble tree models and Neural Networks

Software for training, validating and testing models for predicting next day fire risk\
The models are derived from the algorithms RF, ExtraTrees, XGBoost and Wide and Deep Neural Networks

## Train-validation and Testing

The training validation is performed on a sampled dataset of historical data from years 2010 to 2018 and the testing is evaluated on 2019 and 2020 years, which are "unseen" real world datasets.\
n-fold cross valitation and hyperparameterization is employed for training and tuning of all the models.\
The optimization of the hyperparameters is done using the *Tree Parzen Estimator* method of the hyperopt library.

Under /train\_validation_test the basic modules that handle the cross validation and hyperparameterization and testing are:

* *fires\_hyperopt\_newdata.py* : Handles the cross validation and hyperparameterization on the training dataset for all the algorithms using the space_new.py parameter space and records performance for each model.
* *fires\_test\_cl.py* : Runs the test of the best models on the real world datasets and records their performances.

## Performance

Under /reporting the notebook *bestalltest.ipynb* is used to evaluate the models with best performance displying tables with selected metrics.

Under /findbestbins the module *checkscores.ipynb* and the module *preferred_reports.ipynb* present the performance of predicting fire in five risk levels based on probability for 2019 test data and 2023 pilot operational run.

## Explainability

Under /xai folder explainability methods SHAP (and Couterfactuals experimentaly) are employed to produce local explainability of the algorithms on a representative sample of pixels for each daily dataset.



