# electrical_load_forecast contains data driven forecasting LSTM models. Electrical Demand data was scraped using beautiful soup from AEMO's website & weather data was scraped from a weather forecasting website for Australia's Victoria State. Exploratory data analysis was done using Pandas to identify influencing factors in forecast, matrix calculation was done using Numpy and hyperparameter tuning of model was done using Keras. MAPEs were calculated for several LSTM models and the best model was chosen based on MAPE values.  