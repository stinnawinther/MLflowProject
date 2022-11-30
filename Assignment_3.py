import pandas as pd
import mlflow
import numpy as np
from matplotlib import pyplot as plt 

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

mlflow.set_tracking_uri("http://training.itu.dk:5000/")

# TODO: Set the experiment name
mlflow.set_experiment("stwi - experiment_assignment1")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.

#Load data:
with mlflow.start_run(run_name="assignment1.0"):
    # TODO: Insert path to dataset
    df = pd.read_json("dataset.json", orient="split")

    # TODO: Handle missing data
    #drop missing data.
    df = df.dropna()

    # Check if they are removed:
    df.isnull().sum()

    X = df[['Speed', 'Direction']]
    y = df['Total']

    #Define model and parameters:

    #As the wind direction is a string, it has to be altered to be a usable feature in the model.
    #The categories of wind direction can not be ordered. Therefore, I use HotOneEndcoder to convert each category to coulmns with binary attributes:

    # instantiating and transforming the 'Direction' feature:

    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse = False)

    #transform the column ('Direction'):

    ct = ColumnTransformer([('encoder transformer', cat_encoder, ['Direction'])], remainder="passthrough")


    # Create a LinearRegression object 
    linreg_model = LinearRegression()

    #Create polynominal model:
    Degree = 3
    poly = PolynomialFeatures(degree=Degree, include_bias=False)

    mlflow.log_param("Degree", Degree)

    ### Transformation Pipeline

    # TODO: You can start with your pipeline from assignment 1
    Input=[('my_ct', ct), ('minMax', MinMaxScaler()), ('poly', poly), ('lr', linreg_model)]

    pipeline = Pipeline(Input)
    
    ##Training

    print(len(X))
    print(len(y))
    #TODO: Log your parameters. What parameters are important to log?
    # HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  

    pipeline_model = pipeline.fit(X_train, y_train) #train the model
    pred = pipeline_model.predict(X_test)

    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    
    MAE = mean_absolute_error(y_test, pred)
    mlflow.log_metric("MAE", MAE)
    print("MAE", MAE)

    R2 = r2_score(y_test, pred)
    mlflow.log_metric("R2", R2)
    print("R2", R2)

    MSE = mean_squared_error(y_test, pred)
    mlflow.log_metric("MSE", MSE)
    print("MSE", MSE)

    RMSE = np.sqrt(MSE)
    mlflow.log_metric("RMSE", RMSE)
    print("RMSE", RMSE)

    #print("log the trained model")
    #mlflow.sklearn.log_model(pipeline_model, "model")
    
#mlflow.end_run()
