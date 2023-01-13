# Steps to run this mlflow package:

## Create env using conda.yaml or install mlflow and pyenv:

`conda create -f conda.yaml` \
`conda activate mlflow-env` \

or 

`pip install mlflow` \
`pip install pyenv`

## In order to train a model, run this command from the root directory. 
Alpha and L1_ratio values can be any number between 0 and 1:

`mlflow run . -P alpha=0.1 -P l1_ratio=0.9`

## Start the user interface locally (run from root dir)

`mlflow ui`

Now the UI is availbale at 127.0.0.1:5000

## Serve a model as a REST API locally
Get the model from the UI under artifacts in the experiments module

example: file:///Users/roaldbronstad/PycharmProjects/MlFlowTest/mlruns/0/3599a7c4d00647d3a3021af80a29e5c8/artifacts/model

`mlflow models serve -m **path_to_your_model** -p 1234`


## Invoke a prediction from the served model:

`curl http://127.0.0.1:1234/invocations -H "Content-Type:application/json" -d '{
"dataframe_split": {
"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH"	, "sulphates", "alcohol"],
"data":[[8.5, 0.6, 0, 1.5, 0.065, 17, 75, 0.86, 3.6, 0.6, 14]]
}
}'
`
