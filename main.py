from fastapi import FastAPI
import numpy as np

#for machine learning
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor

import pickle

from pydantic import BaseModel

from fastapi.responses import JSONResponse
#from iris_classifier_object import Iris_Classifier



class Iris_Classifier:
    def __init__(self, model_path:str):
        self.model = self.get_model(model_path)
        self.iris_species = {
            0 : 'Setosa',
            1 : 'Versicolour',
            2 : 'Virginica'
        }
    def get_model(self, model_path:str) -> MLPClassifier:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    def make_prediction(self, features:dict) -> str:
        features = np.array(list(features.values()))
        pred = self.model.predict(features.reshape(1, -1))[0]
        species_pred = self.iris_species[pred]
        return species_pred
    

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mlp_classifier = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)


predictions = mlp_classifier.predict(X_test)

acc = 100*sum([1 if y_p == y_t else 0 for y_p, y_t in zip(predictions, y_test)])/len(y_test)
with open('iris_classifier.pk1', 'wb') as f:
    pickle.dump(mlp_classifier, f)

app = FastAPI(
    title = 'Iris Classifier API',
    version = 1.0,
    description='Simple API to predict class of iris plant'
    )

classifier = Iris_Classifier('iris_classifier.pk1')

class Iris(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

@app.post("/",tags = ["iris_classifier"])
def get_prediction(features:Iris):
    species_pred = classifier.make_prediction(features.dict())
    return JSONResponse({"species":species_pred})
@app.get("/")
def index():
    return {"hello" : "world"}


