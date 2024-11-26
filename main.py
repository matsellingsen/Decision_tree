import pandas as pd
import pathlib
import sys
from pathlib import Path
import os 
import decision_tree as dt

def readData(file):
    path = str(pathlib.Path("main.py").parent.resolve()) #<-- string in Path() should be the name of this python-file.
    return pd.read_csv(path + "\\" + file)

def preprocess(data):
    data = data.drop(columns=["Cabin", "PassengerId", "Name", "Age", "Fare", "Ticket"]) #<-- "Cabin column has over 90% missing values so it tells us nothing. Most of the rest are unique values so these will also tell us nothing."

    #Creating buckets for columns with a lot of varying values like "Age" and "Fare". This should improve 
    # TODO^

    return data.dropna() #Dropping remaining rows with missing values


def main():

    train = readData("train.csv")
    train = preprocess(train)
    test = readData("test.csv")
    test = preprocess(test)
    
    split_index = int(len(train) * 0.8)
    tree_model = dt.Decision_tree(train.iloc[:split_index], train.iloc[split_index:], "Survived")

    tree_model.train()
    
    tree_model.test() #<-- currently achieving accuracy of 82%. This should be improved when further preprocessing (above) is done.

    #preds = tree_model.predict(test)
  

main()

    

