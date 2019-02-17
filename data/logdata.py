import pandas as pd
import numpy as np
class Logistic_reg_data:
    def __init__(self):
        self.read()
    def logisticdata(self,dataframe):                #our X and y (labels and data)
        self.data=dataframe.iloc[:,1:3]
        self.Y=dataframe["label"]

    def read(self):
        df = pd.read_csv("data/data.csv")
        self.logisticdata(df)