import pandas as pd

class MyDataLoader:
    def __init__(self, filename):
        self.filename = filename
        print(filename)

        # function to load csv data
        
    def load_csv_data(self):
        return pd.read_csv(self.filename)
    
    def load_json(self):
        return pd.read_json(self.filename)
    
       # function to load json data
    

        
    
      #  self.X = df.drop('views', axis=1)
     #   self.y = df['views']
    
   # def split_data(self):
   #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
   #         self.X, self.y, test_size=0.2, random_state=42