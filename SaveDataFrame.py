
class SaveDataFrame:
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def save_to_csv(self, filename):
        self.dataframe.to_csv(filename)

    # def save_to_json(self, filename):
