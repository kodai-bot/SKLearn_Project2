import pandas as pd
import matplotlib.pyplot as plt

class DataVisualiser:
    def __init__(self, data):
        self.data = data

    def display_head(self, n=5):
        print(self.data.head(n))

    def display_summary(self):
        print(self.data.describe())

    def plot_histogram(self, column):
        plt.hist(self.data[column])
        plt.title(column + " distribution")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.show()

    def plot_scatter(self, x_column, y_column):
        plt.scatter(self.data[x_column], self.data[y_column])
        plt.title(y_column + " vs " + x_column)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
