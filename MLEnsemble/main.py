from MyDataLoader import MyDataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns



def main():
    print("Hello, Welcome to project 2, SKLearn")

# call loader class
    load_combined_data_df = MyDataLoader("combined_data.csv")
    combined_data = load_combined_data_df.load_csv_data()
    

    combined_data = combined_data.set_index('video_id')

    # Check for null values in combined_data
    null_counts = combined_data.isnull().sum()

    # Print the number of null values in each column
    print(null_counts)

    # breakpoint()

    # This column is all nulls for some reason, so will drop itc
    combined_data.drop('num_tags', axis=1, inplace=True)

    print(combined_data)


    # Add your code here

    label = combined_data['label']
    features = combined_data.drop(['label'],axis=1)

    # print label and feature data
    print('check_x_y_split',([features.shape, label.describe()]))



    # split the features and label data into training and testing sets using train_test_split function
    #  from sklearn's model_selection module. It is also scaling the feature
    #  data using StandardScaler from sklearn's preprocessing module
    # 0.2 means that 20 % will be used for testing and 80 % training

   # x_train, x_test, y_train, y_test = train_test_split(features.values, label.values, test_size=0.2, random_state=0)
   # from sklearn.preprocessing import StandardScaler
   # sc = StandardScaler()
   # x_train = sc.fit_transform(x_train)
   # x_test = sc.transform(x_test)


    # 
    # print('check_data_split',[x_train.shape,x_test.shape,y_train.shape,y_test.shape])



    # Create linear regression object
    # lin_R = LinearRegression()

    # Train the model using the training sets
   # lin_R.fit(x_train, y_train)

    # Make predictions

   # y_predict_test = lin_R.predict(x_test)
   # y_predict_train= lin_R.predict(x_train)

    # The coefficients
    #print('Coefficients: \n', lin_R.coef_)

   # print('r Squared: %.2f'
    #      % lin_R.score(x_test, y_test))

   # print('mse_value of Test=',mean_squared_error(y_test, y_predict_test))
   # print('mse_value of Train=',mean_squared_error(y_train, y_predict_train))
   # mse_test= mean_squared_error(y_test, y_predict_test)

   # print('check_lr', (np.sqrt(mean_squared_error(y_test, y_predict_test))))

   # pca = PCA()
   # sc = StandardScaler()
  #  x_train_std=pca.fit_transform(x_train)
  #  np.set_printoptions(suppress=True)
  #  var_vs_pca = np.cumsum(pca.explained_variance_ratio_)
  #  print([(i,x) for i, x in enumerate(var_vs_pca)])
  #  # plotting the explained_variance_ratio against the number of components
  #  plt.plot(var_vs_pca)
   # # plt.show()

    # Add your code here
   # pca = PCA(n_components=34)
   # x_train_Trans=pca.fit_transform(x_train)
   # x_test_Trans=pca.transform(x_test)


    # print 
  #  print('check_pca', (x_train_Trans[:50,:]))


    # Random Forest - too long to process
    # 
    # Only tune the max depth of the trees in the RF hyperparameter.
    #grid = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'n_estimators':[140],'max_depth':[25,30,35,40,45]},cv=5)
    # grid.fit(x_train_Trans, y_train)
    #grid.best_params_
    # depth = [40]
    # nEstimator = [140]

   # grid = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'max_depth':[25,30,35,40,45]}, cv=5)
    #grid.fit(x_train_Trans, y_train)
    # grid.best_params_
    # print(grid.best_params_)

    # Split the data into training and test sets for a GradientBoost model
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)


    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train the model
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=0)
    model.fit(x_train, y_train)

    # Evaluate the model
    y_pred_train = model.predict(x_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    print(f"Train MSE: {train_mse:.2f}")

    y_pred_test = model.predict(x_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print(f"Test MSE: {test_mse:.2f}")

    # Visualisation 
    # Get feature importances
    feature_importances = model.feature_importances_

    # Plot feature importances using a bar chart
    sns.barplot(x=feature_importances, y=features.columns)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()

# Make predictions with the gradient boost model
    y_pred = model.predict(x_test)

# Create a pie chart of the predicted values
    plt.pie([len(y_pred[y_pred <= 0.5]), len(y_pred[y_pred > 0.5])],
           labels=['Negative', 'Positive'],
           autopct='%1.1f%%',
           colors=['red', 'green'])
    plt.title('Predicted Sentiment')
    plt.show()


    



if __name__ == "__main__":
    main()