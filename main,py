
from MyDataLoader import MyDataLoader
from DataCombiner import DataCombiner
from SaveDataFrame import SaveDataFrame
from SplitData import SplitData
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Hello, Welcome to project 2 of Data Analytics!")
    

    # call loader class
    load_CAvideo_df = MyDataLoader("CAvideos.csv")
    data = load_CAvideo_df.load_csv_data()
    print(data)

    # next load json 
    CA_category_id_df = MyDataLoader("CA_category_id.json")
    data2 = CA_category_id_df.load_json()
    print(data2)
    
    # combine into one dataframe
    getList_csv = DataCombiner("*.csv")
    files = getList_csv.combine_csv()
    print(files) 



    # Merge all dataframes into a single dataframe with a 'country' column
    all_dataframes = []
    for csv_file in files:
        df = pd.read_csv(csv_file, index_col=0)
        country_name = os.path.basename(csv_file).replace('videos.csv', '')
        df['country'] = country_name
        all_dataframes.append(df)

    combined_data = pd.concat(all_dataframes)

    # Strip '+' and '-' characters from country names
    combined_data['country'] = combined_data['country'].map(lambda x: x.strip('+-'))


    # add the json data to new column of the combined_data dataframe

    combined_data['category_id'] = combined_data['category_id'].astype(str)
    js_files = [i for i in glob.glob('*.json')]
    sorted(js_files)

    id_to_category = {}
    for x in js_files:
          js = pd.read_json(x)
          for category in js ["items"]:
            id_to_category[category["id"]] = category["snippet"]["title"]
    combined_data["category"] = combined_data["category_id"].map(id_to_category)
# Add code here

    # combined_data.insert(4, 'category'))

    # combined_data.head(10)
    combined_data.info()

    # Change time format 

    # Convert columns to datetime
    combined_data['trending_date'] = pd.to_datetime(combined_data['trending_date'], format='%y.%d.%m')
    combined_data['publish_time'] = pd.to_datetime(combined_data['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    # Remove rows with missing values
    combined_data.dropna(inplace=True)

    # Print info about the dataframe
    combined_data.info()

    # conert time format and drop nulls
    combined_data.info()
    combined_data['trending_date'] = pd.to_datetime(combined_data["trending_date"],format ="%y.%d.%m")
    combined_data['publish_time'] = pd.to_datetime(combined_data["publish_time"],format = "%Y-%m-%dT%H:%M:%S.%fZ")

# drop nulls 
    combined_data = combined_data.dropna()
    combined_data.info()

# print confirmation 
    print('validate_na',(combined_data.shape))

# visualisation of data and feature identification
# the describe() method calculates the count, mean, standard deviation, minimum, 
# 25th percentile,  median (50th percentile), 75th percentile, and maximum value. 
# uotput is a dataframe
    combined_data.describe()


    if len(combined_data.select_dtypes(include=[np.number]).columns) == 0:
        print("Error: no numerical columns found in combined_data DataFrame.")
    else:
        maxs = combined_data.describe().iloc[7].values.tolist()
        mins = combined_data.describe().iloc[3].values.tolist()
        stds = combined_data.describe().iloc[2].values.tolist()
        means = combined_data.describe().iloc[1].values.tolist()

        # print here
    print('check_min_max_mean_std',([maxs, mins, stds, means]))

    # data has a wide range of values, add columns that are calculated from the natural log of the data + 1
    # scaling can help to smooth out distortions caused by outliers

    combined_data['likes_log'] = np.log(1 + combined_data['likes']) 
    combined_data['views_log'] = np.log(1 + combined_data['views']) 
    combined_data['dislikes_log'] = np.log(1 + combined_data['dislikes']) 
    combined_data['comment_log'] = np.log(1 + combined_data['comment_count']) 

    # Print results
    print('check_feature_rescaling',([np.mean(combined_data['likes_log']),np.mean(combined_data['views_log']),np.mean(combined_data['dislikes_log']),
                                         np.mean(combined_data['comment_log'])]))
    

    # Add your code here for plotting the distribution

    # log_df=combined_data[['likes_log','views_log','dislikes_log','comment_log']]
    # log_df.plot.kde(figsize=(20,10))

   # plot data
  

    log_df = combined_data[['likes_log', 'views_log', 'dislikes_log', 'comment_log']]
    ax = log_df.plot.kde(figsize=(20,10))
    ax.set_xlabel('Logarithm of Count')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimate of Log-Transformed Counts')
    ax.legend()
    plt.show()


# Number of videos per category
    by_category = combined_data.groupby(['category']).size().sort_values(ascending=False)
    ax = by_category.plot(kind='bar', figsize=(10, 6))
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Videos')
    ax.set_title('Total Videos by Category')
    plt.show()

# views against category boxplot 
    by_category = combined_data.groupby(['category'])['views_log'].apply(list)

    plt.figure(figsize=(12,8))
    plt.boxplot(by_category.values)
    plt.xticks(range(1, len(by_category)+1), by_category.index, rotation=90)
    plt.xlabel('Category')
    plt.ylabel('Log-transformed Views')
    plt.title('Distribution of Log-transformed Views by Category')
    plt.show()

# create a boxplot of dislikes_log distribution against categories
    plt.figure(figsize=(10,8))
    combined_data.boxplot(column=['dislikes_log'], by=['category'], vert=False)
    plt.title('Boxplot of Dislikes Log Distribution by Category')
    plt.xlabel('Dislikes Log')
    plt.ylabel('Category')
    plt.show()

# drop duplicates and group by title and country
    df = combined_data.drop_duplicates()
    df1 = df.groupby(['title', 'country']).size().reset_index(name='count')

# group by country and calculate mean trending days
    trending = df1.groupby('country')['count'].mean().to_frame().reset_index().rename(columns={"country": "Country", "count": "Mean Trending Days"})

# create bar plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set(font_scale=1.5)
    sns.barplot(x="Country", y="Mean Trending Days", data=trending, ax=ax)
    plt.title('Mean Number of Video Trending Days in Each Country', color='Blue')
    plt.show()

# feature engineering tags, had to fix initialisation error of the numpy array num_tags

    num_tags = pd.Series(dtype=int)
    xdf=combined_data.reset_index(drop=True) # Reset index to avoid index inconsistencies
    
    for i in range(len(xdf)):
      if xdf.at[i,'tags']=='[none]': # some videos has no tags but instead [none], so we are going to consider it as Zero tags.
        count=0
    else:
        count=(xdf.at[i,'tags']).count("|") + 1 
    num_tags.at[i] = count
    
    # add a new column 'num_tags' to the dataframe
    combined_data['num_tags']=num_tags 

    combined_data

# Add the lengths of the features and titles columns and add as features 
    combined_data["desc_len"]=combined_data["description"].apply(lambda x: len(x))
# Add title length 
    combined_data["len_title"]=combined_data["title"].apply(lambda x: len(x))

# Print summary statistics for num_tags, desc_len, and len_title columns using a loop instead

    columns = ['num_tags', 'desc_len', 'len_title']
    for col in columns:
      print(f"{col}:\n{combined_data[col].describe()}\n")

# --- split date time week into separate columns

    date_data=combined_data['publish_time']

    # Add your code here
    combined_data['publish_time'] =date_data.apply(lambda x: pd.to_datetime(x).time())
    combined_data['publish_date'] =date_data.apply(lambda x: pd.to_datetime(x).date())

#day on which video was published
    combined_data['publish_weekday']=date_data.apply(lambda x: x.dayofweek)+1

    import random
    random_index = random.randint(0,combined_data.shape[0]-1)

    cols = ['publish_time', 'publish_date', 'publish_weekday']
    random_index = random.randint(0, combined_data.shape[0] - 1)

    #print random selection 

    for col in cols:
      print(f'{col}: {combined_data[col].iloc[random_index]}')
    
    weekday_counts = sorted(list(combined_data['publish_weekday'].value_counts()))
    print(f'Publish weekday counts: {weekday_counts}')

# Add 

#Creating dataframe after deleting videos which stay trending for more than one day according to the Video ID
    dfx=combined_data.reset_index(level=0)[['video_id','publish_weekday']].drop_duplicates(subset = ['video_id'], keep = 'last')

##Mapping the day number : day name
    dayOfWeek={1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday', 7:'Sunday'}
    dfx['publish_weekday'] = dfx['publish_weekday'].map(dayOfWeek)

##Calculating and plotting
    videos_weekday = dfx['publish_weekday'].value_counts().to_frame().reset_index().rename(columns={"index": "Week_Days", "publish_weekday": "No_of_videos"})
    fig, ax = plt.subplots(figsize=(15, 10)),sns.set(font_scale=1.5)
    sns.barplot(x="Week_Days", y="No_of_videos", data=videos_weekday,ax=ax),plt.title('Number of puplished videos per week days ',color='Green')


    # remove non-numerical columns
    combined_data.drop(['trending_date', 'title', 'channel_title', 'category_id',
                    'publish_time', 'tags', 'views', 'likes', 'dislikes', 'comment_count',
                    'thumbnail_link', 'description', 'publish_date'], axis = 1,inplace = True)


    


# convert categorical data to numerical codes unsing One Hot Vector encoding 

    combined_data.publish_weekday = combined_data.publish_weekday.astype('category')
    combined_data.country = combined_data.country.astype('category')
    combined_data.category = combined_data.category.astype('category')
    combined_data= pd.get_dummies(combined_data)
 
# check shape of the dataframe after encoding 

    print('check_final_df',(combined_data.shape))
    print(combined_data)

    combined_data_sec_2 = combined_data.copy()
    combined_data_sec_2.rename(columns = {'views_log':'label'}, inplace = True) 


# create instance of SaveDataFrame
    df_saver = SaveDataFrame(combined_data_sec_2)

   
# save dataframe to csv - this done once, was successful then ive commented it out 
    df_saver.save_to_csv('combined_data.csv')


# split data 
     
  #  combined_data=pd.read_csv('combined_data.csv').set_index('video_id')
  #  label = combined_data['label']
  #  features = combined_data.drop(['label'],axis=1)

    # print cell
  #  print('check_x_y_split',([features.shape, label.describe()]))

    #split_data = SplitData(features.values, label.values)
   # x_train, x_test, y_train, y_test = split_data.split()

# print cell.
  #  print('check_data_split',[x_train.shape,x_test.shape,y_train.shape,y_test.shape])
 













if __name__ == "__main__":
    main()







#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error

#import MyDataLoader
# import DataVisualiser






# data_loader.split_data()

# call visualiser class

# visualizer = DataVisualiser(df)
# visualizer.display_head()
# visualizer.display_summary()
# visualizer.plot_histogram("column_name")
# visualizer.plot_scatter("x_column_name", "y_column_name")