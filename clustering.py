
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns



df = pd.read_csv("172a7e65-6411-4db5-8406-2fc77d650f84_Data.csv") 
data = df



print(mp.style.available) 
mp.style.use('ggplot')
data = data.replace('..', np.nan)
data.isnull().sum().any()
data = data.dropna()




data.isnull().sum().any()









from sklearn.cluster import AffinityPropagation
X=data
# Import your dataset
X = data.drop(['Series Name', 'Series Code', 'Country Name', 'Country Code'], axis=1)
X




import warnings
warnings.filterwarnings("ignore")
X = np.array(X)
# Create an instance of Affinity Propagation
af = AffinityPropagation().fit(X)

# Extract the cluster labels
cluster_labels = af.labels_

# Extract the cluster centers
cluster_centers = af.cluster_centers_indices_

# Extract the number of clusters
n_clusters = len(cluster_centers)




from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=1835, centers=4,
                  random_state=0, cluster_std=0.60)

# Extract the cluster labels
cluster_labels = af.labels_

# Plot the data points colored by their cluster label
mp.scatter(X[:, 0], X[:, 1], c=cluster_labels)
mp.show()




# Pie chart, where the slices will be ordered and plotted counter-clockwise:
year = ['1990 [YR1990]','2000 [YR2000]','2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]','2021 [YR2021]']
GDP = [14.4866718355357,12.2862979661957,11.3452427502371,11.2336594326489,11.1023526258084,11.0315763115403,11.0209556144528,10.9790172199535,10.935191266697,10.8924664317099,10.8321540748374,10.8125390809483]
fig1, ax1 = mp.subplots()
ax1.pie(GDP, labels=year, autopct='%1.1f%%', shadow=True, startangle=90)
myexplode = [0.2, 0, 0, 0]
ax1.axis('equal')  
mp.show()




fig = mp.figure(figsize = (20, 5))
 
# creating the bar plot
mp.bar(year,GDP , color ='maroon',width = 0.4)
 
mp.xlabel("10 years of Population")
mp.ylabel("Population in urban agglomerations")
mp.title("Population in urban growth")
mp.show()




from scipy.optimize import curve_fit

# Define the model function
def exp_growth(x, a, b):
    return a * np.exp(b * x)

# Extract the x and y values from the data set
x_data = np.array(['1990 [YR1990]','2000 [YR2000]','2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]','2021 [YR2021]'])
y_data = np.array([14.4866718355357,12.2862979661957,11.3452427502371,11.2336594326489,11.1023526258084,11.0315763115403,11.0209556144528,10.9790172199535,10.935191266697,10.8924664317099,10.8321540748374,10.8125390809483])
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
mp.xticks(rotation=45, horizontalalignment="center")
mp.plot(x_data, y_data, 'o')
mp.show()




# column_data = data["2015 [YR2015]"]

# # Use the numpy function `max()` to find the maximum value in the column
# max_value = np.max(column_data)

# # Use the numpy function `min()` to find the minimum value in the column
# min_value = np.min(column_data)
# average_value = np.mean(column_data)
# print("Max Value: ", max_value)
# print("Min Value: ", min_value)




# normalized_value = (11.2336594326489 - 9917199) / (9917199 - 0.0112391121219639) * (100 - 1) + 1




# average = np.mean(y_data)



# y_data_divided_by_average = 11.2336594326489/average

