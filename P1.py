from pandas import *
from ggplot import *
from numbers import Number
import numpy as np
import datetime as dt
import statsmodels.api as sm
import scipy
import operator
    
def plot_weather_data(turnstile_weather):
    '''
    You are passed in a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.  
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.  

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station
     * Which stations have more exits or entries at different times of day

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/
     
    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
     
    To see all the columns and data points included in the turnstile_weather 
    dataframe. 
     
    However, due to the limitation of our Amazon EC2 server, we are giving you about 1/3
    of the actual data in the turnstile_weather dataframe
    '''

    #print list(turnstile_weather.columns.values)
    
    #turnstile_weather['weekday'] = turnstile_weather['DATEn'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').weekday())
#    print turnstile_weather.describe()
     
    ### Frequency of rider counts in hourly block ranges 
    #plot = ggplot(turnstile_weather, aes('ENTRIESn_hourly')) + geom_histogram(binwidth=100) + xlim(0, 5000)        

#    turnstile_weather = turnstile_weather[['ENTRIESn_hourly', 'rain']]
    with_rain = turnstile_weather[turnstile_weather['rain'] == 1]
    without_rain = turnstile_weather[turnstile_weather['rain'] == 0]
    
    # this fixes the KeyError that shows up
    with_rain.index = range(with_rain.shape[0])
    
    df = DataFrame({
        "with_rain": with_rain['ENTRIESn_hourly'],
        "without_rain": without_rain['ENTRIESn_hourly'],
    })
    df = melt(df)
    
#    print with_rain.describe()
#    print "WITH RAIN\n\n{}\n\n".format(with_rain['ENTRIESn_hourly'].describe())
#    print "WITHOUT RAIN\n\n{}\n\n".format(without_rain['ENTRIESn_hourly'].describe())
    
    plot = ggplot(aes(x='value', color='variable', fill='variable'), data=df) + \
        geom_histogram(alpha=0.6, binwidth=25) + xlim(0,500) + \
        labs(title='NYC Subway Ridership Frequency\n(May 2011)', \
            x='Number of riders\n(blue=with rain, red=without rain)', \
            y='Frequency') + \
        scale_color_manual(values=['blue', 'red'])            
    print plot
#    ggsave('plot.png', plot)
    
    ### Frequency of rider counts in hourly block ranges sub-divided by rain vs/ no rain
    #plot = ggplot(turnstile_weather, aes('rain', 'ENTRIESn_hourly')) + geom_point()

    ### Frequency of rainy days
    #print ggplot(turnstile_weather, aes('rain')) + geom_histogram(binwidth=0.5)

    ### Ridership by time of day
    turnstile_weather = turnstile_weather.groupby('Hour', as_index=False)['ENTRIESn_hourly'].aggregate(sum)
#    print turnstile_weather.describe()
    plot = ggplot(turnstile_weather, aes('Hour', 'ENTRIESn_hourly')) + \
        geom_bar(stat='bar') + xlim(-0.5,23.5) + \
        labs(title='NYC Subway Ridership by Time-of-Day\n(May 2011)', \
            x='Hour (24-hour format)', y='Number of riders (tens of millions)') + \
        scale_x_continuous(breaks=range(24))
    print plot        

def entries_histogram(turnstile_weather):
    '''
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's 
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.
    
    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()
    
    Your histograph may look similar to bar graph in the instructor notes below.
    
    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
    
    You can see the information contained within the turnstile weather data here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''
    
    plt.figure()
    turnstile_weather[turnstile_weather['rain'] == 1]['ENTRIESn_hourly'].hist() # your code here to plot a historgram for hourly entries when it is raining
    turnstile_weather[turnstile_weather['rain'] == 0]['ENTRIESn_hourly'].hist() # your code here to plot a historgram for hourly entries when it is not raining
    return plt

def mann_whitney_plus_means(turnstile_weather):
    '''
    This function will consume the turnstile_weather dataframe containing
    our final turnstile weather data. 
    
    You will want to take the means and run the Mann Whitney U-test on the 
    ENTRIESn_hourly column in the turnstile_weather dataframe.
    
    This function should return:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain
    
    You should feel free to use scipy's Mann-Whitney implementation, and you 
    might also find it useful to use numpy's mean function.
    
    Here are the functions' documentation:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    
    You can look at the final turnstile weather data at the link below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''
    
    with_rain = turnstile_weather[turnstile_weather['rain'] == 1]['ENTRIESn_hourly']
    without_rain = turnstile_weather[turnstile_weather['rain'] == 0]['ENTRIESn_hourly']
    with_rain_mean = np.mean(with_rain)
    without_rain_mean = np.mean(without_rain)
    U, p = scipy.stats.mannwhitneyu(with_rain, without_rain)
    
    # p is one-sided, so it is necessary to double it
    return with_rain_mean, without_rain_mean, U, p * 2
    
def normalize_features(array):
   """
   Normalize the features in the data set.
   """
   mu = array.mean()
   sigma = array.std()
   array_normalized = (array-mu)/sigma

   return array_normalized, mu, sigma
    
def predict_results(weather_turnstile):
    values = weather_turnstile['ENTRIESn_hourly']
    
    # Add some custom features
    turnstile_weather['weekday'] = turnstile_weather['DATEn'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').weekday())
    turnstile_weather['yearday'] = weather_turnstile['DATEn'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday)

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(weather_turnstile['UNIT'], prefix='unit')
    
    # Show the features sorted by impact on R^2 outcome
    individual_features = {}
    df = turnstile_weather
#    df = df.join(dummy_units)
    for col in df:
        if df[col].dtype in [np.float64, np.int64]:
            features = df[[col]]
            features = sm.add_constant(features)
            try:
                model = sm.OLS(values, features, hasconst=True)
                results = model.fit()
                predictions = np.dot(features, results.params)
                r2 = compute_r_squared(values, predictions)
                individual_features[col] = r2
            except ValueError:
                pass

    sorted_features = sorted(individual_features.items(), key=operator.itemgetter(1), reverse=True)
    print "Features sorted by R^2 outcome:"
    for t in sorted_features:
        print "{}: {}".format(t[0], t[1])
    print

#    features = weather_turnstile[['Hour', 'weekday', 'mintempi', 'meanwindspdi', 'yearday', 'fog', 'precipi', 'rain']]
    features = weather_turnstile[['Hour', 'weekday', 'mintempi', 'meanwindspdi', 'minpressurei', 'mindewpti', 'yearday', 'fog', 'precipi']]
#    features = weather_turnstile[['rain', 'precipi']]
  
#    features = dummy_units            
    features = features.join(dummy_units)
    
    features, mu, sigma = normalize_features(features)
    features = sm.add_constant(features)
    
    model = sm.OLS(values, features, hasconst=True)
    results = model.fit()
#    print results.summary()
    print results.params

    predictions = np.dot(features, results.params)
    return predictions    
    
def compute_r_squared(data, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.
    
    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''

    mean = np.mean(data)
    r_squared = 1 - np.square(data - predictions).sum() / np.square(data - mean).sum()
    return r_squared    

def plot_residuals(turnstile_weather, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    
    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist()
    return plt
    
def queries(weather_turnstile):
    # Testing whether rain is highly correlated with precipitation
    total = weather_turnstile['rain'].count()
    df = weather_turnstile
    df = df[(df.rain == 0) & (df.precipi <= 0)]
    norain_noprecipi = df['rain'].count()
    df = weather_turnstile
    df = df[(df.rain == 1) & (df.precipi > 0)]
    rain_precipi = df['rain'].count()
    print total, norain_noprecipi, rain_precipi, (norain_noprecipi + rain_precipi)

if __name__ == "__main__":
    turnstile_weather = pandas.read_csv('turnstile_data_master_with_weather.csv')
    turnstile_weather.is_copy = False
    #turnstile_weather['datetime'] = turnstile_weather['DATEn'] + ' ' + turnstile_weather['TIMEn']

#    queries(turnstile_weather)

    print "Mann-Whitney U: (mean of rain, mean of no rain, U, p)"
    print mann_whitney_plus_means(turnstile_weather), "\n"

    predictions = predict_results(turnstile_weather)
    print "R^2: ", compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predictions)
    print "\n\nResiduals Graph:\n", plot_residuals(turnstile_weather, predictions)        

    plot_weather_data(turnstile_weather)
