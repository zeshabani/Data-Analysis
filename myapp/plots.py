import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from numpy.ma.core import floor
def my_histogram():
    data = pd.read_csv('/home/ZeinabShabani/mysite/sample_data.csv')
    data = data.dropna()

    data['date'] = pd.to_datetime(data['date'], format = '%m/%d/%Y')
    number_of_orders_per_day = data.groupby('date')['order_id'].count()
    number_of_orders_per_day = number_of_orders_per_day.reset_index()
    number_of_orders_per_day.columns = ['date', 'number_of_orders']
    number_of_orders_per_day['weekday'] = number_of_orders_per_day['date'].dt.weekday

    workdays = [5, 6, 0, 1, 2]
    holidays = [3, 4]
    workdays_data = number_of_orders_per_day[number_of_orders_per_day['weekday'].isin(workdays)]
    holidays_data = number_of_orders_per_day[number_of_orders_per_day['weekday'].isin(holidays)]

    plt.hist(workdays_data['number_of_orders'],alpha = 0.7, label = 'workdays')
    plt.hist(holidays_data['number_of_orders'], alpha = 0.7, label = 'holidays')
    plt.xlabel('Demands', fontsize = 16)
    plt.ylabel('Histogram of Demand', fontsize = 16)
    plt.legend()
    plt.savefig('/home/ZeinabShabani/mysite/templates/histogram.jpeg')



def scatter(k):
    data = pd.read_csv('/home/ZeinabShabani/mysite/sample_data.csv')
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], format = '%m/%d/%Y')

    frequency_df = data.groupby('user_id')['order_id'].count()
    frequency_df= frequency_df.reset_index()

    last_date = max(data['date'])
    data['interval'] = last_date - data['date']
    recency_df = data.groupby('user_id')['interval'].min()
    recency_df= recency_df.dt.days
    recency_df= recency_df.reset_index()

    montary_df = data.groupby('user_id')['total_purchase'].sum()
    montary_df = montary_df.reset_index()

    rfm_df = pd.merge(frequency_df, recency_df, on = 'user_id', how = 'inner')
    rfm_df = pd.merge(rfm_df, montary_df, on = 'user_id', how = 'inner')
    rfm_df.columns = ['user_id', 'frequncy', 'recency', 'montary']

    rfm_to_scale = rfm_df[['frequncy', 'recency', 'montary']]
    scaler = StandardScaler()
    rfm_to_scale = scaler.fit_transform(rfm_to_scale)
    rfm_to_scale = pd.DataFrame(rfm_to_scale)
    rfm_to_scale.columns = ['normalized_frequncy', 'normalized_recency', 'normalized_montary']

    kmeans = KMeans(n_clusters = k, max_iter = 100)
    kmeans.fit(rfm_to_scale)

    rfm_to_scale.loc[:, 'user_id'] = rfm_df['user_id']
    rfm_to_scale['cluster'] = kmeans.labels_
    rfm_to_scale = pd.merge(rfm_to_scale, rfm_df, on = 'user_id', how = 'inner')

    avg_freq = rfm_to_scale.groupby('cluster')['frequncy'].mean()
    avg_freq = avg_freq.reset_index()
    avg_freq.columns = ['cluster', 'Ave. F']

    avg_recency = rfm_to_scale.groupby('cluster')['recency'].mean()
    avg_recency = avg_recency.reset_index()
    avg_recency.columns = ['cluster', 'Ave. R']

    avg_mont = rfm_to_scale.groupby('cluster')['montary'].mean()
    avg_mont = avg_mont.reset_index()
    avg_mont.columns = ['cluster', 'Ave. M']

    result2 = pd.merge(avg_recency, avg_freq, on = 'cluster', how = 'inner')
    result2 = pd.merge(result2, avg_mont, on = 'cluster', how = 'inner')

    client_groups = ['خوشه ۱',
                     'خوشه ۲',
                     'خوشه ۳',
                     'خوشه ۴',
                     'خوشه ۵']

    client_groups = client_groups[:k]
    result2['cutomer grooup'] = client_groups
    result2 = result2[['cutomer grooup',
                       'Ave. R',
                       'Ave. F',
                       'Ave. M']]

    result2 = result2.values
    for res in result2:
        res[1] = floor((res[1]*100))/100
        res[2] = floor((res[2]*100))/100
        res[3] = floor((res[3]*100))/100

    f = plt.figure()
    # f.set_figwidth(7)
    # f.set_figheight(5)
    plt.scatter(rfm_to_scale['frequncy'],rfm_to_scale['recency'], c = rfm_to_scale['cluster'], alpha = 0.5, s = 10)
    plt.xlabel('freq', fontsize = 16)
    plt.ylabel('recency', fontsize = 16)
    plt.savefig('/home/ZeinabShabani/mysite/templates/scatter.jpeg')
    return result2