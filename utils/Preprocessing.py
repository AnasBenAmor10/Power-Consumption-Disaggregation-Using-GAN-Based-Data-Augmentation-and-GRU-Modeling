import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import glob
import os


def read_label(dataset_path,  house_list):
    label = {}
    for i in house_list:
        hi = os.path.join(dataset_path,f"house_{i}/labels.dat")
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label


def read_merge_data(dataset_path, house, labels):
    path = os.path.join(dataset_path, f"house_{house}/")
    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep=' ', names=['unix_time', labels[house][1]],
                       dtype={'unix_time': 'int64', labels[house][1]: 'float64'})

    num_apps = len(glob.glob(path + 'channel*.dat'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep=' ', names=['unix_time', labels[house][i]],
                             dtype={'unix_time': 'int64', labels[house][i]: 'float64'})
        df = pd.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)
    return df

def create_house_dataframe(dataset_path , house_list):
    labels = read_label(dataset_path, house_list)
    df = {}
    for i in house_list:
        df[i] = read_merge_data(dataset_path , i, labels)
        print("House {} finish:".format(i))
        print(df[i].head())

    return df

def date(house_list, df):
    dates = {}
    for i in house_list:
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))
        print('House {0} data contain {1} days from {2} to {3}.'.format(i, len(dates[i]), dates[i][0], dates[i][-1]))
        print(dates[i], '\n')

    return dates


