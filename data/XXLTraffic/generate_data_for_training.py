import os
import argparse
import numpy as np
import pandas as pd
import sys

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week, add_month_of_year):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    
    feature_list = [data]
    if add_time_of_day:
        print('add_time_of_day')
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    else:
        print('NO add_time_of_day')
    if add_day_of_week:
        print('add_day_of_week')
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)
    else:
        print('NO add_day_of_week')
    if add_month_of_year:
        print('add_month_of_year')
        moy = df.index.month
        moy_tiled = np.tile(moy, [1, num_nodes, 1]).transpose((2, 1, 0))
        month_of_year = moy_tiled / 12
        feature_list.append(month_of_year)
    else:
        print('NO add_month_of_year')

    data = np.concatenate(feature_list, axis=-1)
    
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def generate_train_val_test(args):
    if args.dataset == 'hour':
        df = pd.read_hdf('his_hour.h5')
    elif args.dataset == 'day':
        df = pd.read_hdf('his_day.h5')
    elif args.dataset == 'common':
        df = pd.read_hdf('his_common.h5')
    else:
        print('No dataset is', args.dataset)
        sys.exit()
    print('original data shape:', df.shape)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.dataset != 'day', args.dow, args.moy)
    print('final data shape:', data.shape, 'idx shape:', idx.shape)

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)   

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    # normalize
    x_train = data[:idx_val[0] - args.seq_length_x, :, 0] 
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    data[..., 0] = scaler.transform(data[..., 0])

    # save
    out_dir = args.dataset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(args.dataset, 'his.npz'), data=data, mean=scaler.mean, std=scaler.std)

    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hour', help='dataset name')
    parser.add_argument('--seq_length_x', type=int, default=96, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=336, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')
    parser.add_argument('--moy', type=int, default=1, help='month of year')
    
    args = parser.parse_args()
    generate_train_val_test(args)
