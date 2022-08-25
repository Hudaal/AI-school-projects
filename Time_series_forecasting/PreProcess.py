import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import make_interp_spline
import math
from RNN_model import RNN_model


class PreProcess:
    def __init__(self, train_file, test_file):
        self.train_df = pd.read_csv(train_file)
        self.train_df["flow"] = -self.train_df["flow"]
        self.test_df = pd.read_csv(test_file)
        self.test_df["flow"] = -self.test_df["flow"]
        self.target_train_df = None
        self.target_test_df = None
        self.minmax_scaler = MinMaxScaler()
        self.minmax_scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.standard_scaler = StandardScaler()
        self.target_standard_scaler = StandardScaler()
        self.feachers_to_scale = ['hydro', 'micro', 'thermal', 'wind', 'river', 'total']
        self.feachers_to_scale_standard = ['sys_reg', 'flow']
        self.feachers_to_scale_standard_target = ['prev_y', 'prev_hour', 'avr', 'prev_day']
        self.train_y = self.train_df['y']
        self.test_y = self.test_df['y']

    def update_upper_lower(self, upper_value, lower_value, df):
        """ updated bigger than upper or smaller than lower limits with the upper and lower values """
        df.y[df.y > upper_value] = upper_value
        df.y[df.y < -lower_value] = -lower_value
        return df

    def add_day_hour_year(self, df):
        """ Add some new helping features """
        df['start_day'] = [d.split()[0] for d in df['start_time']]
        df['start_hour'] = [d.split()[1] for d in df['start_time']]
        df['year'] = [d.split()[0].split('-')[0] for d in df['start_time']]
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['time_of_day'] = df.start_time.dt.hour
        df['time_of_month'] = df.start_time.dt.day
        df['time_of_year'] = df.start_time.dt.month
        df['morning'] = np.logical_and(df['time_of_day'] >= 5, df['time_of_day'] < 12).astype(np.float32)
        df['afternoon'] = np.logical_and(df['time_of_day'] >= 12, df['time_of_day'] < 18).astype(np.float32)
        df['evening'] = np.logical_and(df['time_of_day'] >= 18, df['time_of_day'] <= 23).astype(np.float32)
        df['weekend'] = df.start_time.dt.day_name().isin(['Saturday', 'Sunday']).astype(np.float32)
        df['summer'] = np.logical_and(df['time_of_year'] >= 6, df['time_of_year'] <= 8).astype(np.float32)
        df['autumn'] = np.logical_and(df['time_of_year'] >= 9, df['time_of_year'] < 11).astype(np.float32)
        df['winter'] = np.logical_and(df['time_of_year'] >= 11, df['time_of_year'] <= 2).astype(np.float32)
        df['spring'] = np.logical_and(df['time_of_year'] >= 3, df['time_of_year'] <= 5).astype(np.float32)
        df = self.add_structural_imbalance(df)
        return df

    def add_structural_imbalance(self, df):
        """ Add structural_imbalance for both main and altered tasks """
        sum_total_flow = df['total'] + df['flow']
        sum_total_flow_splined = sum_total_flow.ewm(span=5000).mean()
        df['structural_imbalance'] = sum_total_flow_splined - sum_total_flow
        return df
    
    def altered_forecasting(self, df):
        """ Update the target if altered forecasting """
        final = df['y'] - df['structural_imbalance']
        df['y_old'] = df['y']
        df['y'] = final
        return df

    def drop_col(self, df, col_list):
        for col in col_list:
            df = df.drop(col, 1)
        return df

    def add_previous_target(self, df, rows_shifted, target_name, prev_target_name):
        """ add lagged features from previous targets with row_shifted rows """
        df[prev_target_name] = df[target_name].shift(rows_shifted)
        df.loc[:rows_shifted, prev_target_name] = df.loc[rows_shifted, prev_target_name]
        return df

    def add_avarage_targets(self, df, target, length):
        """ Add the avrage of last length rows as avr column """
        df['avr'] = 0
        df['avr'] = self.find_avarage(target, length)
        return df

    def find_avarage(self, target_arr, length):
        return target_arr.rolling(window=length, min_periods=0).mean()

    def one_step_update(self, df, one_step_size):
        """ Choose if one step is 5 or 15 minutes and update the dataframe """
        if one_step_size == 5:
            return df
        elif one_step_size == 15:
            return df.iloc[::3, :]
        else:
            print('This step size is not implemented')
            return False

    def make_targets(self, df, shift_steps, target_name):
        """ Make a target dataset with next values for each sequence """
        return df[target_name].shift(-shift_steps)

    def train_test_split(self, x, y, train_percent=0.9):
        """ Split the training dataset to train and test datasets """
        num_train = int(train_percent * len(x))
        x_train = np.array(x[:num_train])
        y_train = np.array(y[:num_train])
        x_test = np.array(x[num_train:])
        y_test = np.array(y[num_train:])
        return x_train, y_train, x_test, y_test

    def scale_data(self, df, df_targets, shift_step, train=True):
        if train:
            minmax_scaler = self.minmax_scaler.fit_transform
            standard_scaler = self.standard_scaler.fit_transform
            target_scaler = self.target_standard_scaler.fit_transform
            minmax_target_scaler = self.minmax_scaler_target.fit_transform
        else:
            minmax_scaler = self.minmax_scaler.transform
            standard_scaler = self.standard_scaler.transform
            target_scaler = self.target_standard_scaler.transform
            minmax_target_scaler = self.minmax_scaler_target.transform
        df[self.feachers_to_scale] = minmax_scaler(df[self.feachers_to_scale])
        df[self.feachers_to_scale_standard] = standard_scaler(df[self.feachers_to_scale_standard])
        df_targets = minmax_target_scaler(df_targets)
        # df_targets = target_scaler(df_targets)
        for col in self.feachers_to_scale_standard_target:
            df[col] = self.minmax_scaler_target.transform(np.array(df[col]).reshape(-1, 1))
            # df[col] = self.target_standard_scaler.transform(np.array(df[col]).reshape(-1, 1))
        x_scaled = np.array(df.values[0:-shift_step])
        y_scaled = np.array(df_targets[0:-shift_step])
        return x_scaled, y_scaled

    def batch_generator(self, x, y, batch_size, sequence_length, feature_count, num_train):
        """ returns a generator of a sequence as x and y with random start each time """
        while True:
            x_shape = (batch_size, sequence_length, feature_count)
            x_batch = np.zeros(shape=x_shape)
            y_shape = (batch_size, 1, 1)
            y_batch = np.zeros(shape=y_shape)
            for i in range(batch_size):
                idx = np.random.randint(num_train - sequence_length)
                x_batch[i] = x[idx:idx + sequence_length]
                y_batch[i] = y[idx + sequence_length]
            yield x_batch, y_batch

    def batch_generator_separated(self, x, y, sequence_length):
        """ Make the whole data as sequences of x and y with chosen length """
        x_batch = []
        y_batch = []
        for idx in range(0, len(x) - sequence_length):
            x_batch.append(x[idx:idx + sequence_length])
            y_batch.append(y[idx + sequence_length])
        return np.array(x_batch), np.array(y_batch)

    def prepare_dataset(self, df, shift_steps, altered_forecasting=False):
        """ Preprocessing and adding new features to the dataset, then it returns the prepared dataframe with the targets """
        target_name = ['y']
        df = self.add_day_hour_year(df)
        df = self.update_upper_lower(1150, 1150, df)
        if altered_forecasting:
            df = self.altered_forecasting(df)
            to_drop_list = ['start_time', 'start_day', 'start_hour', 'year', 'y_old']
        else:
            to_drop_list = ['start_time', 'start_day', 'start_hour', 'year']
        df = self.drop_col(df, to_drop_list)
        df = self.one_step_update(df, 5)
        df = self.add_previous_target(df, 1, target_name, 'prev_y')
        df = self.add_previous_target(df, 12, target_name, 'prev_hour')
        df = self.add_previous_target(df, 288, target_name, 'prev_day')
        df_targets = self.make_targets(df, shift_steps, target_name)
        df = self.add_avarage_targets(df, df['y'], 4)
        df = self.drop_col(df, target_name)
        return df, df_targets

    def main_preprocess(self, shift_steps, split=True, altered_forecasting=False):
        """ Prepare the dataset, make targets, scale the data, and get both train and test data (by splitting the train 
        data or keep it and use the validation dataset).
        It returns the data sequences for both train and test data with the targets.
        It returns also the train scaled data befor deviding into sequences.
        """
        self.train_df, self.target_train_df = self.prepare_dataset(self.train_df, shift_steps, altered_forecasting)
        self.test_df, self.target_test_df = self.prepare_dataset(self.test_df, shift_steps, altered_forecasting)

        x_train_scaled, y_train_scaled = self.scale_data(self.train_df, self.target_train_df, shift_steps, train=True)
        print(x_train_scaled.shape, y_train_scaled.shape)
        if split:
            x_train, y_train, x_test, y_test = self.train_test_split(x_train_scaled, y_train_scaled, 0.9)
        else:
            x_train = x_train_scaled
            y_train = y_train_scaled
            x_test, y_test = self.scale_data(self.test_df, self.target_test_df, shift_steps, train=False)
        # gen = self.batch_generator(x_train, y_train, 256, shift_steps, self.train_df.shape[1], len(x_train))
        gen = self.batch_generator_separated(x_train, y_train, shift_steps)
        x_val_batch, y_val_batch = self.batch_generator_separated(x_test, y_test, shift_steps)
        return gen, x_val_batch, y_val_batch, x_train_scaled, y_train_scaled

    def predict_multi_steps(self, model, time_length, x, y_true, step_size, start=0, show=False):
        """ Predices multi steps by sending a sequence of data and predict the next target, then updates the predicted target in 
        The previous features for the next sequences.
        It returns a list of the targets and a list of the predict targets
        """
        if x.shape[0] - start < time_length:
            print('wrong subset!')
            return False
        x_true_copied = copy.deepcopy(x)
        y_predicted = []
        start_point = start
        y_true_values = y_true[start_point:start_point + time_length]
        y_true_copied = copy.deepcopy(y_true)
        for i in range(time_length):
            x_batch = x_true_copied[start_point:start_point + step_size]
            x_batch = np.expand_dims(x_batch, axis=0)
            y_pred = model.predict(x_batch)
            y_pred[0] = y_pred[0] - 0.0
            x_true_copied[start_point + step_size + 1, -4] = y_pred[0]
            x_true_copied[start_point + step_size + 12, -3] = y_pred[0]
            x_true_copied[start_point + step_size + 288, -2] = y_pred[0]
            y_predicted.append(y_pred[0])
            y_true_copied[start_point+i] = y_pred[0]
            x_true_copied[:, -1] = np.roll(y_true_copied.reshape(len(y_true_copied),), 4, axis=None)
            start_point += 1
            if show:
                print('y_true', 'y_predicted')
                print(y_true_values[i], y_pred[0])
        return y_predicted, y_true_values

    def plot_forecast(self, y_true, y_true_forecast, y_predicted):
        """ Plotting the target for a period and the prediced targets for the next period with the correct targets for this next
        period in the future.
        """
        y_true_nan = np.full(np.array(y_true).shape, None)
        min_l = min(len(y_true), len(y_true_forecast))
        # Padding the lists with None to make correct plot
        y_true_forecast = np.concatenate((y_true_nan[:-min_l+1], y_true_forecast[:min_l-1]), axis=0).reshape((len(y_true),))
        y_predicted = np.concatenate((y_true_nan[:-min_l+1], y_predicted[:min_l-1]), axis=0).reshape((len(y_true),))
        y_true_new = np.concatenate((y_true[:-min_l+1], y_true_nan[:min_l-1]), axis=0).reshape((len(y_true),))
        fig, ax = plt.subplots()
        plt.title('Forecasting')
        ax.plot(list(range(len(y_true))), np.array(y_true_new), label='all')
        ax.plot(list(range(len(y_true))), np.array(y_true_forecast), label='true')
        ax.plot(list(range(len(y_true))), np.array(y_predicted), label='pred')
        leg = ax.legend()
        plt.show()


if __name__ == '__main__':
    model_forecast = PreProcess('no1_train.csv', 'no1_validation.csv')
    steps_shift = 12 * 2
    features_count = 24
    rnn_model = RNN_model((steps_shift, features_count), 1)
    rnn_model.define_RNN_model()
    g, x_val_batch, y_val_batch, x, y = model_forecast.main_preprocess(steps_shift, split=False, altered_forecasting=False)
    history = rnn_model.fit_model(g, (x_val_batch, y_val_batch), rnn_model.define_callbacks())
    print(history.history)

    x_test_scaled, y_test_scaled = model_forecast.scale_data(model_forecast.test_df, model_forecast.target_test_df, steps_shift)
    x_test_batch, y_test_batch = model_forecast.batch_generator_separated(x_test_scaled, y_test_scaled, steps_shift)

    rnn_model.evaluate_test(x_test_batch, y_test_batch)
    # rnn_model.load_checkpoint('27_1_checkpoint')
    y_predicted, y_true_forecast = model_forecast.predict_multi_steps(rnn_model, 12 * 2, x, y, steps_shift, 500)
    model_forecast.plot_forecast(y[400:524], y_true_forecast, y_predicted)
