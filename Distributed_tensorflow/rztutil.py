"""
RZT Utils

Created on 20 May 2016,
Updated on 06th Dec 2016
@author: Prathyush SP, Umesh Kumar
@version: 0.0.6

"""

import pandas as pd
from ast import literal_eval
import numpy as np
from pyhdfs import HdfsClient


def convert_df_tolist(*input_data):
    """
    @author: Prathyush SP

    Convert Dataframe to List

    ..todo::
        Prathyush SP; check list logic
    :param input_data: Input Data (*args)
    :return: Dataframe
    """
    # todo Prathyush SP: Perform Dataframe Validation
    dataframes = []
    for df in input_data:
        if isinstance(df, pd.DataFrame):
            if len(input_data) == 1:
                return df.values.tolist()
            dataframes.append(df.values.tolist())
        elif isinstance(df, pd.Series):
            df_list = df.to_frame().values.tolist()
            if isinstance(df_list, list):
                if isinstance(df_list[0][0], list):
                    dataframes.append([i[0] for i in df.to_frame().values.tolist()])
                else:
                    dataframes.append(df.to_frame().values.tolist())
            else:
                dataframes.append(df.to_frame().values.tolist())
    return dataframes


def read_csv(filename, split_ratio, delimiter=',', normalize=False, dtype=None, header=None, skiprows=None,
             index_col=False, output_label=True, randomize=False, return_as_dataframe=False, describe=False,
             label_vector=False):
    """
    @author: Prathyush SP

    The function is used to read a csv file with a specified delimiter

    :param filename: File name with absolute path
    :param split_ratio: Ratio used to split data into train and test
    :param delimiter: Delimiter used to split columns
    :param normalize: Normalize the Data
    :param dtype: Data Format
    :param header: Column Header
    :param skiprows: Skip specified number of rows
    :param index_col: Index Column
    :param output_label: Column which specifies whether output label should be available or not.
    :param randomize: Randomize data
    :param return_as_dataframe: Returns as a dataframes
    :param describe: Describe Input Data
    :param label_vector: True if output label is a vector
    :return: return train_data, train_label, test_data, test_label based on return_as_dataframe
    """
    df = pd.read_csv(filename, sep=delimiter, index_col=index_col, header=header, dtype=dtype, skiprows=skiprows)
    if describe:
        print(df.describe())
    df = df.sample(frac=1) if randomize else df
    df = df.apply(lambda x: np.log(x)) if normalize else df
    if split_ratio < 0 or split_ratio > 100:
        raise Exception('Split Ratio should be between 0 and 100')
    elif split_ratio == 0 or split_ratio == 100:
        if output_label is None or output_label is False:
            if return_as_dataframe:
                return df
            else:
                return convert_df_tolist(df)
        else:
            column_drop = len(df.columns) - 1 if output_label is True else output_label
            if header is None:
                label = df[column_drop].apply(literal_eval) if label_vector else df[column_drop]
                data = df.drop(column_drop, axis=1)
            else:
                label = df[[column_drop]]
                data = df.drop(df.columns[column_drop], axis=1)
            if return_as_dataframe:
                return data, label
            else:
                return convert_df_tolist(data, label)
    else:
        train_size = len(df) * split_ratio / 100
        test_size = len(df) - train_size
        train_data_df, test_data_df = df.head(int(train_size)), df.tail(int(test_size))
        if output_label is None or output_label is False:
            if return_as_dataframe:
                return train_data_df, test_data_df
            else:
                return convert_df_tolist(train_data_df, test_data_df)
        elif output_label is not None:
            column_drop = len(train_data_df.columns) - 1 if output_label is True else output_label
            if header is None:
                train_label_df = train_data_df[column_drop].apply(literal_eval) if label_vector else train_data_df[
                    column_drop]
                train_data_df = train_data_df.drop(column_drop, axis=1)
                test_label_df = test_data_df[column_drop].apply(literal_eval) if label_vector else test_data_df[
                    column_drop]
                test_data_df = test_data_df.drop(column_drop, axis=1)
            else:
                train_label_df = train_data_df[[column_drop]]
                train_data_df = train_data_df.drop(df.columns[column_drop], axis=1)
                test_label_df = test_data_df[[column_drop]]
                test_data_df = test_data_df.drop(df.columns[column_drop], axis=1)
            if return_as_dataframe:
                return train_data_df, train_label_df, test_data_df, test_label_df
            else:
                return convert_df_tolist(train_data_df, train_label_df, test_data_df, test_label_df)


def read_hdfs(filename, host, split_ratio, delimiter=',', normalize=False, dtype=None, header=None, skiprows=None,
              index_col=False, output_label=True, randomize=False, return_as_dataframe=False, describe=False,
              label_vector=False):
    client = HdfsClient(hosts=host)
    return read_csv(client.open(filename), split_ratio, delimiter=delimiter, normalize=normalize, dtype=dtype,
                    header=header, skiprows=skiprows, index_col=index_col, output_label=output_label,
                    randomize=randomize, return_as_dataframe=return_as_dataframe, describe=describe,
                    label_vector=label_vector)