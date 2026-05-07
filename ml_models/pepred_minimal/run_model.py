# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
#
#
# class PE_Model(nn.Module):
#     def __init__(self, input_size=29, hidden1_size=200, hidden2_size=50, activation_fun=nn.ReLU(), dropout=0.05):
#         super(PE_Model, self).__init__()
#
#         self.layer_1 = nn.Linear(input_size, hidden1_size)
#         self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
#         self.layer_out = nn.Linear(hidden2_size, 1)
#
#         self.activation = activation_fun
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, inputs):
#         x = self.activation(self.layer_1(inputs))
#         x = self.activation(self.layer_2(x))
#         x = self.dropout(x)
#         x = self.layer_out(x)
#         return x
#
#
# def fill_missing_data(user_data, features_data):
#     data_type = dict(zip(features_data["feature"], features_data["numeric/category"]))
#     median_values = dict(zip(features_data['feature'], features_data['median']))
#
#     for key, val in user_data.items():
#         if val is None:  # todo. maybe == 'unknown'?
#             user_data[key] = median_values[key] if data_type[key] == 'category' else float(median_values[key])
#
#
# def category2oneHot(user_data, features_data):
#     data_type = dict(zip(features_data["feature"], features_data["numeric/category"]))
#     options_dict = dict(zip(features_data["feature"], features_data["valid_options"]))
#
#     for key, val in user_data.items():
#         if data_type[key] == 'category':
#             options_lst = options_dict[key].split('/')
#             options = {x.split('=')[0]: int(x.split('=')[1]) for x in options_lst}
#             if len(options) == 2:
#                 user_data[key] = options[val]
#             else:
#                 oneHot = [0] * len(options)
#                 oneHot[options[val]] = 1
#                 user_data[key] = oneHot
#
#
# def data2vector(user_data, features_data):
#     vec = []
#     features_lst = features_data["feature"]
#
#     for feat in features_lst:
#         if not isinstance(user_data[feat], list):
#             user_data[feat] = [user_data[feat]]
#         vec += user_data[feat]
#
#     return vec
#
#
# def z_score_normalization(vec, features_data):
#     final_means, final_vars = [], []
#     means_, vars_ = features_data["mean"], features_data["var"]
#     for m, v in zip(means_, vars_):
#         if '/' in m:
#             final_means += m.split('/')
#             final_vars += v.split('/')
#         else:
#             final_means.append(m)
#             final_vars.append(v)
#     final_means = np.array(final_means, dtype=float)
#     final_vars = np.array(final_vars, dtype=float)
#
#     return (np.array(vec) - final_means) / np.sqrt(final_vars)
#
#
# def processing(user_data, features_data):
#     fill_missing_data(user_data, features_data)
#     category2oneHot(user_data, features_data)
#     vec = data2vector(user_data, features_data)
#     vec = z_score_normalization(vec, features_data)
#
#     return vec
#
#
# def main(user_data, binary_case):  # user_data need to be dict, binary_case=1/3
#     saved_model_path = f'models/full_data_case{binary_case}.pt' # load
#     features_data = pd.read_csv('Features.csv')
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     user_data = processing(user_data, features_data)
#     user_data = torch.FloatTensor(user_data)
#
#     if torch.cuda.is_available(): # load
#         model_dict = torch.load(saved_model_path)
#     else:
#         model_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
#
#     model = PE_Model().to(device) # load
#     state_dict = model_dict['model_state_dict'] # load
#     model.load_state_dict(state_dict) # load
#
#     sigmoid = nn.Sigmoid()
#     model.eval()
#     with torch.no_grad():
#         user_data = user_data.to(device)
#         pred = model(user_data)
#         pred = sigmoid(pred).item()
#     return pred
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class PE_Model(nn.Module):
    def __init__(self, is_full, input_size=29, hidden1_size=200, hidden2_size=50, activation_fun=nn.ReLU()):
        super(PE_Model, self).__init__()

        if not is_full:
            input_size = 23
            hidden1_size = 300
            hidden2_size = 200

        self.layer_1 = nn.Linear(input_size, hidden1_size)
        self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer_out = nn.Linear(hidden2_size, 1)

        self.activation = activation_fun

    def forward(self, inputs):
        x = self.activation(self.layer_1(inputs))
        x = self.activation(self.layer_2(x))
        x = self.layer_out(x)
        return x


def fill_missing_data(user_data, features_data):
    data_type = dict(zip(features_data["feature"], features_data["numeric/category"]))
    median_values = dict(zip(features_data['feature'], features_data['median']))

    for key, val in user_data.items():
        if val is None:  # todo. maybe == 'unknown'?
            user_data[key] = median_values[key] if data_type[key] == 'category' else float(median_values[key])


def category2oneHot(user_data, features_data):
    data_type = dict(zip(features_data["feature"], features_data["numeric/category"]))
    options_dict = dict(zip(features_data["feature"], features_data["valid_options"]))

    for key, val in user_data.items():
        if data_type[key] == 'category':
            options_lst = options_dict[key].split('/')
            options = {x.split('=')[0]: int(x.split('=')[1]) for x in options_lst}
            if len(options) == 2:
                user_data[key] = options[val]
            else:
                oneHot = [0] * len(options)
                oneHot[options[val]] = 1
                user_data[key] = oneHot


def data2vector(user_data, features_data):
    vec = []
    features_lst = features_data["feature"]

    for feat in features_lst:
        if feat in user_data:  # in partial data, not all features are existing
            if not isinstance(user_data[feat], list):
                user_data[feat] = [user_data[feat]]
            vec += user_data[feat]

    return vec


def z_score_normalization(vec, features_data, is_full):
    final_means, final_vars = [], []
    means_, vars_ = features_data["mean"], features_data["var"]

    # indexes of features that exist in full data only, so in partial data we ignore them
    idx_ignore = [0, 1, 2, 3, 4, 10]  # ['ga', 'pappa', 'plgf', 'utpi', 'map', 'plgf.machine']

    for idx, (m, v) in enumerate(zip(means_, vars_)):

        if not is_full and idx in idx_ignore:  # partial data and this is feature that exists in full data only
            continue

        if '/' in m:
            final_means += m.split('/')
            final_vars += v.split('/')
        else:
            final_means.append(m)
            final_vars.append(v)
    final_means = np.array(final_means, dtype=float)
    final_vars = np.array(final_vars, dtype=float)

    return (np.array(vec) - final_means) / np.sqrt(final_vars)


def processing(user_data, features_data, is_full):
    fill_missing_data(user_data, features_data)
    category2oneHot(user_data, features_data)
    vec = data2vector(user_data, features_data)
    vec = z_score_normalization(vec, features_data, is_full)

    return vec


def main(user_data, binary_case, is_full):
    """
    :param user_data: a dict with keys according the file 'Features.csv'. when data is not full,
           the following features are omitted: ga, pappa, plgf, utpi, map, plgf.machine
    :param binary_case: 1 or 3
    :param is_full: True or False
    """
    full_or_par = 'full' if is_full else 'partial'
    saved_model_path = f'models/{full_or_par}_data_case{binary_case}.pt' # load
    features_data = pd.read_csv('Features.csv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_data = processing(user_data, features_data, is_full)
    user_data = torch.FloatTensor(user_data)

    if torch.cuda.is_available(): # load
        model_dict = torch.load(saved_model_path)
    else:
        model_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))

    model = PE_Model(is_full).to(device) # load
    state_dict = model_dict['model_state_dict'] # load
    model.load_state_dict(state_dict) # load

    sigmoid = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        user_data = user_data.to(device)
        pred = model(user_data)
        pred = sigmoid(pred).item()
    return pred