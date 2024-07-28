import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from torch.autograd import Variable
from torch.utils.data import DataLoader

from project.rul_pre_data import get_train_data, get_test_data
from project.rul_result_visualize import result_visualize


def load_model():
    predict_model = torch.load('E:\\pythonProject\\project\\model\\baseline.pth')

    return predict_model


def calculate_mape(predicted, actual):
    epsilon = 1e-9
    return np.mean(np.abs((predicted - actual) / (actual + epsilon))) * 100


def test_model(data, label, name):
    total_predict_data = []
    test_data = Variable(torch.Tensor(np.asarray(data, dtype='float64')))
    actual_rul = Variable(torch.Tensor(np.asarray(label, dtype='float64')))
    test_data_p = torch.permute(test_data, (0, 2, 1))
    test_data_final = torch.reshape(test_data_p, (test_data_p.shape[0], 1, test_data_p.shape[1], test_data_p.shape[2]))
    test_data_loader = DataLoader(test_data_final, batch_size=32)

    print("--------test begin--------")
    model.eval()

    with torch.no_grad():
        for test_sub_data in test_data_loader:
            output = model.forward(test_sub_data)

            for predict_label in output:
                total_predict_data.append(predict_label)

    mse = mean_squared_error(total_predict_data, actual_rul)
    mae = mean_absolute_error(total_predict_data, actual_rul)
    r2 = r2_score(total_predict_data, actual_rul)
    r_mse = np.sqrt(mse)
    mape = calculate_mape(np.array(total_predict_data), np.array(actual_rul))
    result_data = [[name, mae, mse, r_mse, mape, r2]]
    pd_data = pd.DataFrame(result_data, columns=['Bearing', 'MAE', 'MSE', 'R_MSE', 'MAPE', 'r2'])

    print('{} MAE: {}'.format(name, mae))
    print('{} MSE: {}'.format(name, mse))
    print('{} R_MSE: {}'.format(name, r_mse))
    print('{} MAPE: {}'.format(name, mape))

    print('')

    print("--------test end--------")

    return r_mse, total_predict_data, pd_data


model = load_model()

data_0, label_0, data_1, label_1, data_2, label_2, data_3, label_3, data_4, label_4, data_5, label_5 = get_train_data()

data_6, label_6, data_7, label_7, data_8, label_8, data_9, label_9, data_10, label_10, data_11, label_11, data_12, \
    label_12, data_13, label_13, data_14, label_14, data_15, label_15, data_16, label_16 = get_test_data()

result_data_0, predict_rul_0, index_0 = test_model(data_0[0], label_0[0], '1_1')
result_data_1, predict_rul_1, index_1 = test_model(data_1[0], label_1[0], '1_2')
result_data_2, predict_rul_2, index_2 = test_model(data_2[0], label_2[0], '2_1')
result_data_3, predict_rul_3, index_3 = test_model(data_3[0], label_3[0], '2_2')
result_data_4, predict_rul_4, index_4 = test_model(data_4[0], label_4[0], '3_1')
result_data_5, predict_rul_5, index_5 = test_model(data_5[0], label_5[0], '3_2')
result_data_6, predict_rul_6, index_6 = test_model(data_6[0], label_6[0], '1_3')
result_data_7, predict_rul_7, index_7 = test_model(data_7[0], label_7[0], '1_4')
result_data_8, predict_rul_8, index_8 = test_model(data_8[0], label_8[0], '1_5')
result_data_9, predict_rul_9, index_9 = test_model(data_9[0], label_9[0], '1_6')
result_data_10, predict_rul_10, index_10 = test_model(data_10[0], label_10[0], '1_7')
result_data_11, predict_rul_11, index_11 = test_model(data_11[0], label_11[0], '2_3')
result_data_12, predict_rul_12, index_12 = test_model(data_12[0], label_12[0], '2_4')
result_data_13, predict_rul_13, index_13 = test_model(data_13[0], label_13[0], '2_5')
result_data_14, predict_rul_14, index_14 = test_model(data_14[0], label_14[0], '2_6')
result_data_15, predict_rul_15, index_15 = test_model(data_15[0], label_15[0], '2_7')
result_data_16, predict_rul_16, index_16 = test_model(data_16[0], label_16[0], '3_3')

total_result_data_0 = result_data_0
total_result_data_1 = result_data_1
total_result_data_2 = result_data_2
total_result_data_3 = result_data_3
total_result_data_4 = result_data_4
total_result_data_5 = result_data_5
total_result_data_6 = result_data_6
total_result_data_7 = result_data_7
total_result_data_8 = result_data_8
total_result_data_9 = result_data_9
total_result_data_10 = result_data_10
total_result_data_11 = result_data_11
total_result_data_12 = result_data_12
total_result_data_13 = result_data_13
total_result_data_14 = result_data_14
total_result_data_15 = result_data_15
total_result_data_16 = result_data_16
total_predict_rul_0 = predict_rul_0
total_predict_rul_1 = predict_rul_1
total_predict_rul_2 = predict_rul_2
total_predict_rul_3 = predict_rul_3
total_predict_rul_4 = predict_rul_4
total_predict_rul_5 = predict_rul_5
total_predict_rul_6 = predict_rul_6
total_predict_rul_7 = predict_rul_7
total_predict_rul_8 = predict_rul_8
total_predict_rul_9 = predict_rul_9
total_predict_rul_10 = predict_rul_10
total_predict_rul_11 = predict_rul_11
total_predict_rul_12 = predict_rul_12
total_predict_rul_13 = predict_rul_13
total_predict_rul_14 = predict_rul_14
total_predict_rul_15 = predict_rul_15
total_predict_rul_16 = predict_rul_16
total_index_0 = index_0
total_index_1 = index_1
total_index_2 = index_2
total_index_3 = index_3
total_index_4 = index_4
total_index_5 = index_5
total_index_6 = index_6
total_index_7 = index_7
total_index_8 = index_8
total_index_9 = index_9
total_index_10 = index_10
total_index_11 = index_11
total_index_12 = index_12
total_index_13 = index_13
total_index_14 = index_14
total_index_15 = index_15
total_index_16 = index_16


for i in range(1, 10):
    result_data_0, predict_rul_0, index_0 = test_model(data_0[i], label_0[i], '1_1')
    result_data_1, predict_rul_1, index_1 = test_model(data_1[i], label_1[i], '1_2')
    result_data_2, predict_rul_2, index_2 = test_model(data_2[i], label_2[i], '2_1')
    result_data_3, predict_rul_3, index_3 = test_model(data_3[i], label_3[i], '2_2')
    result_data_4, predict_rul_4, index_4 = test_model(data_4[i], label_4[i], '3_1')
    result_data_5, predict_rul_5, index_5 = test_model(data_5[i], label_5[i], '3_2')
    result_data_6, predict_rul_6, index_6 = test_model(data_6[i], label_6[i], '1_3')
    result_data_7, predict_rul_7, index_7 = test_model(data_7[i], label_7[i], '1_4')
    result_data_8, predict_rul_8, index_8 = test_model(data_8[i], label_8[i], '1_5')
    result_data_9, predict_rul_9, index_9 = test_model(data_9[i], label_9[i], '1_6')
    result_data_10, predict_rul_10, index_10 = test_model(data_10[i], label_10[i], '1_7')
    result_data_11, predict_rul_11, index_11 = test_model(data_11[i], label_11[i], '2_3')
    result_data_12, predict_rul_12, index_12 = test_model(data_12[i], label_12[i], '2_4')
    result_data_13, predict_rul_13, index_13 = test_model(data_13[i], label_13[i], '2_5')
    result_data_14, predict_rul_14, index_14 = test_model(data_14[i], label_14[i], '2_6')
    result_data_15, predict_rul_15, index_15 = test_model(data_15[i], label_15[i], '2_7')
    result_data_16, predict_rul_16, index_16 = test_model(data_16[i], label_16[i], '3_3')

    total_result_data_0 += result_data_0
    total_result_data_1 += result_data_1
    total_result_data_2 += result_data_2
    total_result_data_3 += result_data_3
    total_result_data_4 += result_data_4
    total_result_data_5 += result_data_5
    total_result_data_6 += result_data_6
    total_result_data_7 += result_data_7
    total_result_data_8 += result_data_8
    total_result_data_9 += result_data_9
    total_result_data_10 += result_data_10
    total_result_data_11 += result_data_11
    total_result_data_12 += result_data_12
    total_result_data_13 += result_data_13
    total_result_data_14 += result_data_14
    total_result_data_15 += result_data_15
    total_result_data_16 += result_data_16

    total_predict_rul_0 = np.add(total_predict_rul_0, predict_rul_0).tolist()
    total_predict_rul_1 = np.add(total_predict_rul_1, predict_rul_1).tolist()
    total_predict_rul_2 = np.add(total_predict_rul_2, predict_rul_2).tolist()
    total_predict_rul_3 = np.add(total_predict_rul_3, predict_rul_3).tolist()
    total_predict_rul_4 = np.add(total_predict_rul_4, predict_rul_4).tolist()
    total_predict_rul_5 = np.add(total_predict_rul_5, predict_rul_5).tolist()
    total_predict_rul_6 = np.add(total_predict_rul_6, predict_rul_6).tolist()
    total_predict_rul_7 = np.add(total_predict_rul_7, predict_rul_7).tolist()
    total_predict_rul_8 = np.add(total_predict_rul_8, predict_rul_8).tolist()
    total_predict_rul_9 = np.add(total_predict_rul_9, predict_rul_9).tolist()
    total_predict_rul_10 = np.add(total_predict_rul_10, predict_rul_10).tolist()
    total_predict_rul_11 = np.add(total_predict_rul_11, predict_rul_11).tolist()
    total_predict_rul_12 = np.add(total_predict_rul_12, predict_rul_12).tolist()
    total_predict_rul_13 = np.add(total_predict_rul_13, predict_rul_13).tolist()
    total_predict_rul_14 = np.add(total_predict_rul_14, predict_rul_14).tolist()
    total_predict_rul_15 = np.add(total_predict_rul_15, predict_rul_15).tolist()
    total_predict_rul_16 = np.add(total_predict_rul_16, predict_rul_16).tolist()

    total_index_0 += index_0
    total_index_1 += index_1
    total_index_2 += index_2
    total_index_3 += index_3
    total_index_4 += index_4
    total_index_5 += index_5
    total_index_6 += index_6
    total_index_7 += index_7
    total_index_8 += index_8
    total_index_9 += index_9
    total_index_10 += index_10
    total_index_11 += index_11
    total_index_12 += index_12
    total_index_13 += index_13
    total_index_14 += index_14
    total_index_15 += index_15
    total_index_16 += index_16


total_result_data_0 /= 10
total_result_data_1 /= 10
total_result_data_2 /= 10
total_result_data_3 /= 10
total_result_data_4 /= 10
total_result_data_5 /= 10
total_result_data_6 /= 10
total_result_data_7 /= 10
total_result_data_8 /= 10
total_result_data_9 /= 10
total_result_data_10 /= 10
total_result_data_11 /= 10
total_result_data_12 /= 10
total_result_data_13 /= 10
total_result_data_14 /= 10
total_result_data_15 /= 10
total_result_data_16 /= 10
total_predict_rul_0 = [x / 10 for x in total_predict_rul_0]
total_predict_rul_1 = [x / 10 for x in total_predict_rul_1]
total_predict_rul_2 = [x / 10 for x in total_predict_rul_2]
total_predict_rul_3 = [x / 10 for x in total_predict_rul_3]
total_predict_rul_4 = [x / 10 for x in total_predict_rul_4]
total_predict_rul_5 = [x / 10 for x in total_predict_rul_5]
total_predict_rul_6 = [x / 10 for x in total_predict_rul_6]
total_predict_rul_7 = [x / 10 for x in total_predict_rul_7]
total_predict_rul_8 = [x / 10 for x in total_predict_rul_8]
total_predict_rul_9 = [x / 10 for x in total_predict_rul_9]
total_predict_rul_10 = [x / 10 for x in total_predict_rul_10]
total_predict_rul_11 = [x / 10 for x in total_predict_rul_11]
total_predict_rul_12 = [x / 10 for x in total_predict_rul_12]
total_predict_rul_13 = [x / 10 for x in total_predict_rul_13]
total_predict_rul_14 = [x / 10 for x in total_predict_rul_14]
total_predict_rul_15 = [x / 10 for x in total_predict_rul_15]
total_predict_rul_16 = [x / 10 for x in total_predict_rul_16]

total_index_0['MAE'] = total_index_0['MAE'].astype(float) / 10
total_index_0['MSE'] = total_index_0['MSE'].astype(float) / 10
total_index_0['R_MSE'] = total_index_0['R_MSE'].astype(float) / 10
total_index_0['MAPE'] = total_index_0['MAPE'].astype(float) / 10
total_index_0['r2'] = total_index_0['r2'].astype(float) / 10
total_index_1['MAE'] = total_index_1['MAE'].astype(float) / 10
total_index_1['MSE'] = total_index_1['MSE'].astype(float) / 10
total_index_1['R_MSE'] = total_index_1['R_MSE'].astype(float) / 10
total_index_1['MAPE'] = total_index_1['MAPE'].astype(float) / 10
total_index_1['r2'] = total_index_1['r2'].astype(float) / 10
total_index_2['MAE'] = total_index_2['MAE'].astype(float) / 10
total_index_2['MSE'] = total_index_2['MSE'].astype(float) / 10
total_index_2['R_MSE'] = total_index_2['R_MSE'].astype(float) / 10
total_index_2['MAPE'] = total_index_2['MAPE'].astype(float) / 10
total_index_2['r2'] = total_index_2['r2'].astype(float) / 10
total_index_3['MAE'] = total_index_3['MAE'].astype(float) / 10
total_index_3['MSE'] = total_index_3['MSE'].astype(float) / 10
total_index_3['R_MSE'] = total_index_3['R_MSE'].astype(float) / 10
total_index_3['MAPE'] = total_index_3['MAPE'].astype(float) / 10
total_index_3['r2'] = total_index_3['r2'].astype(float) / 10
total_index_4['MAE'] = total_index_4['MAE'].astype(float) / 10
total_index_4['MSE'] = total_index_4['MSE'].astype(float) / 10
total_index_4['R_MSE'] = total_index_4['R_MSE'].astype(float) / 10
total_index_4['MAPE'] = total_index_4['MAPE'].astype(float) / 10
total_index_4['r2'] = total_index_4['r2'].astype(float) / 10
total_index_5['MAE'] = total_index_5['MAE'].astype(float) / 10
total_index_5['MSE'] = total_index_5['MSE'].astype(float) / 10
total_index_5['R_MSE'] = total_index_5['R_MSE'].astype(float) / 10
total_index_5['MAPE'] = total_index_5['MAPE'].astype(float) / 10
total_index_5['r2'] = total_index_5['r2'].astype(float) / 10
total_index_6['MAE'] = total_index_6['MAE'].astype(float) / 10
total_index_6['MSE'] = total_index_6['MSE'].astype(float) / 10
total_index_6['R_MSE'] = total_index_6['R_MSE'].astype(float) / 10
total_index_6['MAPE'] = total_index_6['MAPE'].astype(float) / 10
total_index_6['r2'] = total_index_6['r2'].astype(float) / 10
total_index_7['MAE'] = total_index_7['MAE'].astype(float) / 10
total_index_7['MSE'] = total_index_7['MSE'].astype(float) / 10
total_index_7['R_MSE'] = total_index_7['R_MSE'].astype(float) / 10
total_index_7['MAPE'] = total_index_7['MAPE'].astype(float) / 10
total_index_7['r2'] = total_index_7['r2'].astype(float) / 10
total_index_8['MAE'] = total_index_8['MAE'].astype(float) / 10
total_index_8['MSE'] = total_index_8['MSE'].astype(float) / 10
total_index_8['R_MSE'] = total_index_8['R_MSE'].astype(float) / 10
total_index_8['MAPE'] = total_index_8['MAPE'].astype(float) / 10
total_index_8['r2'] = total_index_8['r2'].astype(float) / 10
total_index_9['MAE'] = total_index_9['MAE'].astype(float) / 10
total_index_9['MSE'] = total_index_9['MSE'].astype(float) / 10
total_index_9['R_MSE'] = total_index_9['R_MSE'].astype(float) / 10
total_index_9['MAPE'] = total_index_9['MAPE'].astype(float) / 10
total_index_9['r2'] = total_index_9['r2'].astype(float) / 10
total_index_10['MAE'] = total_index_10['MAE'].astype(float) / 10
total_index_10['MSE'] = total_index_10['MSE'].astype(float) / 10
total_index_10['R_MSE'] = total_index_10['R_MSE'].astype(float) / 10
total_index_10['MAPE'] = total_index_10['MAPE'].astype(float) / 10
total_index_10['r2'] = total_index_10['r2'].astype(float) / 10
total_index_11['MAE'] = total_index_11['MAE'].astype(float) / 10
total_index_11['MSE'] = total_index_11['MSE'].astype(float) / 10
total_index_11['R_MSE'] = total_index_11['R_MSE'].astype(float) / 10
total_index_11['MAPE'] = total_index_11['MAPE'].astype(float) / 10
total_index_11['r2'] = total_index_11['r2'].astype(float) / 10
total_index_12['MAE'] = total_index_12['MAE'].astype(float) / 10
total_index_12['MSE'] = total_index_12['MSE'].astype(float) / 10
total_index_12['R_MSE'] = total_index_12['R_MSE'].astype(float) / 10
total_index_12['MAPE'] = total_index_12['MAPE'].astype(float) / 10
total_index_12['r2'] = total_index_12['r2'].astype(float) / 10
total_index_13['MAE'] = total_index_13['MAE'].astype(float) / 10
total_index_13['MSE'] = total_index_13['MSE'].astype(float) / 10
total_index_13['R_MSE'] = total_index_13['R_MSE'].astype(float) / 10
total_index_13['MAPE'] = total_index_13['MAPE'].astype(float) / 10
total_index_13['r2'] = total_index_13['r2'].astype(float) / 10
total_index_14['MAE'] = total_index_14['MAE'].astype(float) / 10
total_index_14['MSE'] = total_index_14['MSE'].astype(float) / 10
total_index_14['R_MSE'] = total_index_14['R_MSE'].astype(float) / 10
total_index_14['MAPE'] = total_index_14['MAPE'].astype(float) / 10
total_index_14['r2'] = total_index_14['r2'].astype(float) / 10
total_index_15['MAE'] = total_index_15['MAE'].astype(float) / 10
total_index_15['MSE'] = total_index_15['MSE'].astype(float) / 10
total_index_15['R_MSE'] = total_index_15['R_MSE'].astype(float) / 10
total_index_15['MAPE'] = total_index_15['MAPE'].astype(float) / 10
total_index_15['r2'] = total_index_15['r2'].astype(float) / 10
total_index_16['MAE'] = total_index_16['MAE'].astype(float) / 10
total_index_16['MSE'] = total_index_16['MSE'].astype(float) / 10
total_index_16['R_MSE'] = total_index_16['R_MSE'].astype(float) / 10
total_index_16['MAPE'] = total_index_16['MAPE'].astype(float) / 10
total_index_16['r2'] = total_index_16['r2'].astype(float) / 10

total_index_0['Bearing'] = ['1_1']
total_index_1['Bearing'] = ['1_2']
total_index_2['Bearing'] = ['2_1']
total_index_3['Bearing'] = ['2_2']
total_index_4['Bearing'] = ['3_1']
total_index_5['Bearing'] = ['3_2']
total_index_6['Bearing'] = ['1_3']
total_index_7['Bearing'] = ['1_4']
total_index_8['Bearing'] = ['1_5']
total_index_9['Bearing'] = ['1_6']
total_index_10['Bearing'] = ['1_7']
total_index_11['Bearing'] = ['2_3']
total_index_12['Bearing'] = ['2_4']
total_index_13['Bearing'] = ['2_5']
total_index_14['Bearing'] = ['2_6']
total_index_15['Bearing'] = ['2_7']
total_index_16['Bearing'] = ['3_3']

final_data = pd.concat([total_index_0, total_index_1, total_index_6, total_index_7, total_index_8, total_index_9, total_index_10,
                        total_index_2, total_index_3, total_index_11, total_index_12, total_index_13, total_index_14, total_index_15,
                        total_index_4, total_index_5, total_index_16])
with pd.ExcelWriter('./data/total_result.xlsx', mode='a', engine='openpyxl') as writer:
    final_data.to_excel(writer, index=False, sheet_name='CNN_Bi-LSTM')

# 结果可视化
result_visualize(label_0[0], total_predict_rul_0, len(data_0[0]), total_result_data_0, 'Bearing 1_1')
result_visualize(label_1[0], total_predict_rul_1, len(data_1[0]), total_result_data_1, 'Bearing 1_2')
result_visualize(label_6[0], total_predict_rul_6, len(data_6[0]), total_result_data_6, 'Bearing 1_3')
result_visualize(label_7[0], total_predict_rul_7, len(data_7[0]), total_result_data_7, 'Bearing 1_4')
result_visualize(label_8[0], total_predict_rul_8, len(data_8[0]), total_result_data_8, 'Bearing 1_5')
result_visualize(label_9[0], total_predict_rul_9, len(data_9[0]), total_result_data_9, 'Bearing 1_6')
result_visualize(label_10[0], total_predict_rul_10, len(data_10[0]), total_result_data_10, 'Bearing 1_7')

result_visualize(label_2[0], total_predict_rul_2, len(data_2[0]), total_result_data_2, 'Bearing 2_1')
result_visualize(label_3[0], total_predict_rul_3, len(data_3[0]), total_result_data_3, 'Bearing 2_2')
result_visualize(label_11[0], total_predict_rul_11, len(data_11[0]), total_result_data_11, 'Bearing 2_3')
result_visualize(label_12[0], total_predict_rul_12, len(data_12[0]), total_result_data_12, 'Bearing 2_4')
result_visualize(label_13[0], total_predict_rul_13, len(data_13[0]), total_result_data_13, 'Bearing 2_5')
result_visualize(label_14[0], total_predict_rul_14, len(data_14[0]), total_result_data_14, 'Bearing 2_6')
result_visualize(label_15[0], total_predict_rul_15, len(data_15[0]), total_result_data_15, 'Bearing 2_7')

result_visualize(label_4[0], total_predict_rul_4, len(data_4[0]), total_result_data_4, 'Bearing 3_1')
result_visualize(label_5[0], total_predict_rul_5, len(data_5[0]), total_result_data_5, 'Bearing 3_2')
result_visualize(label_16[0], total_predict_rul_16, len(data_16[0]), total_result_data_16, 'Bearing 3_7')
