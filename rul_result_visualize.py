from matplotlib import pyplot as plt


def result_visualize(actual_targets, predict_result, num_test_units, data_r_mse, name):

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    plt.xticks(fontproperties='Times New Roman', size=18)
    plt.yticks(fontproperties='Times New Roman', size=18)

    plt.tick_params(axis='both', which='major', direction='in', width=1, length=5, pad=5)

    plt.axvline(x=num_test_units, c='r', linestyle='--')

    plt.plot(actual_targets, label='Actual RUL')
    plt.plot(predict_result, label='Predicted RUL (R_MSE = {})'.format(round(data_r_mse, 3)))
    plt.title('Remaining Useful Life Prediction', fontdict={'family': "Times New Roman", "size": 20})
    plt.xlabel("Time(10s)", fontdict={'family': "Times New Roman", "size": 20}, labelpad=5)
    plt.ylabel("RUL", fontdict={'family': "Times New Roman", "size": 20}, labelpad=5)
    plt.legend(loc='upper left', prop={'family': "Times New Roman", "size": 18})

    plt.savefig('./visualize_result/seed1_CNN_Bi-LSTM {} {} RUL Prediction with LSTM.png'.format(name, round(data_r_mse, 3)))
    plt.show()
