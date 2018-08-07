
from math import *
from numpy import *
from random import *


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + exp(-in_x))


def grad_ascent(data_mat, class_labels):
    data_matrix = mat(data_mat)
    label_matrix = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    # weights = w.getA()
    data_mat, label_mat = load_data_set()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stochastic_gradient_ascent0(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i]*weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def stochastic_gradient_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights += alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    display_flag = False
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weights = stochastic_gradient_ascent1(array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0
    for line in fr_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        classify_result = classify_vector(array(line_arr), train_weights)
        if int(classify_result) != int(curr_line[21]):
            error_count += 1
            if display_flag:
                print('%d classify error: %d, should be %d' % (num_test_vec, int(classify_result), int(curr_line[21])))
    error_rate = float(error_count) / num_test_vec
    print('error rate: ', error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0
    for k in range(num_tests):
        error_sum += colic_test()
    print('after %d iterations the average error rate is %f' % (num_tests, error_sum / float(num_tests)))


if __name__ == '__main__':
    # data_arr, label_mat = load_data_set()
    # w = grad_ascent(data_arr, label_mat)
    # print(w)
    # plot_best_fit(w.getA())
    #
    # data_arr, label_mat = load_data_set()
    # w = stochastic_gradient_ascent0(array(data_arr), label_mat)
    # print(w)
    # plot_best_fit(w)
    #
    # data_arr, label_mat = load_data_set()
    # w = stochastic_gradient_ascent1(array(data_arr), label_mat)
    # print(w)
    # plot_best_fit(w)
    #
    # data_arr, label_mat = load_data_set()
    # w = stochastic_gradient_ascent1(array(data_arr), label_mat, 4000)
    # print(w)
    # plot_best_fit(w)

    multi_test()
















































