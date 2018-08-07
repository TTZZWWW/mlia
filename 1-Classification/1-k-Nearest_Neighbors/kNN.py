
from numpy import *
import operator
from os import listdir


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    fr = open(filename)
    num_lines = len(fr.readlines())
    return_mat = zeros((num_lines, 3))
    class_label_vector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)
    error_count = 0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs: m, :], dating_labels[num_test_vecs: m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    print('total error rate is: %f' % float(error_count/num_test_vecs))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed every week?'))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals)/ranges, norm_mat, dating_labels, 3)
    print('you probably like this person:', result_list[classifier_result - 1])


def img2vector(filename):
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32*i+j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        filename = training_file_list[i]
        file_str = filename.split('.')[0]
        class_num = int(file_str.split('_')[0])
        hw_labels.append(class_num)
        training_mat[i, :] = img2vector('trainingDigits/%s' % filename)
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        filename = test_file_list[i]
        file_str = filename.split('.')[0]
        class_num = int(file_str.split('_')[0])
        test_vector = img2vector('testDigits/%s' % filename)
        classifier_result = classify0(test_vector, training_mat, hw_labels, 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_num))
        if classifier_result != class_num:
            error_count += 1
    print('total number of errors is: %d' % error_count)
    print('total error rate is: %f' % float(error_count/m_test))
















