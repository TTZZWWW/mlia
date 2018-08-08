
import random


def load_data_set(filename):
    data_mat, label_mat = [], []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


if __name__ == '__main__':
    data_arr, label_arr = load_data_set('testSet.txt')
    print(data_arr)
    print(label_arr)







































