import random
from numpy import *


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


def smo_simple(data_mat, class_labels, c, toler, max_iter):
    data_matrix = mat(data_mat)
    label_matrix = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    b, iter = 0, 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            fx_i = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b
            e_i = fx_i - float(label_matrix[i])
            if ((label_matrix[i] * e_i < -toler) and (alphas[i] < c)) or (
                    (label_matrix[i] * e_i > toler) and (alphas[i] > 0)):
                j = select_j_rand(i, m)
                fx_j = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                e_j = fx_j - float(label_matrix[j])
                alpha_i_old, alpha_j_old = alphas[i].copy(), alphas[j].copy()
                if label_matrix[i] != label_matrix[j]:
                    ll, hh = max(0, alphas[j] - alphas[i]), min(c, c + alphas[j] - alphas[i])
                else:
                    ll, hh = max(0, alphas[j] + alphas[i] - c), min(c, alphas[j] + alphas[i])
                if ll == hh:
                    print('ll==hh')
                    continue
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                      data_matrix[i, :] * data_matrix[i, :].T - \
                      data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= label_matrix[j] * (e_i - e_j) / eta
                alphas[j] = clip_alpha(alphas[j], hh, ll)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_j_old - alphas[j])
                b1 = b - e_i - \
                     label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - e_j - \
                     label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('iter: %d, i: %d, pairs changed: %d' % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: ', iter)
    return b, alphas


if __name__ == '__main__':
    data_arr, label_arr = load_data_set('testSet.txt')
    print(data_arr)
    print(label_arr)

    b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
    print(alphas)
    print(alphas[alphas > 0])
    print(b)
