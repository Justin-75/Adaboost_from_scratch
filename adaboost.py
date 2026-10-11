import os
import sys
import time
import numpy as np

# AdaBoost from scratch - versi numpy
# alur sama: D = 1/m -> loop T kali -> train h_t pakai D -> predict -> eps -> alpha -> update D -> normalisasi
# bedanya: loop "for i in range(m)" diganti operasi array sekaligus (vectorized)
#   X : array (m, d)   -> m sample, d feature
#   y : array (m,)     -> +1 / -1
#   D : array (m,)     -> bobot tiap sample
#
# cara pakai (sama kayak main.py):
#   python adaboost.py <data.csv> <targets.csv> <base_id>   -> 10-fold CV, hasil ke experiments/base#_fold#.csv
#   python adaboost.py <base_id>                            -> pakai data.csv / targets.csv
#   python adaboost.py test                                 -> jalanin Task 0 - 4 (toy data)
#   base_id: 0 = logistic regression, 1 = decision stump


# ===================== Helper =====================

def sign(values):
    # >= 0 -> +1, < 0 -> -1   (np.sign(0) = 0, makanya ga dipake)
    return np.where(values >= 0, 1, -1)

def convert_label(y):
    y = np.asarray(y)
    return np.where(y <= 0, -1, 1)   # 0 -> -1, sisanya +1   (<= 0 biar label -1 ga ikut jadi +1)

def restore_label(pred, classes):
    # kebalikan convert_label: -1 -> class kecil, +1 -> class besar (misal 0 / 1)
    return np.where(pred == 1, classes[1], classes[0])

def weighted_error(D, y, pred):
    # pred != y -> array True/False, ambil D yang True terus dijumlah
    return D[pred != y].sum()

def normalize(D):
    return D / D.sum()

def accuracy(y, pred):
    return (y == pred).mean()        # True = 1, False = 0, rata2 = akurasi


# ===================== Base learner 1: Decision Stump =====================
# hypothesis: x[feature] <= threshold -> direction, else -> -direction
# train: coba SEMUA feature, threshold, direction -> ambil weighted error terkecil

def stump_predict(stump, X):
    column = X[:, stump['feature']]  # ambil 1 kolom
    return np.where(column <= stump['threshold'], stump['direction'], -stump['direction'])

def stump_train(X, y, D):
    m, d = X.shape
    best_stump = None
    best_error = np.inf

    for feature in range(d):
        values = np.unique(X[:, feature])                          # udah sorted otomatis
        thresholds = np.concatenate([[values[0] - 1], (values[:-1] + values[1:]) / 2])  # titik tengah

        for threshold in thresholds:
            for direction in [1, -1]:
                stump = {'feature': feature, 'threshold': float(threshold), 'direction': direction}
                error = weighted_error(D, y, stump_predict(stump, X))
                if error < best_error:
                    best_error = error
                    best_stump = stump

    return best_stump


# ===================== Base learner 2: Weighted Logistic Regression =====================
# hypothesis: p = sigmoid(X @ w + b), predict +1 kalau p >= 0.5
# cost: J = - sum D * [ y01 * ln(p) + (1 - y01) * ln(1 - p) ]
# gradient: dJ/dw = X.T @ (D * (p - y01)),   dJ/db = sum D * (p - y01)
# lr (learning rate) punya LR, BUKAN punya AdaBoost

def sigmoid(z):
    z = np.clip(z, -500, 500)        # biar exp ga overflow kalau z gede banget
    return 1 / (1 + np.exp(-z))

def lr_prob(model, X):
    return sigmoid(X @ model['w'] + model['b'])   # @ = perkalian matrix, hasilnya (m,)

def lr_predict(model, X):
    return np.where(lr_prob(model, X) >= 0.5, 1, -1)

def lr_cost(model, X, y, D):
    p = np.clip(lr_prob(model, X), 1e-12, 1 - 1e-12)   # biar ga ln(0)
    y01 = (y + 1) / 2
    return -(D * (y01 * np.log(p) + (1 - y01) * np.log(1 - p))).sum()

def lr_gradient(model, X, y, D):
    y01 = (y + 1) / 2                  # +1/-1 -> 1/0 khusus di dalem LR
    diff = D * (lr_prob(model, X) - y01)   # (m,)
    grad_w = X.T @ diff                # (d,)  = sum diff[i] * X[i] buat semua i sekaligus
    grad_b = diff.sum()
    return grad_w, grad_b

def lr_train(X, y, D, lr=1.0, epochs=2000):
    model = {'w': np.zeros(X.shape[1]), 'b': 0.0}
    for epoch in range(epochs):
        grad_w, grad_b = lr_gradient(model, X, y, D)
        model['w'] = model['w'] - lr * grad_w
        model['b'] = model['b'] - lr * grad_b
    return model

def check_gradient(X, y, D):
    # Task 3: gradient rumus vs gradient numerik
    model = {'w': np.array([0.3, -0.2]), 'b': 0.1}
    grad_w, grad_b = lr_gradient(model, X, y, D)
    analytic = np.append(grad_w, grad_b)

    h = 1e-5
    numeric = []
    for j in range(len(model['w'])):
        plus = {'w': model['w'].copy(), 'b': model['b']}
        minus = {'w': model['w'].copy(), 'b': model['b']}
        plus['w'][j] += h
        minus['w'][j] -= h
        numeric.append((lr_cost(plus, X, y, D) - lr_cost(minus, X, y, D)) / (2 * h))
    plus = {'w': model['w'], 'b': model['b'] + h}
    minus = {'w': model['w'], 'b': model['b'] - h}
    numeric.append((lr_cost(plus, X, y, D) - lr_cost(minus, X, y, D)) / (2 * h))
    numeric = np.array(numeric)

    max_diff = (np.abs(analytic - numeric) / np.maximum(np.abs(analytic) + np.abs(numeric), 1e-12)).max()
    return analytic, numeric, max_diff


# ===================== Base learner 3: Line search (2 feature) =====================
# bentuk sama kayak LR (garis lurus), tapi train nya langsung cari weighted ERROR terkecil

def line_predict(model, X):
    return np.where(X @ model['w'] + model['b'] >= 0, 1, -1)

def line_train(X, y, D, n_angle=72):
    best_model = None
    best_error = np.inf
    for angle in np.linspace(0, 2 * np.pi, n_angle, endpoint=False):
        w = np.array([np.cos(angle), np.sin(angle)])
        proj = np.unique(X @ w)                                    # posisi titik di arah ini
        cuts = np.concatenate([[proj[0] - 1], (proj[:-1] + proj[1:]) / 2])
        for c in cuts:
            model = {'w': w, 'b': -c}
            error = weighted_error(D, y, line_predict(model, X))
            if error < best_error:
                best_error = error
                best_model = model
    return best_model


# ===================== AdaBoost =====================
# base learner dikasih sebagai function (higher order function)

def adaboost_train(X, y, T, train_func, predict_func, verbose=True):
    m = len(y)
    D = np.full(m, 1 / m)          # line 1
    models = []                    # simpen MODEL nya, bukan prediksinya
    alphas = []

    for t in range(1, T + 1):                       # line 2
        h = train_func(X, y, D)                     # line 3: train pakai D sekarang
        pred = predict_func(h, X)
        eps = weighted_error(D, y, pred)            # line 4

        # line 5: pakai >= 0.5 (eps = 0.5 -> alpha = 0 -> D ga berubah -> muter2 ga ada progress)
        if eps >= 0.5:
            if verbose:
                print(f"round {t}: eps = {eps:.3f} >= 0.5, ga lebih bagus dari tebak koin, stop")
            break
        eps = max(eps, 1e-10)                       # jaga2 eps = 0

        alpha = 0.5 * np.log((1 - eps) / eps)       # line 6: kekuatan vote, BUKAN learning rate

        # line 7-13: SATU baris buat semua sample sekaligus (ga perlu loop i lagi)
        # benar: y*pred = +1 -> * e^(-alpha) | salah: y*pred = -1 -> * e^(+alpha)
        D = D * np.exp(-alpha * y * pred)
        D = normalize(D)                            # line 14-15

        models.append(h)
        alphas.append(alpha)

        if verbose:
            acc = accuracy(y, adaboost_predict(models, alphas, predict_func, X))
            print(f"round {t}: eps = {eps:.3f}  alpha = {alpha:.3f}  "
                  f"D = {np.round(D, 3)}  train acc = {acc:.2f}")

    return models, alphas

def adaboost_score(models, alphas, predict_func, X):
    # H(x) = sum alpha_t * h_t(x)
    scores = np.zeros(len(X))
    for h, alpha in zip(models, alphas):
        scores += alpha * predict_func(h, X)
    return scores

def adaboost_predict(models, alphas, predict_func, X):
    return sign(adaboost_score(models, alphas, predict_func, X))   # sign SEKALI di akhir


# ===================== Eksperimen (data asli) =====================
# 10-fold cross validation, T = 1 / 5 / 10 / 100 -> experiments/base{T}_fold{k}.csv
# pembagian fold & format file sama persis kayak main.py, jadi bisa dibandingin pakai evaluate.py

T_LIST = [1, 5, 10, 100]

BASE_LEARNERS = {
    0: (lr_train, lr_predict),         # 0 = logistic regression
    1: (stump_train, stump_predict),   # 1 = decision stump
}

def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0             # kolom konstan, biar ga bagi 0
    return mean, std

def run_experiments(data_path='data.csv', target_path='targets.csv', base_id=1, output_dir='experiments'):
    start = time.time()
    X = np.genfromtxt(data_path, delimiter=',')
    y_raw = np.genfromtxt(target_path, delimiter=',')
    if y_raw.ndim > 1:
        y_raw = y_raw[:, -1]           # kalau kolomnya banyak, label = kolom terakhir
    print(f"base_id = {base_id}, X = {X.shape}, y = {y_raw.shape}", flush=True)

    classes = np.unique(y_raw)
    y = convert_label(y_raw)
    if len(classes) != 2 or set(y.tolist()) != {-1, 1}:
        raise ValueError(f"label harus 2 class (0/1 atau -1/+1), dapetnya {classes}")

    # folder experiments udah disiapin sama penilai, JANGAN dibikin sendiri
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"folder '{output_dir}' ga ada, bikin dulu sebelum run")

    train_func, predict_func = BASE_LEARNERS[base_id]
    m = len(y)
    folds = np.array_split(np.arange(m), 10)

    for k, test_idx in enumerate(folds, start=1):
        t0 = time.time()
        train_idx = np.setdiff1d(np.arange(m), test_idx)

        # mean / std dihitung dari train fold AJA, biar info test ga bocor ke training
        # (wajib buat LR: tanpa ini exp() overflow & gradient descent ga konvergen)
        mean, std = standardize_fit(X[train_idx])
        X_train = (X[train_idx] - mean) / std
        X_test = (X[test_idx] - mean) / std

        # AdaBoost itu greedy: round 1..T ga terpengaruh round sesudahnya,
        # jadi cukup train SEKALI sampai T terbesar, terus ambil models[:T] buat tiap T
        models, alphas = adaboost_train(X_train, y[train_idx], max(T_LIST),
                                        train_func, predict_func, verbose=False)

        for T in T_LIST:
            pred = adaboost_predict(models[:T], alphas[:T], predict_func, X_test)
            # kolom 1: nomor sample (mulai dari 1), kolom 2: label prediksi (format asli)
            out = np.column_stack([test_idx + 1, restore_label(pred, classes)]).astype(int)
            np.savetxt(os.path.join(output_dir, f'base{T}_fold{k}.csv'), out, fmt='%d', delimiter=',')

        print(f"fold {k:2d}/10 selesai: {len(models)} model, {time.time() - t0:.1f}s", flush=True)

    print(f"total {time.time() - start:.1f}s", flush=True)


# ===================== Testing (toy data) =====================

def run_tests():
    # Task 0 + 1
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = convert_label([1, 1, 0, 0, 1])
    print("Task 0: label =", y, " valid:", set(y.tolist()) == {-1, 1})

    D = np.full(len(y), 1 / len(y))
    stump = stump_train(X, y, D)
    print("Task 1: stump =", stump, " eps =", round(weighted_error(D, y, stump_predict(stump, X)), 3))
    print()

    # Task 2
    print("Task 2: AdaBoost + stump, T = 3")
    models, alphas = adaboost_train(X, y, 3, stump_train, stump_predict)
    print("alphas =", np.round(alphas, 3))
    print("final predict =", adaboost_predict(models, alphas, stump_predict, X), " true =", y)
    print()

    # Task 3
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([-1, 1, 1, -1])
    analytic, numeric, max_diff = check_gradient(X_xor, y_xor, np.array([0.1, 0.2, 0.3, 0.4]))
    print("Task 3: analytic =", np.round(analytic, 6))
    print("        numeric  =", np.round(numeric, 6))
    print("        max relative diff =", max_diff, " pass:", max_diff < 1e-6)
    print()

    # Task 4
    print("Task 4a: XOR + stump")
    adaboost_train(X_xor, y_xor, 10, stump_train, stump_predict)
    print()

    print("Task 4b: XOR + weighted LR")
    adaboost_train(X_xor, y_xor, 10, lr_train, lr_predict)
    print("LR gradient di w=0, b=0 :", lr_gradient({'w': np.zeros(2), 'b': 0.0}, X_xor, y_xor, np.full(4, 0.25)))
    print()

    print("Task 4c: XOR + line search")
    models, alphas = adaboost_train(X_xor, y_xor, 6, line_train, line_predict)
    print("final predict =", adaboost_predict(models, alphas, line_predict, X_xor), " true =", y_xor)


def main():
    args = sys.argv[1:]
    if args == ['test']:
        run_tests()
        return

    data_path, target_path, base_id = 'data.csv', 'targets.csv', '1'
    if len(args) == 1:
        base_id = args[0]
    elif len(args) == 3:
        data_path, target_path, base_id = args

    if len(args) not in (0, 1, 3) or base_id not in ('0', '1'):
        print("cara pakai: python adaboost.py <data.csv> <targets.csv> <base_id>   (base_id: 0 = LR, 1 = stump)\n"
              "            python adaboost.py <base_id>\n"
              "            python adaboost.py test")
        sys.exit(1)

    run_experiments(data_path, target_path, int(base_id))


if __name__ == "__main__":
    main()
