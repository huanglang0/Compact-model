#数据预处理
import numpy as np
from sklearn.preprocessing import StandardScaler
from read_csv import read_csv
from config import (LOG10_TARGETS, LOG10_EPSILON_QUANTILE, LOG10_EPSILON_SCALE,
                    HIGH_VALUE_QUANTILE)


class SignedLog10StandardScaler:
    def __init__(self, epsilon: float, high_value_threshold: float):
        self.epsilon = float(epsilon)
        self.high_value_threshold = float(high_value_threshold)
        self.scaler = StandardScaler()
        self.is_log10 = True

    def _to_log(self, values):
        values = np.asarray(values, dtype=np.float64)
        return np.sign(values) * np.log10(1.0 + np.abs(values) / self.epsilon)

    def _from_log(self, values):
        values = np.asarray(values, dtype=np.float64)
        return np.sign(values) * self.epsilon * (np.power(10.0, np.abs(values)) - 1.0)

    def fit_transform(self, values):
        return self.scaler.fit_transform(self._to_log(values))

    def transform(self, values):
        return self.scaler.transform(self._to_log(values))

    def inverse_transform(self, values):
        log_values = self.scaler.inverse_transform(values)
        return self._from_log(log_values)

    @property
    def mean_(self):
        return self.scaler.mean_

    @property
    def scale_(self):
        return self.scaler.scale_


def _auto_log10_epsilon(values):
    abs_values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    nonzero = abs_values[abs_values > 0]
    if nonzero.size == 0:
        return 1e-30
    eps = np.quantile(nonzero, LOG10_EPSILON_QUANTILE) * LOG10_EPSILON_SCALE
    return max(float(eps), np.finfo(np.float64).tiny)


def _auto_high_value_threshold(values):
    abs_values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if abs_values.size == 0:
        return 0.0
    return float(np.quantile(abs_values, HIGH_VALUE_QUANTILE))

def load_dataset(file_list):
    all_inputs = []
    all_outputs = []
    input_head = None
    output_head = None

    for file in file_list:
        i_head, inputs, o_head, outputs = read_csv(file)
        if input_head is None:
            input_head = i_head
            output_head = o_head
        # 校验文件头一致性
        assert i_head == input_head, "输入特征头不匹配"
        assert o_head == output_head, "输出特征头不匹配"

        all_inputs.extend(inputs)
        all_outputs.extend(outputs)

    return (np.array(all_inputs, dtype=np.float64),
            np.array(all_outputs, dtype=np.float64),
            input_head,
            output_head)


def preprocess_data(X_train, y_train, X_valid, y_valid, X_test, y_test,
                   input_head, output_head, TARGETS):
    # 提取目标列
    y_train = y_train[:, [output_head.index(t) for t in TARGETS]]
    y_valid = y_valid[:, [output_head.index(t) for t in TARGETS]]
    y_test = y_test[:, [output_head.index(t) for t in TARGETS]]
    print("\nKeeping samples with zero target values...")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Valid samples: {X_valid.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def scale_data(X_train, X_valid, X_test, y_train, y_valid, y_test, TARGETS):
    # 数据标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)

    # 为每个目标创建单独的标准化器
    scalers_y = []
    y_train_scaled = np.zeros_like(y_train)
    y_valid_scaled = np.zeros_like(y_valid)
    y_test_scaled = np.zeros_like(y_test)

    for i in range(len(TARGETS)):
        if TARGETS[i] in LOG10_TARGETS:
            epsilon = _auto_log10_epsilon(y_train[:, i])
            threshold = _auto_high_value_threshold(y_train[:, i])
            scaler = SignedLog10StandardScaler(epsilon, threshold)
            print(
                f"Target {TARGETS[i]} uses signed-log10 transform "
                f"(epsilon={epsilon:.4e}, high_value_threshold={threshold:.4e})"
            )
        else:
            scaler = StandardScaler()
        y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
        y_valid_scaled[:, i] = scaler.transform(y_valid[:, i].reshape(-1, 1)).flatten()
        y_test_scaled[:, i] = scaler.transform(y_test[:, i].reshape(-1, 1)).flatten()
        scalers_y.append(scaler)
        print(f"Target {TARGETS[i]} - Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, scalers_y, scaler_X
