import numpy as np
import pandas as pd
from collections import OrderedDict
from functools import total_ordering

class FunctionCountingDecor(type):
    @staticmethod
    def call_counter(func):
        def helper(*args, **kwargs):
            helper.calls += 1
            return func(*args, **kwargs)
        helper.calls = 0
        helper.__name__= func.__name__
        return helper

    def __new__(cls, clsname, superclasses, attributedict):
        for attr in attributedict:
            if callable(attributedict[attr]) and not attr.startswith("__"):
                attributedict[attr] = cls.call_counter(attributedict[attr])
        return type.__new__(cls, clsname, superclasses, attributedict)



@total_ordering
class AnomalyDetection(metaclass=FunctionCountingDecor):

    def __init__(self,
                 input_data="",
                 data_name="JPY_Swap_outright",
                 input_unit="percentage",
                 std_threhold=4,
                 upper_bound=15,
                 lower_bound=3,
                 term_struc_neigh_span=1,
                 term_struc_neigh_tolerance_ratio=2,
                 term_struc_neigh_tolerance_decay_min=0.9,
                 term_struc_neigh_tolerance_decay_plu=1.1,
                 head_tail_tolerance=5, head_tail_ignore=True):

        self.input_data = pd.read_csv(input_data)
        self.head_tail_tolerance = head_tail_tolerance
        print(data_name + " anomaly detection and remediation in progress....")
        self.data_name = data_name
        if input_unit == "percentage":
            self.scaling = 0.01
        elif input_unit == "bps":
            self.scaling = 1
        elif input_unit == "unity":
            self.scaling = 0.0001
        else:
            print("input unit error, class instantiation failed, check again....")
            return

        self.spike_std_thres = std_threhold
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.dates = list(self.input_data.iloc[:, 0].values)
        self.no_inst = len(list(self.input_data.columns.values)) - 1
        delta_minus_1 = term_struc_neigh_tolerance_ratio
        delta_plus_1 = term_struc_neigh_tolerance_ratio
        delta_minus_2 = term_struc_neigh_tolerance_ratio * term_struc_neigh_tolerance_decay_min
        delta_plus_2 = term_struc_neigh_tolerance_ratio * term_struc_neigh_tolerance_decay_plu
        delta_minus_3 = term_struc_neigh_tolerance_ratio * term_struc_neigh_tolerance_decay_min ** 2
        delta_plus_3 = term_struc_neigh_tolerance_ratio * term_struc_neigh_tolerance_decay_plu ** 2

        if term_struc_neigh_span == 1:
            self.term_predicate = [delta_minus_1, delta_plus_1]
            self.term_pred_ix = [-1, 1]
        elif term_struc_neigh_span == 2:
            self.term_predicate = [delta_minus_2, delta_minus_1, delta_plus_1, delta_plus_2]
            self.term_pred_ix = [-2, -1, 1, 2]
        elif term_struc_neigh_span == 3:
            self.term_predicate = [delta_minus_3, delta_minus_2, delta_minus_1,
                                   delta_plus_1, delta_plus_2, delta_plus_3]
            self.term_pred_ix = [-3, -2, -1, 1, 2, 3]
        else:
            print("neighbouring filter overly wide for data set to be effective, "
                  "class instantiation failed...")
            return

        self.detection_range = [x for x in range(1 + term_struc_neigh_span, self.no_inst - term_struc_neigh_span)]
        self.std_dict = OrderedDict()
        self.anom_res_dict = OrderedDict()
        for i in range(1, self.no_inst + 1):
            self.std_dict[i] = np.std(np.diff(self.input_data.iloc[:, i]))
        if head_tail_ignore: print(
            "the front and back end of the curves are excluded from the algo but with hard coded thresholds as " + str(
                head_tail_tolerance) + " times the std move of 2nd first/last tenor bucket")
        return

    def detect_spikes(self):
        for i in self.detection_range:
            for ix, v1 in enumerate(np.diff(self.input_data.iloc[:, i]), 1):
                avg_adj_shock1 = np.mean([np.diff(self.input_data.iloc[:, i - 1])[ix - 1],
                                          np.diff(self.input_data.iloc[:, i + 1])[ix - 1]]) / self.scaling
                if abs(v1 / self.scaling) > self.upper_bound:
                    self.anom_res_dict[("belly", self.input_data.columns.values[i],
                                         self.input_data.iloc[ix, 0],
                                         self.input_data.index.values[ix])] = (v1 / self.scaling, avg_adj_shock1)
                    break
                shock_ratio_i = abs(v1 / self.std_dict[i])

                if shock_ratio_i > self.spike_std_thres and abs(v1 / self.scaling) > self.lower_bound:

                    filter_res = []
                    for x, y in zip(self.term_pred_ix, self.term_predicate):
                        temp = abs(np.diff(self.input_data.iloc[:, i + x])[ix - 1] / self.std_dict[i + x])
                        filter_res.append(shock_ratio_i / temp > y)

                    if all(filter_res):
                        self.anom_res_dict[("belly", self.input_data.columns.values[i],
                                            self.input_data.iloc[ix, 0],
                                            self.input_data.index.values[ix])] = (v1/self.scaling,
                                                                                  avg_adj_shock1)
        for i in set(range(1, self.no_inst + 1))-set(self.detection_range):
            for ix, v2 in enumerate(np.diff(self.input_data.iloc[:, i]), 1):
                avg_shoc = self.std_dict[i] * self.upper_bound / self.scaling if v2 > 0 \
                                        else -self.std_dict[i] * self.upper_bound / self.scaling

                if abs(v2 / self.scaling) > self.upper_bound:
                    self.anom_res_dict[("head_tail",
                                        self.input_data.columns.values[i],
                                        self.input_data.iloc[ix, 0],
                                        self.input_data.index.values[ix])] = (v2 / self.scaling,
                                                                              avg_shoc)
                    break

                if (abs(v2 / self.std_dict[i]) > self.head_tail_tolerance
                        and abs(v2 / self.scaling) > self.lower_bound):
                    self.anom_res_dict[("head_tail", self.input_data.columns.values[i], self.input_data.iloc[ix, 0],
                                self.input_data.index.values[ix])] = (v2 / self.scaling, avg_shoc)
            return

    def smooth_data(self):
        for i, ((section, inst, anom_date, ix), (shock, remedy)) in enumerate(self.anom_res_dict.items()):
            temp = self.input_data[inst].iloc[ix] - (shock - remedy) * self.scaling
            flag = "upwards" if shock < 0 else "downwards"
            self.input_data[inst].iloc[ix] = temp
            print("anomaly date " + anom_date + " instrument " + inst + " " + flag +
                  " revised by " + str(shock - remedy) + " bps")
        return

    def __str__(self):
        return self.data_name

    def __eq__(self, other):
        return len(self.anom_res_dict.keys()) == len(other.anom_res_dict.keys())

    def __lt__(self, other):
        return len(self.anom_res_dict.keys()) < len(other.anom_res_dict.keys())

unit_test = AnomalyDetection("D:\PythonCSV\Yen_test.csv")
unit_test.detect_spikes()
for k, v in unit_test.anom_res_dict.items():
    print(str(k)+"\n")
    print(v)
unit_test.smooth_data()
unit_test.detect_spikes()
for k, v in unit_test.anom_res_dict.items():
    print(str(k)+"\n")
    print(v)


