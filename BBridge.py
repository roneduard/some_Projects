import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt


class EmpiricalData:
    def __init__(self, pricedatainput, User_Path, GARCH_return,
                 Path_no = 100, GARCH_flag = 1, temporal_scaling = 1,
                 freq_scaling = 1.1, p_user = 1, o_user = 0, q_user = 1,
                 distri ='Normal', BB_TimeStep = 20, VaR_Cutoff = 95):
        self._nameofclass = 'price set in GARCH'
        self._path = Path_no
        self._scaling = freq_scaling
        self._VaR_Cutoff = VaR_Cutoff
        self._steps = BB_TimeStep
        self._temporal_scaling = temporal_scaling
        self._User_Path = User_Path
        self._GARCH = GARCH_flag
        self._GARCH_return = GARCH_return
        self._p = p_user
        self._o = o_user
        self._q = q_user
        self._dist = distri
        self._pricedatainput = pricedatainput
        self._header = [column for column in pricedatainput]
        self._dflength = len(self._header)
        self._abs_return = {}
        self._abs_return_std = {}
        self._abs_return_normalized = {}
        self._GARCH_models = {}
        self._final_Total_result = {}
        self._final_Total_result_shocks = {}
        self._final_Total_percentiled_up_shocks = {}
        self._final_up_shocks = {}
        self._final_Total_percentiled_down_shocks = {}
        self._final_down_shocks = {}
        self._shock_average = {}
        self._shock_average_final = {}
        self._sigma_ratio = []
        self._GARCH_path = {}
        self._GARCH_models_alphabetasum = {}
        for n in range(self._dflength):
            self._abs_return[n] = self._pricedatainput.ix[:, n].diff().dropna()
            price_return = self._abs_return[n]

            self._sigma_ratio.append(np.sqrt(self._steps) / np.sqrt(3) * np.std(price_return) / ((np.std(
                price_return[0: -1: 3]) + np.std(price_return[1: -1: 3]) + np.std(price_return[2: -1: 3])) / 3))

            if self._GARCH == 1:
                print(self._GARCH_return.ix[:, n].values)
                model = arch_model(self._GARCH_return.ix[:, n].values, p=self._p, o=self._o, q=self._q, dist=self._dist)
                self._GARCH_models[n] = model.fit()
        print(self._abs_return)
        for n in range(self._dflength):
            print(self._GARCH_models[n].summary())
            print(self._GARCH_models[n].params['beta[1]'], self._GARCH_models[n].params['alpha[1]'])

    def __str__(self):
        return self._nameofclass + self._p + self._o + self._q + self._dist

    def Bridging_Sim(self, Param_0, Param_1, Sigma_Prime):
        Timestep = self._steps

        dt = 1.0 / self._steps
        dt_sqrt = np.sqrt(dt)
        Result_Raw = np.empty((self._path, self._steps + 1), dtype=np.float32)
        Result_Raw[:, 0] = Param_0
        Result_Raw[:, Timestep] = Param_1
        for n in range(self._steps): dZ = np.random.normal(0, Sigma_Prime, self._path) * dt_sqrt
        Result_Raw[:, n + 1] = Result_Raw[:, n] * (1 - dt / (1 - n * dt)) + Result_Raw[:, self._steps] * dt / (
                    1 - n * dt) + dZ
        return Result_Raw

    def Serial_bridging(self, monthly_price, Sigma_x, tenor):
        N = len(monthly_price) - 1
        Plus_Steps = self._steps + 1
        Total_result = np.empty((self._path, N * Plus_Steps + 1), dtype=np.float32)
        Sigma_monthly = np.empty((self._path, N + 1), dtype=np.float32)
        Shock_average = np.empty((N * Plus_Steps), dtype=np.float32)
        Shock_sum_up = 0.0
        Shock_sum_down = 0.0
        w_0 = self._GARCH_models[tenor].params['omega']
        alpha_0 = self._GARCH_models[tenor].params['alpha[1]']
        beta_0 = self._GARCH_models[tenor].params['beta[1]']
        self._GARCH_models_alphabetasum[tenor] = alpha_0 + beta_0
        Sigma_monthly[:, 0] = np.sqrt(w_0 + beta_0 * (Sigma_x ** 2.0))
        for i in range(N):
            Total_result[:, (Plus_Steps * i): (Plus_Steps * (i + 1))] = self.Bridging_Sim(
                monthly_price[i], monthly_price[i + 1], Sigma_monthly[:, i])
            residual = Total_result[:, (Plus_Steps * (i + 1))] - Total_result[:, (Plus_Steps * i)]
            Sigma_monthly[:, i + 1] = np.sqrt(w_0 + alpha_0 * (residual ** 2.0) + beta_0 * (Sigma_monthly[:, i] ** 2.0))
            # plt.plot(Total_result.T)
            # plt.show()
            Total_result[:, N * Plus_Steps] = monthly_price[-1]
            Total_result_shock = np.diff(Total_result)

            for j in range(self._path):
                Shock_sum_up = Shock_sum_up + np.percentile(Total_result_shock[j, :], self._VaR_Cutoff)
                Shock_sum_down = Shock_sum_down + np.percentile(Total_result_shock[j, :], 100 - self._VaR_Cutoff)
            for k in range(N * Plus_Steps):
                Shock_average[k] = np.sum(Total_result_shock[:, k]) / self._path
            return (Total_result, Total_result_shock, Shock_sum_up /
                    self._path, Shock_sum_down / self._path, Shock_average, Sigma_monthly[0, :])

    def Tenor_looping(self, loop_name='dummy', partial_tenor=None):
        if partial_tenor == None:
            for n in range(self._dflength):
                if self._temporal_scaling == 1:
                    self._final_Total_result[n], self._final_Total_result_shocks[n], \
                    self._final_Total_percentiled_up_shocks[n], self._final_Total_percentiled_down_shocks[n], self._shock_average[n], \
                    self._GARCH_path[n] = self.Serial_bridging(self._pricedatainput.ix[:, n].values,
                                               self._sigma_ratio[n] * np.std(self._abs_return[n].values), n)
                else:
                    self._final_Total_result[n], self._final_Total_result_shocks[n], \
                    self._final_Total_percentiled_up_shocks[n], self._final_Total_percentiled_down_shocks[n], \
                    self._shock_average[n], _ = self.Serial_bridging(self._pricedatainput.ix[:, n].values,
                                                                     self._scaling * np.std(self._abs_return[n].values),
                                                                     n)

                    # pd.DataFrame(self._final_Total_result[n]).to_csv(self._User_Path + '\resultpath' + loop_name + str(n) + 'y.csv')
                    # pd.DataFrame(self._final_Total_result_shocks[n]).to_csv(self._User_Path + '\resultpath' + loop_name + str(n) + 'yShocks.csv')
                    self._shock_average_final[n] = self._shock_average[n]
                    # self._final_up_shocks[n] = self._final_Total_percentiled_up_shocks[n]
                    # self._final_down_shocks[n] = self._final_Total_percentiled_down_shocks[n]
        else:
            for n in partial_tenor:
                if self._temporal_scaling == 1:
                    (self._final_Total_result[n - 1],
                     self._final_Total_result_shocks[n - 1],
                     self._final_Total_percentiled_up_shocks[n - 1],
                     self._final_Total_percentiled_down_shocks[n - 1],
                     self._shock_average[n - 1],
                     self._GARCH_path[n]) = (
                        self.Serial_bridging(self._pricedatainput.ix[:, n - 1].values,
                                             self._sigma_ratio[n - 1] * np.std(self._abs_return[n - 1].values), n))

                else:
                    self._final_Total_result[n - 1], self._final_Total_result_shocks[n - 1], \
                    self._final_Total_percentiled_up_shocks[n - 1], self._final_Total_percentiled_down_shocks[n - 1], \
                    self._shock_average[n - 1], _ = self.Serial_bridging(self._pricedatainput.ix[:, n - 1].values,
                                                                         self._scaling * np.std(
                                                                             self._abs_return[n - 1].values), n)
                    # pd.DataFrame(self._final_Total_result[n - 1]).to_csv(self._User_Path + '\resultpath' + loop_name + str(n) + 'y.csv')
                    # pd.DataFrame(self._final_Total_result_shocks[n - 1]).to_csv(self._User_Path + '\resultpath' + loop_name + str(n) + 'yShocks.csv')
                    self._shock_average_final[n] = self._shock_average[n - 1]
                    self._final_up_shocks[n] = self._final_Total_percentiled_up_shocks[n - 1]
                    self._final_down_shocks[n] = self._final_Total_percentiled_down_shocks[n - 1]

            print(self._GARCH_path)
            pd.DataFrame(self._GARCH_path).to_csv(
                self._User_Path + '\\resultpath' + loop_name + 'GARCHpath.csv')
            print(self._shock_average_final)
            pd.DataFrame(self._shock_average_final).to_csv(
                self._User_Path + '\\resultpath' + loop_name + 'averageShocks.csv')
            print(
                self._GARCH_models_alphabetasum)
            # print(self._final_up_shocks)
            # pd.DataFrame([self._final_up_shocks], columns =
            # self._final_up_shocks.keys()).to_csv(self._User_Path + '\finalUP' + loop_name + 'yShocks.csv')
            # print(self._final_down_shocks)
            # pd.DataFrame([self._final_down_shocks], columns = self._final_down_shocks.keys()).to_csv(self._User_Path + '\finalDOWN' + loop_name + 'yShocks.csv')
            return self._final_Total_result

User_Path_Folder = 'D:\\'
User_Result_Folder = 'D:\\'
# price_hisc_df03 = pd.read_csv(User_Path_Folder + '\BBgarchTest03.csv', index_col = 'Dates', parse_dates = True).sort_index()
returns_garch_df03 = pd.read_csv(User_Path_Folder + '\BBgarchTest03Return.csv',
                                 index_col = 'Dates', parse_dates = True).sort_index()
inflation_0_3_totem_2006_2011 = EmpiricalData(returns_garch_df03, User_Result_Folder, returns_garch_df03, 5000)
result03 = inflation_0_3_totem_2006_2011.Tenor_looping('zerothreeLPI')
# price_hisc_df05 = pd.read_csv(User_Path_Folder + '\BBgarchTest05.csv', index_col = 'Dates', parse_dates = True).sort_index()
returns_garch_df05 = pd.read_csv(User_Path_Folder + '\BBgarchTest05Return.csv', index_col = 'Dates', parse_dates = True)


























