# -*- coding: utf-8 -*-

from sklearn.kernel_approximation import RBFSampler
import numpy as np
import numpy.matlib
#import matplotlib
#import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import sys
import time
from scipy.integrate import simps   

class MultiFidelityMaxvalueEntroySearch():
    def __init__(self, M, cost, size, beta,RegressionModel,approximation, sampling_num=10):
        self.M = M
        self.cost = cost
        self.size = size 
        self.RegModel = RegressionModel
        self.y_max = np.max(RegressionModel.y_train_[RegressionModel.X_train_[:,0]==M-1])
        self.sampling_num = sampling_num
        self.beta=beta
        self.approximation=approximation

    def Sampling_RFM(self):
        self.rbf_features = RBFSampler(gamma=1.0/(2*self.RegModel.kernel.kernel_f.get_params()['k2__length_scale']**2), n_components=100, random_state=0)
        X_train_features= self.rbf_features.fit_transform(self.RegModel.X_train_)
        A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(self.rbf_features.n_components) /self.beta)
        self.y_mean=np.mean(self.RegModel.y_train_)
        self.y_std=np.std(self.RegModel.y_train_)
        if self.y_std==0:
            print('here')
            self.y_std=1
        y=np.c_[(self.RegModel.y_train_ - self.y_mean)/self.y_std]
#        print("y",y)
        self.weights_mean = np.ravel(A_inv.dot(X_train_features.T).dot(y))
        weights_var = A_inv / self.beta 
        self.L = np.linalg.cholesky(weights_var)

    def weigh_sampling(self):
        standard_normal_rvs = np.random.normal(0, 1, (np.size(self.weights_mean), self.sampling_num))
        self.weights_sample = np.matlib.repmat(np.c_[self.weights_mean], 1, self.sampling_num) + self.L.dot(standard_normal_rvs)
        
    def f_regression(self,x):
        X_test_features = self.rbf_features.fit_transform(x[:self.size,:])#全入力(fidelity1個分)の特徴行列作成
        func_sample = X_test_features.dot(self.weights_sample) * self.y_std+ self.y_mean
        return -1*func_sample
    

    def calc_acq_TG(self,x,max_sample):
        c = np.zeros(self.M)
        max_sample[max_sample < self.y_max + 5*np.sqrt(1.0/self.beta)] = self.y_max + 5*np.sqrt(1.0/self.beta)
        max_sample = (np.c_[max_sample] + np.c_[c].T).T
        temp = np.matlib.repmat(np.c_[max_sample[0]].T, self.size, 1)
        for m in range(1, self.M):
            temp = np.r_[temp, np.matlib.repmat(np.c_[max_sample[m]].T, self.size, 1)]
        max_sample = temp
        mean,std = self.RegModel.predict(x,return_std=True)
        std[std==0]=np.sqrt(1e-5)*self.y_std
        normalized_max = (max_sample - np.c_[mean]) / np.c_[std]
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        cdf[cdf==0]=1e-30
        acq_func = (normalized_max * pdf) / (2*cdf) - np.log(cdf)
        acq_func = np.mean(acq_func, 1)

        return acq_func
    def approx_int(self,rhos_one,gammas_one,upper_lim_one,lower_lim_one):
        	denom1=1/((np.sqrt(1-rhos_one**2)))
        	denom2=1/norm.cdf(gammas_one)
        	z=np.linspace(np.min(lower_lim_one),np.max(upper_lim_one),num=100).reshape(-1,1)   
        	pdf=norm.pdf(z)*norm.cdf((gammas_one-rhos_one*z)*denom1)*denom2
        	fun=-pdf*np.log(pdf)
        	where_are_NaNs = np.isnan(fun)
        	fun[where_are_NaNs] = 0
        	integral=simps(fun.T,z.T).reshape(-1,1)
        	return np.mean(integral)  
    def calc_acq_EG(self,x,max_sample):

        max_sample[max_sample < self.y_max + 5*np.sqrt(1.0/self.beta)] = self.y_max + 5*np.sqrt(1.0/self.beta)
        mean,std = self.RegModel.predict(x[self.M-1:],return_std=True)
        std[std==0]=np.sqrt(1e-5)*self.y_std

        normalized_max = (max_sample - np.c_[mean]) / np.c_[std]
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        cdf[cdf==0]=1e-30
        high_acq_func = (normalized_max * pdf) / (2*cdf) - np.log(cdf)
        high_acq_func = np.mean(high_acq_func, 1)
        
        means,covs = self.RegModel.predict(x,return_cov=True)
        means_z,stds_z= self.RegModel.predict(x[:-1],return_std=True)
        stds_z[stds_z==0]=np.sqrt(1e-5)*self.y_std
        covs_z=covs[self.M-1:,:-1][0]
        rhos=covs_z/(stds_z*std)
        c = np.zeros(self.M)
        cdf = (np.c_[cdf[0]] + np.c_[c].T).T
        pdf = (np.c_[pdf[0]] + np.c_[c].T).T
        temp = np.matlib.repmat(np.c_[cdf[0]].T, self.size, 1)
        temp1 = np.matlib.repmat(np.c_[pdf[0]].T, self.size, 1)

        for m in range(1, self.M-1):
            temp = np.r_[temp, np.matlib.repmat(np.c_[cdf[m]].T, self.size, 1)]
            temp1 = np.r_[temp1, np.matlib.repmat(np.c_[pdf[m]].T, self.size, 1)]
        cdf = temp
        pdf=temp1
        means_eg=rhos.reshape(-1,1)*(pdf/cdf)
        stds_eg=1-np.square(rhos).reshape(-1,1)*(pdf/cdf)*(normalized_max+(pdf/cdf))
        upper_lim=means_eg+ 8*stds_eg
        lower_lim=means_eg- 8*stds_eg
        approx=np.array([[self.approx_int(rhos[j],normalized_max[0][i],upper_lim[j][i],lower_lim[j][i]) for i in range(len(normalized_max[0]))] for j in range(len(rhos))])
        approx=0.5*np.log(2*np.pi*np.e)-approx
        low_acq_func=np.mean(approx, 1)

        acq_func = np.r_[low_acq_func, high_acq_func]
        return acq_func
    def calc_acq(self,x,max_sample):
        if self.approximation=='TG':
            return self.calc_acq_TG(x,max_sample)
        else:
            return self.calc_acq_EG(x,max_sample)
    


