
import torch
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import truncnorm
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
from model.helper import *
ro.numpy2ri.activate()
#GPfit = importr('BayesGPfit')
pgdraw = importr('pgdraw')
pgdraw_f = rpy2.robjects.r['pgdraw']
truncnorm_r = importr("truncnorm")
r_truncnorm = ro.r["rtruncnorm"]

import os
import pandas as pd

def get_basis(V, d, poly_degree, a, b, grids=None):
    if grids is None:
        grids = GPfit.GP_generate_grids(d=d, num_grids=V, grids_lim=np.array((-1, 1)))
    # grids = grids.numpy()
    lamb = GPfit.GP_eigen_value(poly_degree=poly_degree, a=a, b=b, d=d)

    if os.path.exists(f"basis_a{a}_b{b}_deg{poly_degree}.dat"):
        Xmat = pd.read_csv(f"basis_a{a}_b{b}_deg{poly_degree}.dat", delimiter=' ', header=None)
        Xmat = Xmat.values
    else:
        Xmat = GPfit.GP_eigen_funcs_fast(grids=grids, poly_degree=poly_degree, a=a, b=b)
    # Xmat = np.loadtxt(f"data/rtgp_basis_a{a}_b{b}_deg{poly_degree}_T{V}.dat", delimiter=' ')
    # lamb = np.loadtxt(f"data/rtgp_lambda_a{a}_b{b}_deg{poly_degree}_T{V}.dat", delimiter=' ')

    Xmat = torch.from_numpy(Xmat).float()
    Xmat, _ = torch.linalg.qr(Xmat)
    lamb = torch.from_numpy(lamb).float().squeeze(-1)

    return Xmat, lamb

class SIRTGP_logit():
    def __init__(self, Y, X, X0,
                poly_degree=10, a=0.1, b=1,
                init_e = None, 
                init_beta0 = None, 
                init_beta_hat = None, 
                init_eta = None,
                init_eta_hat = None, 
                init_w1 = None,
                init_w2 = None,
                init_sigma2_eta = None,
                init_omega = None,
                xisq = torch.ones(1)*1,
                grid_step=10, 
                first_burnin=100, second_burnin = 100, 
                thin=1, mcmc_sample=100):
        
        self.y = Y # binary out come, N
        self.X = X # EEG data, T by K by N
        self.X0 = X0 # EEG interaction data, N by V
        self.N = X.shape[0] # num of images
        self.K = X.shape[1] # num of channels
        self.T = X.shape[2] # num of timepoints

        #grids = GPfit.GP_generate_grids(d = 1, num_grids = self.T, grids_lim = torch.array((-1,1)))
        self.B, self.lamb = get_basis(V=int(self.T), d=1, poly_degree=poly_degree, a=a, b=b)
        self.B_lamb = self.B @ torch.diag(self.lamb.sqrt())
        
        self.L = self.B.shape[1]

        self.ns1 = self.T * self.K
        self.ns2 = int(self.K * (self.K - 1) / 2)
        self.V = int(self.K * (self.K - 1) / 2)
        # self.ns2=1
        # self.ns1=1

        if self.X0.shape[0] == self.V:
            self.X0 = self.X0.T


        self.one_num = int(self.y.sum())
        self.zero_num = int((self.N - self.one_num))
        self.y_one = (self.y == 1)
        self.y_zero = (self.y == 0)


        self.a_sig_eta = 0.0001
        self.b_sig_eta = 0.0001

        self.M = grid_step
        self.w_log_prior = torch.log(torch.ones(self.M) / self.M)
        self.w_log_post = torch.zeros(self.M)

        ### mcmc settings
        self.first_burnin = first_burnin
        self.second_burnin = second_burnin
        self.mcmc_burnin = first_burnin + second_burnin 
        self.mcmc_thinning = thin
        self.mcmc_sample = mcmc_sample 
        self.total_iter = self.mcmc_burnin + self.mcmc_sample * self.mcmc_thinning

        if self.second_burnin > 0:
            # self.xisq_vec = torch.linspace(xisq, 0.01, second_burnin)
            # self.xisq = self.xisq_vec[0]
            self.xisq_vec = torch.linspace(xisq.item(), 0.0001, steps=second_burnin // 50)
            self.xisq_vec = torch.repeat_interleave(self.xisq_vec, repeats=50)
            self.xisq = self.xisq_vec[0]
        else:
            self.xisq = xisq

        #self.xisq = 1
        self.sigma2_e = 10000


        #initialization
        if init_e is None:
            init_e = torch.randn(self.L, self.K)

        if init_beta_hat is None:
            init_beta_hat = torch.randn(self.T, self.K)

        if init_eta is None:
            init_eta = torch.randn(self.V)

        if init_eta_hat is None:
            init_eta_hat = torch.randn(self.V)

        if init_w1 is None:
            init_w1 = torch.zeros(1)

        if init_w2 is None:
            init_w2 = torch.zeros(1)
        
        if init_sigma2_eta is None:
            init_sigma2_eta = torch.ones(1)
        
        if init_beta0 is None:
            init_beta0 = torch.zeros(1)


        self.e = init_e
        self.beta = self.B @ self.e
        self.beta_hat = init_beta_hat
        self.eta = init_eta
        self.eta_hat = init_eta_hat
        self.w1 = init_w1
        self.w2 = init_w2
        self.sigma2_eta = init_sigma2_eta
        self.beta0 = init_beta0
        self.sigma2_beta0 = torch.ones(1)


        self.thresh_beta = self.RTGP_thresh(self.beta_hat, self.w1)
        self.thresh_eta = self.RTGP_thresh(self.eta_hat, self.w2)

        self.SR = self.update_SR()
        self.S0 = self.update_S0()

        if init_omega is None:
            init_omega = torch.randn(self.N)
        self.omega = init_omega
        
        self.loglik_y = torch.zeros(self.total_iter)
        self.accept_rate_vec = torch.zeros(self.total_iter)

        self.make_mcmc_samples()
           

    def RTGP_thresh(self, x, nu):
        return (torch.abs(x) > nu) * 1
    
    def update_SR(self):
        SR = torch.einsum('tk,nkt->n', self.beta * self.thresh_beta, self.X) / self.ns1
        return SR

    def update_S0(self):
        S0 = self.X0 @ (self.eta * self.thresh_eta)  / self.ns2
        return S0
    
    def fit_si(self, mute=False):
        for i in tqdm(range(self.total_iter), disable=mute):
            if (i >= self.first_burnin) & (i < self.mcmc_burnin):
                self.xisq = self.xisq_vec[i-self.first_burnin]
            #print(self.xisq)
            self.update_omega()
            self.update_e()
            self.update_beta_hat()
            self.update_eta()
            self.update_eta_hat()
            #self.update_xisq()
            self.update_sigma2_e()
            #if (i >= self.first_burnin//2):  
            self.update_w1()
            self.update_w2()
            self.update_sigma2_eta()
            self.update_beta0()
            self.update_sigma2_beta0()
            self.loglik_y[i] = self.update_loglik_y()
            if i >= self.mcmc_burnin:
                    if (i - self.mcmc_burnin) % self.mcmc_thinning == 0:
                        mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                        self.save_mcmc_samples(mcmc_iter)

    def fit_s(self, mute=False):
        self.w2=0
        self.eta = torch.zeros(self.V)
        self.thresh_eta = torch.zeros(self.V)
        self.update_S0()
        for i in tqdm(range(self.total_iter), disable=mute):
            self.update_omega()
            self.update_e()
            self.update_beta_hat()
            self.update_beta0()
            self.update_sigma2_beta0()
            # self.update_xisq()
            # self.update_sigma2_e()
            #if ((i >= self.first_burnin//2)):
            self.update_w1()
            self.loglik_y[i] = self.update_loglik_y()
            if i >= self.mcmc_burnin:
                    if (i - self.mcmc_burnin) % self.mcmc_thinning == 0:
                        mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                        self.save_mcmc_samples(mcmc_iter)

    def update_omega(self):
        zz = self.SR + self.S0 + self.beta0
        #self.omega = torch.from_numpy(pgdraw_f(1, ro.FloatVector(zz))).double()
        self.omega = torch.from_numpy(pgdraw_f(1, ro.FloatVector(zz.numpy()))).float()

    def update_e(self):
        temp = self.SR + self.S0 + self.beta0
        self.B_thresh_X = torch.einsum('tl,nkt->nlk', self.B_lamb, self.thresh_beta.t()[None,:,:] * self.X) / self.ns1 #Ci
        beta_res = self.beta_hat - self.beta
        for k in range(self.K): 
            for l in range(self.L):
                beta_res[:,k] += self.B_lamb[:,l] * self.e[l,k]
                temp -= self.B_thresh_X[:,l,k] * self.e[l,k] #Si
                sig2 = 1 / (torch.sum((self.B_thresh_X[:,l,k] ** 2) * self.omega) + 1/(self.sigma2_e) + torch.sum((self.B_lamb[:,l] ** 2))/self.xisq)
                mu = sig2 * (torch.sum((self.y - 0.5 -  temp * self.omega)*self.B_thresh_X[:,l,k]) + torch.sum(self.B_lamb[:,l] * beta_res[:,k] /self.xisq))
                #print(mu_alpha)
                self.e[l,k] = torch.randn(1) * sig2.sqrt() + mu
                beta_res[:,k] -= self.B_lamb[:,l] * self.e[l,k]
                temp += self.B_thresh_X[:,l,k] * self.e[l,k] 
        self.beta = self.B_lamb @ self.e
        self.SR = self.update_SR()

    def update_beta_hat(self):
        temp = self.SR + self.S0 + self.beta0
        
        #self.scale = torch.sqrt(self.tau2)
        xi = self.xisq.sqrt().item()
        thresh = self.w1.item()
        if self.w1 > 0:
            lower_vec = [-np.inf, - thresh, thresh]
            upper_vec = [-thresh, thresh, np.inf]
            p_mp_mat = torch.special.log_ndtr( -(self.w1 + self.beta) / xi) 
            p_m_mat = torch.special.log_ndtr((self.w1 - self.beta) / xi) 
            p_mm_mat = torch.special.log_ndtr( -(self.w1 - self.beta) / xi) 

            for k in range(self.K):
                for t in range(self.T):
                    logc_vec = torch.zeros(3)
                    mu = self.beta[t,k].item()

                    temp -= (self.beta[t,k] * self.thresh_beta[t,k] * self.X[:,k,t]) / self.ns1
                    c_sum = torch.sum( -0.5 * self.omega * ((temp + self.beta[t,k] * self.X[:,k,t] / self.ns1) ** 2)+ (self.y - 0.5) * (temp + self.beta[t,k] * self.X[:,k,t] / self.ns1))
                    p_mp = p_mp_mat[t,k] 
                    p_m = p_m_mat[t,k] 
                    p_mm = p_mm_mat[t,k]
                    # p_mp = torch.from_numpy(pnorm_f( (-(self.w1 + self.beta[t,k]) / xi).numpy().reshape(1), 0, 1, True,True)).float()
                    # p_m = torch.from_numpy(pnorm_f( ((self.w1 - self.beta[t,k]) / xi).numpy().reshape(1), 0, 1, True,True)).float()
                    # p_mm = torch.from_numpy(pnorm_f(( -(self.w1 - self.beta[t,k]) / xi).numpy().reshape(1), 0, 1, True,True)).float()


                    logc_vec[0] = c_sum + p_mp #C-
                    #logc_vec[1] = - 0.5 * torch.sum(temp ** 2)  + log1pexp(p_mp - p_m) + p_m #C0
                    #logc_vec[1] = torch.sum( -0.5 * self.omega * (temp ** 2) + (self.y - 0.5) * temp) + p_m+torch.log(torch.exp(p_m-p_mp)-1)
                    logc_vec[1] = torch.sum( -0.5 * self.omega * (temp ** 2) + (self.y - 0.5) * temp)+ p_m + torch.log1p(-torch.exp(p_mp-p_m) + 1e-10)
                    logc_vec[2] = c_sum + p_mm #C1

                    logc_vec_max = max(logc_vec)
                    c_vec = torch.exp(logc_vec - logc_vec_max)
                    w_vec = c_vec/(torch.sum(c_vec))
                    c_ind = torch.multinomial(w_vec, num_samples=1).item()

         
                    lower = lower_vec[c_ind]
                    upper = upper_vec[c_ind]
                    
                    #self.a, self.b = (lower - mu) / xi, (upper - mu) / xi
                    #self.beta_hat[t,k] = truncnorm.rvs(self.a,self.b,loc=mu,scale=xi, size=1)
                    self.beta_hat[t,k] =  torch.from_numpy(r_truncnorm(1, mean=mu, sd=xi, a=lower, b=upper)).float()
                    self.thresh_beta[t,k] = self.RTGP_thresh(self.beta_hat[t,k], self.w1)
                    temp += (mu * self.thresh_beta[t,k] * self.X[:,k,t]) / self.ns1
        else:
            logc_vec = torch.zeros(2)
            lower_vec = [-np.inf, thresh]
            upper_vec = [thresh, np.inf]
            
            p_mp_mat = torch.special.log_ndtr( -(self.w1 + self.beta) / xi) 
            p_mm_mat = torch.special.log_ndtr( -(self.w1 - self.beta) / xi) 

            for k in range(self.K):
                for t in range(self.T):
        
                    mu = self.beta[t,k].item()
                    temp -= (mu * self.thresh_beta[t,k] * self.X[:,k,t]) / self.ns1
                    c_sum = torch.sum( -0.5 * self.omega * ((temp + self.beta[t,k] * self.X[:,k,t] / self.ns1) ** 2)+ (self.y - 0.5) * (temp + self.beta[t,k] * self.X[:,k,t] / self.ns1))
                    p_mp = p_mp_mat[t,k] 
                    p_mm = p_mm_mat[t,k] 

                    logc_vec[0] = c_sum + p_mp #C-
                    #logc_vec[1] = - 0.5 * torch.sum(temp ** 2)  + log1pexp(p_mp - p_m) + p_m #C0
                    logc_vec[1] = c_sum + p_mm #C1

                    logc_vec_max = max(logc_vec)
                    c_vec = torch.exp(logc_vec - logc_vec_max)
                    w_vec = c_vec/(torch.sum(c_vec))
                    c_ind = torch.multinomial(w_vec, num_samples=1).item()

                    lower = lower_vec[c_ind]
                    upper = upper_vec[c_ind]

                    #self.a, self.b = (lower - mu) / xi, (upper - mu) / xi

                    self.beta_hat[t,k] = torch.from_numpy(r_truncnorm(1, mean=mu, sd=xi, a=lower, b=upper)).float()
                    self.thresh_beta[t,k] = self.RTGP_thresh(self.beta_hat[t,k], self.w1)
                    temp += (self.beta[t,k] * self.thresh_beta[t,k] * self.X[:,k,t]) / self.ns1
        self.SR = self.update_SR()


    def update_eta(self):
        temp = self.SR + self.S0 + self.beta0
        thresh_eta_X0 = self.thresh_eta[None, :] * self.X0 / self.ns2
        for v in range(self.V): 
            temp -= self.eta[v] * thresh_eta_X0[:,v] 
            sigma2_eta = 1 / (torch.sum((thresh_eta_X0[:,v] ** 2) * self.omega) + 1/(self.sigma2_eta) + 1/self.xisq)
            mu = sigma2_eta * (torch.sum((self.y - 0.5 - temp * self.omega) * thresh_eta_X0[:,v]) + self.eta_hat[v] /self.xisq)
            self.eta[v] = torch.randn(1) * sigma2_eta.sqrt() + mu
            temp += self.eta[v] * thresh_eta_X0[:,v] 
        self.S0 = self.update_S0()

    def update_eta_hat(self):
        temp = self.SR + self.S0 + self.beta0
        
        #self.scale = torch.sqrt(self.tau2)

        xi = self.xisq.sqrt().item()
        thresh =  self.w2.item()
        if self.w2 > 0:
            logc_vec = torch.zeros(3)
            lower_vec = [-np.inf, - thresh, thresh]
            upper_vec = [-thresh, thresh, np.inf]

            p_mp_mat = torch.special.log_ndtr( -(self.w2 + self.eta) / xi) 
            p_m_mat = torch.special.log_ndtr( (self.w2 - self.eta) / xi) 
            p_mm_mat = torch.special.log_ndtr(-(self.w2 - self.eta) / xi) 

            for v in range(self.V):
                mu = self.eta[v].item()

                temp -= mu * self.thresh_eta[v] * self.X0[:,v] / self.ns2
                c_sum = torch.sum( -0.5 * self.omega * ((temp + self.eta[v] * self.X0[:,v] / self.ns2) ** 2)+ (self.y - 0.5) * (temp + self.eta[v] * self.X0[:,v] / self.ns2))
                
                p_mp = p_mp_mat[v] 
                p_m = p_m_mat[v] 
                p_mm = p_mm_mat[v]  

                logc_vec[0] = c_sum + p_mp #C-
                #logc_vec[1] = - 0.5 * torch.sum(temp ** 2)  + log1pexp(p_mp - p_m) + p_m #C0
                #logc_vec[1] = torch.sum( -0.5 * self.omega * (temp ** 2) + (self.y - 0.5) * temp) + p_m+torch.log(torch.exp(p_m-p_mp)-1)
                logc_vec[1] = torch.sum( -0.5 * self.omega * (temp ** 2) + (self.y - 0.5) * temp) + p_m + torch.log1p(-torch.exp(p_mp-p_m) + 1e-10)
                logc_vec[2] = c_sum + p_mm #C1

                logc_vec_max = max(logc_vec)
                c_vec = torch.exp(logc_vec - logc_vec_max)
                w_vec = c_vec/(torch.sum(c_vec))
                c_ind = torch.multinomial(w_vec, num_samples=1).item()

                lower = lower_vec[c_ind]
                upper = upper_vec[c_ind]
                #self.a, self.b = (lower-mu)/xi, (upper-mu)/xi
                #self.eta_hat[v] = truncnorm.rvs(self.a,self.b,loc=mu,scale=xi, size=1)
                self.eta_hat[v] = torch.from_numpy(r_truncnorm(1, mean=mu, sd=xi, a=lower, b=upper)).float()
                self.thresh_eta[v] = self.RTGP_thresh(self.eta_hat[v], self.w2)
                temp += mu * self.thresh_eta[v] * self.X0[:,v] / self.ns2
        else:
            logc_vec = torch.zeros(2)
            lower_vec = [-torch.inf, thresh]
            upper_vec = [thresh, torch.inf]

            p_mp_mat = torch.special.log_ndtr( -(self.w2 + self.eta) / xi) 
            p_mm_mat = torch.special.log_ndtr( -(self.w2 - self.eta) / xi)

            for v in range(self.V):
                mu = self.eta[v].item()
                temp -= mu * self.thresh_eta[v] * self.X0[:,v] / self.ns2
                c_sum = torch.sum( -0.5 * self.omega * ((temp + self.eta[v] * self.X0[:,v] / self.ns2) ** 2)+ (self.y - 0.5) * (temp + self.eta[v] * self.X0[:,v] / self.ns2))
                p_mp = p_mp_mat[v]
                p_mm = p_mm_mat[v]

                logc_vec[0] = c_sum + p_mp #C0
                logc_vec[1] = c_sum + p_mm #C1

                logc_vec_max = max(logc_vec)
                c_vec = torch.exp(logc_vec - logc_vec_max)
                w_vec = c_vec/(torch.sum(c_vec))
                c_ind = torch.multinomial(w_vec, num_samples=1).item()
                
                lower = lower_vec[c_ind]
                upper = upper_vec[c_ind]
                #self.a, self.b = (lower-mu)/xi, (upper-mu)/xi

                self.eta_hat[v] = torch.from_numpy(r_truncnorm(1, mean=mu, sd=xi, a=lower, b=upper)).float()
                self.thresh_eta[v] = self.RTGP_thresh(self.eta_hat[v], self.w2)
                temp += mu * self.thresh_eta[v] * self.X0[:,v] / self.ns2
        self.thresh_eta = self.RTGP_thresh(self.eta_hat, self.w2)
        self.S0 = self.update_S0()

    def update_w1(self):
        w_lower = torch.quantile(torch.abs(self.beta_hat), 0.25)
        w_upper = torch.quantile(torch.abs(self.beta_hat), 0.9)
        w_cand = torch.linspace(w_lower, w_upper, self.M)
        # temp = self.z - self.S0
        self.w_log_post = torch.zeros(self.M)
        for ind in range(self.M):
            thresh_beta_temp = self.RTGP_thresh(self.beta_hat, w_cand[ind])
            SR_temp =  torch.einsum('tk,nkt->n', self.beta * thresh_beta_temp, self.X) / self.ns1
            eta = self.S0 + SR_temp + self.beta0
            self.w_log_post[ind] = torch.sum(self.y * eta - log1pexp(eta)) + self.w_log_prior[ind]
            #self.w_log_post[ind] = torch.sum( -0.5 * self.omega * eta + (self.y - 0.5) * eta) + self.w_log_prior[ind]

        logp_max = torch.max(self.w_log_post)
        prob = torch.exp(self.w_log_post - logp_max)
        #print(prob)
        pp = prob / torch.sum(prob)
        self.w1 = w_cand[torch.multinomial(pp, num_samples=1).item()] 
        self.thresh_beta = self.RTGP_thresh(self.beta_hat, self.w1)
        self.SR = self.update_SR()

    def update_w2(self):
        w_lower = torch.quantile(torch.abs(self.eta_hat), 0.25)
        w_upper = torch.quantile(torch.abs(self.eta_hat), 0.9)
        w_cand = torch.linspace(w_lower, w_upper, self.M)
        #temp = self.z - self.SR
        self.w_log_post = torch.zeros(self.M)
        for ind in range(self.M):
            thresh_eta_temp = self.RTGP_thresh(self.eta_hat, w_cand[ind])
            S0_temp =  self.X0 @ (self.eta * thresh_eta_temp) / self.ns2
            eta = self.SR + S0_temp
            self.w_log_post[ind] = torch.sum(self.y * eta - log1pexp(eta))  + self.w_log_prior[ind]
            #self.w_log_post[ind] = torch.sum( -0.5 * self.omega * eta + (self.y - 0.5) * eta) + self.w_log_prior[ind]

        logp_max = torch.max(self.w_log_post)
        prob = torch.exp(self.w_log_post - logp_max)
        #print(prob)
        pp = prob / torch.sum(prob)
        self.w2 = w_cand[torch.multinomial(pp, num_samples=1).item()] 
        self.thresh_eta = self.RTGP_thresh(self.eta_hat, self.w2)
        self.S0 = self.update_S0()

    def update_beta0(self):
        temp = self.SR + self.S0 
        sigma2_beta0 = 1 / (torch.sum(self.omega) + 1/self.sigma2_beta0)
        mu_beta = sigma2_beta0 * (torch.sum(self.y - 0.5 - temp * self.omega))
        self.beta0 = torch.randn(1) * sigma2_beta0.sqrt() + mu_beta 

    def update_sigma2_beta0(self):
        a_eps_new = 1 / 2 + 0.0001
        b_eps_new = torch.sum(self.beta0 ** 2) / 2 + 0.0001
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.sigma2_beta0 = 1 / m.sample()

    def update_sigma2_eta(self):
        a_eps_new = self.V / 2 + self.a_sig_eta
        b_eps_new = torch.sum(self.eta ** 2) / 2 + self.b_sig_eta
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.sigma2_eta = 1 / m.sample()
        
    def update_xisq(self):
        if self.w2 == 0:
            a_eps_new = (self.V + self.K*self.T) / 2 + 0.0001
        else:
            a_eps_new = (self.K*self.T) / 2 + 0.0001
        b_eps_new = torch.sum((self.eta_hat - self.eta) ** 2) / 2 + torch.sum((self.beta_hat - self.beta) ** 2) / 2 + 0.0001
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.xisq = 1 / m.sample()


    def update_a_xi(self):
        a_eps_new = 1 
        b_eps_new = 1/self.xisq + 1/1
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.a_xi = 1 / m.sample()

    def update_sigma2_e(self):
        a_eps_new = (self.K*self.L )/ 2 +0.0001
        b_eps_new = torch.sum((self.e  ** 2 / self.lamb[:,None])) / 2  + 0.0001
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.sigma2_e = 1 / m.sample()


    def update_a_e(self):
        a_eps_new = 1 
        b_eps_new = 1/self.sigma2_e + 1/1
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.a_e = 1 / m.sample()
        
    def update_loglik_y(self):
        mu = self.SR + self.S0 + self.beta0
        logll = torch.sum(self.y * mu - log1pexp(mu))
        return logll
    
    def make_mcmc_samples(self):
        self.mcmc_e = torch.zeros((self.mcmc_sample, self.L, self.K))
        self.mcmc_beta = torch.zeros((self.mcmc_sample, self.T, self.K))
        self.mcmc_beta_hat = torch.zeros((self.mcmc_sample, self.T, self.K))
        self.mcmc_thresh_beta = torch.zeros((self.mcmc_sample, self.T, self.K))
        self.mcmc_w1 = torch.zeros(self.mcmc_sample)

        self.mcmc_eta = torch.zeros((self.mcmc_sample, self.V))
        self.mcmc_eta_hat = torch.zeros((self.mcmc_sample, self.V))
        self.mcmc_thresh_eta = torch.zeros((self.mcmc_sample, self.V))
        self.mcmc_w2 = torch.zeros((self.mcmc_sample))

        self.mcmc_sigma2_eta = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_e = torch.zeros(self.mcmc_sample)
        self.mcmc_xisq = torch.zeros(self.mcmc_sample)

        self.mcmc_beta0 = torch.zeros(self.mcmc_sample)
       
    
    def save_mcmc_samples(self, mcmc_iter):
        self.mcmc_e[mcmc_iter,:,:] = self.e
        self.mcmc_beta[mcmc_iter,:,:] = self.beta
        self.mcmc_beta_hat[mcmc_iter,:,:] = self.beta_hat
        self.mcmc_thresh_beta[mcmc_iter,:,:] = self.thresh_beta
        self.mcmc_w1[mcmc_iter] = self.w1

        self.mcmc_eta[mcmc_iter,:] = self.eta
        self.mcmc_eta_hat[mcmc_iter,:] = self.eta_hat
        self.mcmc_thresh_eta[mcmc_iter,:] = self.thresh_eta
        self.mcmc_w2[mcmc_iter] = self.w2

        self.mcmc_sigma2_eta[mcmc_iter] = self.sigma2_eta
        self.mcmc_sigma2_e[mcmc_iter] = self.sigma2_e
        self.mcmc_xisq[mcmc_iter] = self.xisq
        self.mcmc_beta0[mcmc_iter] = self.beta0 
    
    def post_means(self):
        post_beta = torch.mean(self.mcmc_beta, 0)
        post_eta = torch.mean(self.mcmc_eta, 0)
       
        post_beta_hat = torch.mean(self.mcmc_beta_hat, 0)
        post_eta_hat = torch.mean(self.mcmc_eta_hat, 0)

        post_thresh_beta = torch.mean(self.mcmc_thresh_beta, 0)
        post_thresh_eta = torch.mean(self.mcmc_thresh_eta, 0)

        post_w1 = torch.mean(self.mcmc_w1)
        post_w2 = torch.mean(self.mcmc_w2)
        post_sigma2_eta = torch.mean(self.mcmc_sigma2_eta)
        post_sigma2_e = torch.mean(self.mcmc_sigma2_e)
        post_xisq = torch.mean(self.mcmc_xisq)
        post_beta0 = torch.mean(self.mcmc_beta0)

        return post_beta, post_eta, post_beta_hat, post_eta_hat, post_thresh_beta, post_thresh_eta, post_w1, post_w2, post_sigma2_eta, post_sigma2_e, post_xisq, post_beta0
       

    def predict_score(self, X_test, X0_test):
        ndraw = self.mcmc_sample
        test_n = X_test.shape[0]

        pred_U = torch.zeros((ndraw, test_n))
        for i in range(ndraw):
            mu = torch.einsum('tk,nkt->n', self.mcmc_beta[i,:,:] * self.mcmc_thresh_beta[i,:,:], X_test) / self.ns1
            mu += X0_test @ (self.mcmc_eta[i,:] * self.mcmc_thresh_eta[i,:])  / self.ns2
            mu += self.mcmc_beta0[i]
            # z = torch.randn(test_n) + mu
            # pred_U[i] = (z > 0)*1
            p = 1 / (1 + torch.exp(-mu))
            pred_U[i] = torch.bernoulli(p)
        prob = torch.mean(pred_U,0)
        return prob







