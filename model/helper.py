import numpy as np
import torch
import torch.nn as nn
####################
# two correlation matrices
Sigma1 = np.array([
    [1, 0.7, 0.1, 0.1, 0.1, 0.2],
    [0.7, 1, 0.1, 0.1, 0.6, 0.1],
    [0.1, 0.1, 1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.7, 1, 0.1, 0.1],
    [0.1, 0.6, 0.1, 0.1, 1, 0.4],
    [0.2, 0.1, 0.1, 0.1, 0.4, 1]
])
Sigma0 = np.array([
    [1, 0.1, 0.1, 0.5, 0.1, 0.8],
    [0.1, 1, 0.1, 0.1, 0.3, 0.1],
    [0.1, 0.1, 1, 0.1, 0.1, 0.1],
    [0.5, 0.1, 0.1, 1, 0.1, 0.1],
    [0.1, 0.3, 0.1, 0.1, 1, 0.3],
    [0.8, 0.1, 0.1, 0.1, 0.3, 1]
])
######################
# hand-craft beta and visualize
# beta[i, 0]/beta[i, 1] is mean response for channel i for non-target/target respectively
beta = [[None, None] for _ in range(6)]
beta[0][1]=(0.000000e+00, 8.229730e-01, 1.623497e+00, 2.379737e+00, 3.071064e+00, 3.678620e+00, 4.185832e+00, 4.578867e+00, 4.847001e+00, 
                4.982922e+00, 4.982922e+00, 4.847001e+00, 4.578867e+00, 4.185832e+00, 3.678620e+00, 3.071064e+00, 2.379737e+00,  1.623497e+00,  
                8.229730e-01,  6.123234e-16, -5.877853e-01, -9.510565e-01, -9.510565e-01, -5.877853e-01, -2.449294e-16,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
beta[0][0]=(0.000000e+00,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
                7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
                9.965845e-01,  9.694003e-01,  9.157733e-01, 8.371665e-01,  7.357239e-01,
                6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  1.224647e-16,
            -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01, -4.898587e-17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
beta[1][1]=(0, 0, 0, 0, 0, -2.449294e-16, -5.877853e-01, -9.510565e-01, -9.510565e-01,
                -5.877853e-01,  6.123234e-16,  8.229730e-01,  1.623497e+00,
                2.379737e+00,  3.071064e+00,  3.678620e+00,  4.185832e+00,
                4.578867e+00,  4.847001e+00,  4.982922e+00,  4.982922e+00,
                4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
                3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
beta[1][0]=(0, 0, 0, 0, 0, -4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01,
                -1.175571e-01,  1.224647e-16,  1.645946e-01,  3.246995e-01,
                4.759474e-01,  6.142127e-01,  7.357239e-01,  8.371665e-01,
                9.157733e-01,  9.694003e-01,  9.965845e-01,  9.965845e-01,
                9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
                6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
beta[2][1]=(0, 0, 0, 0, 0,0, 0, 0, 0.000000e+00,  8.229730e-01,  1.623497e+00,  2.379737e+00,  3.071064e+00,
                3.678620e+00,  4.185832e+00,  4.578867e+00,  4.847001e+00,  4.982922e+00,
                4.982922e+00,  4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
                3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,  6.123234e-16,
                -5.877853e-01, -9.510565e-01, -9.510565e-01, -5.877853e-01, -2.449294e-16, 0, 0, 0, 0, 0, 0, 0)
beta[2][0]=(0, 0, 0, 0, 0,0, 0, 0, 0.000000e+00,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
                7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
                9.965845e-01,  9.694003e-01,  9.157733e-01, 8.371665e-01,  7.357239e-01,
                6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  1.224647e-16,
            -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01, -4.898587e-17, 0, 0, 0, 0, 0, 0, 0)
beta[3][1]=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.449294e-16, -5.877853e-01, -9.510565e-01, -9.510565e-01,
                -5.877853e-01,  6.123234e-16,  8.229730e-01,  1.623497e+00,
                2.379737e+00,  3.071064e+00,  3.678620e+00,  4.185832e+00,
                4.578867e+00,  4.847001e+00,  4.982922e+00,  4.982922e+00,
                4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
                3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01, 0,  0,  0)
beta[3][0]=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0, -4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01,
                -1.175571e-01,  1.224647e-16,  1.645946e-01,  3.246995e-01,
                4.759474e-01,  6.142127e-01,  7.357239e-01,  8.371665e-01,
                9.157733e-01,  9.694003e-01,  9.965845e-01,  9.965845e-01,
                9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
                6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01, 0,  0,  0)
beta[4][0] = beta[4][1] = beta[1][0]
beta[5][0] = beta[5][1] = -np.array((0, 0, 0, 0, 0, 0, 0,0,0, -4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01,
                 -1.175571e-01,  1.224647e-16,  1.645946e-01,  3.246995e-01,
                 4.759474e-01,  6.142127e-01,  7.357239e-01,  8.371665e-01,
                 9.157733e-01,  9.694003e-01,  9.965845e-01,  9.965845e-01,
                 9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
                 6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01, 0,  0,  0,0, 0, 0, 0))
beta = np.array(beta)

def log1pexp(u):
    val = np.zeros_like(u)
    val += np.where(u <= -37, np.exp(u), 0)
    val += np.where((u > -37 ) & (u <= 18), np.log(1+np.exp(u)), 0)
    val += np.where((u > 18 ) & (u <= 33.3), u + np.exp(-u), 0)
    val += np.where((u >= 33.3 ) , u , 0)
    return val

def change_beta(beta, peak_ratio):
    print(peak_ratio)
    beta[:4,1] = beta[:4,1] / 5 * peak_ratio
    return beta

class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.body(x)
        return y

def STGP_thresh(x, thresh):
    #thresh = (x > thresh)*1
    #thresh = torch.sign(x) * (torch.abs(x) - thresh) * (torch.abs(x) > thresh)
    thresh = torch.sign(x) * (torch.abs(x) - thresh) * (torch.abs(x) > thresh)
    return(thresh)

def RTGP_thresh(x, x_hat, thresh):
    #thresh = x * (torch.abs(x_hat) > thresh)
    thresh = x * (torch.abs(x_hat) > thresh)
    return(thresh)

import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
ro.numpy2ri.activate()
GPfit = importr('BayesGPfit')
def gen_basis(T, d=1, poly_degree=10, a=0.1, b=1):
    grids = GPfit.GP_generate_grids(d = 1, num_grids = T, grids_lim = np.array((-1,1)))
    lamb = GPfit.GP_eigen_value(poly_degree=poly_degree, a=a, b=b, d=d)
    Xmat = GPfit.GP_eigen_funcs_fast(grids = grids, poly_degree=poly_degree, a=a, b=b)
    B, _ = np.linalg.qr(Xmat)
    B = torch.from_numpy(B).float()
    lamb = torch.from_numpy(lamb).float()
    return B, lamb


def evaluate_chr_accu(yhat, y, R=19, S=5):
  yhat = yhat.reshape(R,S,2,6)
  y = y.reshape(R,S,2,6)
  correct = (yhat.mean(axis=1).argmax(axis=2)) == (y.mean(axis=1).argmax(axis=2))
  return np.mean(correct.sum(axis=1) == 2)


def true_char(Y_test_ss):
    flash_type = Y_test_ss[:,0]
    code = Y_test_ss[:,1]
    character = Y_test_ss[:,2]
    nchar = len(np.unique(character))
    true_row=np.zeros(nchar,dtype=int)
    true_col=np.zeros(nchar,dtype=int)

    for i in range(nchar):
        truth = np.unique(code[(character == i) & (flash_type == 1)])
        true_row[i]=(int(np.min(truth)))
        true_col[i]=(int(np.max(truth)))
    return np.column_stack((true_row, true_col)) 

def pred_char(Y_test_ss, post_pred):
    character = Y_test_ss[:,2]
    nchar = len(np.unique(character))

    row=np.zeros(nchar,dtype=int)
    col=np.zeros(nchar,dtype=int)
    for i in range(nchar):
        ind = (character == i)
        char_i = Y_test_ss[ind,1]
        post_pred_temp = post_pred[ind]
        prob=[np.mean(post_pred_temp[(char_i == (j+1))]) for j in range(12)]
        row[i]=(np.array(prob[:6]).argmax()+1)
        col[i]=(np.array(prob[6:]).argmax()+7)
    return np.column_stack((row, col))  


def true_char_tch(Y_test_ss):
    flash_type = Y_test_ss[:,0]
    code = Y_test_ss[:,1]
    character = Y_test_ss[:,2]
    nchar = len(np.unique(character))
    true_row=np.zeros(nchar,dtype=int)
    true_col=np.zeros(nchar,dtype=int)

    for i in range(nchar):
        truth = np.unique(code[(character == i) & (flash_type == 1)])
        true_row[i]=(int(np.min(truth)))
        true_col[i]=(int(np.max(truth)))
    return np.column_stack((true_row, true_col)) 

def pred_char_tch(Y_test_ss, post_pred):
    character = Y_test_ss[:,2]
    nchar = len(np.unique(character))

    row=np.zeros(nchar,dtype=int)
    col=np.zeros(nchar,dtype=int)
    for i in range(nchar):
        ind = (character == i)
        char_i = Y_test_ss[ind,1]
        post_pred_temp = post_pred[ind]
        prob=[torch.mean(post_pred_temp[(char_i == (j+1))]) for j in range(12)]
        row[i]=(np.array(prob[:6]).argmax()+1)
        col[i]=(np.array(prob[6:]).argmax()+7)
    return np.column_stack((row, col))  


def FDR(prob, level, intercept = 0):
    V = prob.shape[0]
    J = prob.shape[1]
    pred = np.ones((V,J))
    sort_p = np.sort(prob, 0)[::-1]
    lamb = np.zeros(V)
    p_cutoff = np.zeros(J)
    for j in range(J) :

        for v in range(V):
            lamb[v] = np.sum(1 - sort_p[:v,j])/(v+1)
        ind = (lamb <= level).nonzero()[0]
        if len(ind) == 0:
            p_lam = 0
        else:
            p_lam = sort_p[ind[-1],j]
        pred[:,j] = np.where(prob[:,j] > p_lam, 1.0, 0.0)
        p_cutoff[j] = p_lam
    prob_cut = prob * pred
    return pred

# def FDR(prob, level, intercept = 0):
#     V = prob.shape[0]
#     J = prob.shape[1]
#     pred = torch.ones(V,J)
#     sort_p,ind = torch.sort(prob, dim=0,descending=True)
#     lamb = torch.zeros(V)
#     p_cutoff = torch.zeros(J)
#     for j in range(J) :
#         if (intercept == 1) and (j == 0):
#             continue

#         for v in range(V):
#             lamb[v] = torch.sum(1 - sort_p[:v,j])/(v+1)
#         ind = (lamb <= level).nonzero().flatten()
#         if ind.nelement() == 0:
#             p_lam = 0
#         else:
#             p_lam = sort_p[ind[-1],j]
#         pred[:,j] = torch.where(prob[:,j] > p_lam, 1.0, 0.0)
#         p_cutoff[j] = p_lam
#     prob_cut = prob * pred
#     return pred