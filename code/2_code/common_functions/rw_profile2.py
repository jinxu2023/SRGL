import tensorflow as tf
import numpy as np
import os
import scipy.io as scio
import time
import scipy.sparse as sp
from common_functions.utils import Data

profile_len = 18


def subgraph_sampling(D: np.ndarray, T: np.ndarray, R: np.ndarray,
                      center_d_idx: int, center_t_idx: int, hops: int):
    D_power = np.linalg.matrix_power(D, hops)
    T_power = np.linalg.matrix_power(T, hops)
    sampled_d_idxs = np.where(D_power[center_d_idx, :] >= 1)[0]
    sampled_t_idxs = np.where(T_power[center_t_idx, :] >= 1)[0]
    D_sub = D[sampled_d_idxs, :][:, sampled_d_idxs]
    T_sub = T[sampled_t_idxs, :][:, sampled_t_idxs]
    R_sub = R[sampled_d_idxs, :][:, sampled_t_idxs]
    A_sub = np.r_[np.c_[D_sub, R_sub], np.c_[R_sub.T, T_sub]]

    return sampled_d_idxs, sampled_t_idxs, D_sub, T_sub, R_sub, A_sub


def cal_profile(D, T, R, A, d_idx, t_idx, min_step, max_step):
    D_power = np.linalg.matrix_power(D, min_step)
    T_power = np.linalg.matrix_power(T, min_step)

    d_num, t_num = D.shape[0], T.shape[0]
    R = np.r_[np.c_[np.zeros(D.shape), R], np.c_[R.T, np.zeros(T.shape)]]
    Rp = R.copy()
    Rm = R.copy()
    Rp[d_idx, d_num + t_idx] = 1
    Rm[d_idx, d_num + t_idx] = 0
    Rp_power = np.linalg.matrix_power(Rp, min_step)
    Rm_power = np.linalg.matrix_power(Rm, min_step)
    Ap = A.copy()
    Am = A.copy()
    Ap[d_idx, d_num + t_idx] = 1
    Am[d_idx, d_num + t_idx] = 0
    Ap_power = np.linalg.matrix_power(Ap, min_step)
    Am_power = np.linalg.matrix_power(Am, min_step)

    profile = tuple()
    for i in range(max_step - min_step + 1):
        p_ds_D = D_power[d_idx, d_idx]
        p_ds_Ap = Ap_power[d_idx, d_idx]
        p_ds_Am = Am_power[d_idx, d_idx]
        p_ts_T = T_power[t_idx, t_idx]
        p_ts_Ap = Ap_power[d_num + t_idx, d_num + t_idx]
        p_ts_Am = Am_power[d_num + t_idx, d_num + t_idx]

        p_dt_Rp = Rp_power[d_idx, d_num + t_idx]
        p_dt_Rm = Rm_power[d_idx, d_num + t_idx]
        p_dt_Ap = Ap_power[d_idx, d_num + t_idx]
        p_dt_Am = Am_power[d_idx, d_num + t_idx]

        p_ads_D = np.trace(D_power) / d_num
        p_ats_T = np.trace(T_power) / t_num
        p_ads_Rp = np.trace(Rp_power[:d_num, :d_num]) / d_num
        p_ats_Rp = np.trace(Rp_power[d_num:, d_num:]) / t_num
        p_ads_Rm = np.trace(Rm_power[:d_num, :d_num]) / d_num
        p_ats_Rm = np.trace(Rm_power[d_num:, d_num:]) / t_num
        p_adat_Ap = np.sum(Ap_power[:d_num, d_num:]) / d_num
        p_adat_Am = np.sum(Am_power[:d_num, d_num:]) / d_num

        profile += (p_ds_D, p_ds_Ap, p_ds_Am, p_ts_T, p_ts_Ap, p_ts_Am, p_dt_Rp, p_dt_Rm, p_dt_Ap, p_dt_Am,
                    p_ads_D, p_ats_T, p_ads_Rp, p_ats_Rp, p_ads_Rm, p_ats_Rm, p_adat_Ap, p_adat_Am)

        D_power = D_power @ D
        T_power = T_power @ T
        Rp_power = Rp_power @ Rp
        Rm_power = Rm_power @ Rm

    return profile


def cal_profiles(D, T, R, mask, hops, min_step, max_step):
    d_num, t_num = D.shape[0], T.shape[0]
    pos_d_idxs, pos_t_idxs = np.where((mask > 0) & (R == 1))
    neg_d_idxs, neg_t_idxs = np.where((mask > 0) & (R == 0))
    pos_num, neg_num = len(pos_d_idxs), len(neg_d_idxs)
    pos_link_profiles = np.zeros((pos_num, profile_len * (max_step - min_step + 1)), 'float32')
    neg_link_profiles = np.zeros((neg_num, profile_len * (max_step - min_step + 1)), 'float32')

    for i, d_idx, t_idx in zip(range(pos_num), pos_d_idxs, pos_t_idxs):
        _, _, D_sub, T_sub, R_sub, A_sub = subgraph_sampling(D, T, R, d_idx, t_idx, hops)
        profile = cal_profile(D_sub, T_sub, R_sub, A_sub, d_idx, t_idx, min_step, max_step)
        pos_link_profiles[i, :] = profile
        print(i)

    for i, d_idx, t_idx in zip(range(neg_num), neg_d_idxs, neg_t_idxs):
        _, _, D_sub, T_sub, R_sub, A_sub = subgraph_sampling(D, T, R, d_idx, t_idx, hops)
        profile = cal_profile(D_sub, T_sub, R_sub, A_sub, d_idx, t_idx, min_step, max_step)
        neg_link_profiles[i, :] = profile

    pos_link_idxs = np.stack([pos_d_idxs, pos_t_idxs], axis=1)
    neg_link_idxs = np.stack([neg_d_idxs, neg_t_idxs], axis=1)

    return pos_link_profiles, neg_link_profiles, pos_link_idxs, neg_link_idxs


def get_profiles(D, T, R, mask, hops, min_step, max_step, full_dataset):
    profile_dir = '../../1_processed_data/rw_profiles'
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    profile_file = profile_dir + '/' + full_dataset + '_rw_profile(hops=' \
                   + str(hops) + ',min_step=' + str(min_step) \
                   + ',max_step=' + str(max_step) + ').mat'
    if os.path.isfile(profile_file):
        dic = scio.loadmat(profile_file)
        return dic['pos_link_profiles'], dic['neg_link_profiles'], dic['pos_link_idxs'], dic['neg_link_idxs']
    else:
        pos_link_profiles, neg_link_profiles, pos_link_idxs, neg_link_idxs = cal_profiles(
            D, T, R, mask, hops, min_step, max_step)
        scio.savemat(profile_file, {
            'pos_link_profiles': pos_link_profiles,
            'neg_link_profiles': neg_link_profiles,
            'pos_link_idxs': pos_link_idxs,
            'neg_link_idxs': neg_link_idxs
        }, do_compression=True)
        return pos_link_profiles, neg_link_profiles, pos_link_idxs, neg_link_idxs
