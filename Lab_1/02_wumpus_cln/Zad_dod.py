import numpy as np
P = np.array([[[0.108, 0.012], [0.072, 0.008]], [[0.016, 0.064], [0.144, 0.576]]]) #[cav][toothache][catch]

# P(cav|tot v cat) = P(cav,(too v cat))/P(too v cat)
P_cav_u_too_or_cat = (np.sum(P, axis=(-1, 0))[0] - P[0, 1, 1]) / ((np.sum(P, axis=(-1, 0))[0] - P[0, 1, 1] ) + np.sum(P, axis=(-1, 0))[1] - P[1, 1, 1])
print(f'P(cav|tot v cat) = {P_cav_u_too_or_cat}')

# Wersja macierzowa
# P(cav|tot v cat) = P(cav,(too v cat))/P(too v cat)
P_cav_too = np.sum(P, axis=-1)
P_cav = np.sum(P_cav_too, axis=-1)
P_cav_u_too_or_cat_ = (P_cav - P[:, 1, 1]) / (np.sum(np.sum(np.sum(P, axis=-1), axis=-1),axis=-1) - np.sum(P[:, 1, 1], axis=0))
print(f'P(cav|tot v cat)_ = {P_cav_u_too_or_cat_}')