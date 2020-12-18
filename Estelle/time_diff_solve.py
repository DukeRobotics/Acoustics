import numpy as np
from scipy.optimize import fsolve


# ## 2d
#
# def diff_equation(hp1, hp2, target, t_diff):
#     return (np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)) - t_diff
#
# def system(target, *data):
#     hp1, hp2, hp3, diff_12, diff_23= data
#     return (diff_equation(hp1, hp2, target, diff_12), diff_equation(hp2, hp3, target, diff_23))
#
# if __name__ == '__main__':
#     hp1 = np.array([0, 0])
#     hp2 = np.array([0, 1])
#     hp3 = np.array([1, 0])
#     df_12 = 1
#     df_23 = -(np.sqrt(5) - 1)
#     data_val = (hp1, hp2, hp3, df_12, df_23)
#     solution = fsolve(system, (0, 0), args=data_val)
#     print solution


## 3d

def diff_equation(hp1, hp2, target, t_diff):
    return (np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)) - t_diff

def system(target, *data):
    hp1, hp2, hp3, hp4, diff_12, diff_23, diff_34= data
    return (diff_equation(hp1, hp2, target, diff_12), diff_equation(hp2, hp3, target, diff_23), diff_equation(hp3, hp4, target, diff_34))

if __name__ == '__main__':
    hp1 = np.array([0, 0, 0])
    hp2 = np.array([0, 1, 0])
    hp3 = np.array([1, 1, 0])
    hp4 = np.array([1, 0, 0])
    df_12 = np.sqrt(5) - np.sqrt(2)
    df_23 = np.sqrt(2) - np.sqrt(3)
    df_34 = np.sqrt(3) - np.sqrt(6)
    data_val = (hp1, hp2, hp3, hp4, df_12, df_23, df_34)
    solution = fsolve(system, (0, 0, 0), args=data_val)
    print solution


#[0, 2, -1]
