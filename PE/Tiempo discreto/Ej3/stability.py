import numpy as np

def compute_stability(M):
    """ Decides on the stabilty of the dicrete time linear dynamical
        system based on the matrix 'M'. """

    ls, vs = np.linalg.eig(M)
    ls = np.round(ls, decimals = 5)
    ls_abs = np.abs(ls)

    if (ls_abs < 1).all():
        return "stable", ls, vs
    if (ls_abs > 1).any():
        return "unstable", ls[ls_abs > 1], vs[:, ls_abs > 1]
    if (ls == -1).any() and (ls_abs[ls != -1] < 1).all():
        return "oscillatory stable", ls[ls == -1], vs[:, ls == -1]
    if (ls_abs == 1).any() and (ls_abs[ls_abs != 1] < 1).all():
        return "marginally stable", ls[ls_abs == 1], vs[:, ls_abs == 1]


A = np.array([[0.175, 0.125, 0.725, -0.225],
              [0.125, 0.175, -0.225, 0.725],
              [0.725, -0.225, 0.175, 0.125],
              [-0.225, 0.725, 0.125, -0.175]])
B = np.array([[0.3, -0.3, 0.95, -0.45],
             [-0.3, 0.3, -0.45, 0.95],
             [0.95, -0.45, 0.3, -0.3],
             [-0.45, 0.95, -0.3, 0.3]])
C = np.array([[0.05, -0.05, 0.7, -0.2],
              [-0.05, 0.05, -0.2, 0.7],
              [0.7, -0.2, 0.05, -0.05],
              [-0.2, 0.7, -0.05, 0.05]])
D = np.array([[-0.125, -0.125, 0.775, -0.025],
             [-0.125, -0.125, -0.025, 0.775],
             [0.775, -0.025, -0.125, -0.125],
             [-0.025, 0.775, -0.125, -0.125]])

np.set_printoptions(precision = 5)
for M, name in zip([A, B, C, D], ["A", "B", "C", "D"]):
    type, ls, vs = compute_stability(M)
    print(f"The system associated with {name} is {type}")
    print(f"The {type} components are:")
    for i, (l, v) in enumerate(zip(ls, vs.T)):
        print(f"l{i+1} = {l}")
        print(f"v{i+1} = {v}")
    print("------")
