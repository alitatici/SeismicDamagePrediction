import numpy as np
import Units


def get_ROTDpp(disp1, disp2):
    disp_X_Y = np.column_stack((disp1, disp2))
    # Rotating the Spectra (Projections)
    Rot_Matrix = np.zeros((2, 2))
    Rot_Disp = np.zeros((180, 1))
    for theta in range(0, 180, 1):
        Rot_Matrix[0, 0] = np.cos(np.deg2rad(theta))
        Rot_Matrix[0, 1] = np.sin(np.deg2rad(-theta))
        Rot_Matrix[1, 0] = np.sin(np.deg2rad(theta))
        Rot_Matrix[1, 1] = np.cos(np.deg2rad(theta))
        Rot_Disp[theta, 0] = np.max(np.matmul(disp_X_Y, Rot_Matrix)[:, 0])

    return Rot_Disp


def get_CumAbsVel(ground_acc, dt, factor):
    CAV = np.cumsum(np.abs(ground_acc) * dt)

    return CAV[-1] * factor


def get_AriasIntensity(ground_acc, dt, factor):
    g = 981
    AI = (np.pi / (2 * g)) * np.cumsum((np.abs(ground_acc) ** 2) * dt)

    return AI[-1] * factor


def get_PGA(ground_acc, factor):
    PGA = np.max(np.abs(ground_acc))

    return PGA * factor


def get_PGV(ground_vel, factor):
    PGV = np.max(np.abs(ground_vel))

    return PGV * factor


def get_PGD(ground_disp, factor):
    PGD = np.max(np.abs(ground_disp))

    return PGD * factor
