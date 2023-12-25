import numpy as np


class NumericalEvaluationDynamicResponse:
    def __init__(self, T, ksi, ag, dt):
        self.T = T
        self.ksi = ksi
        self.ag = ag
        self.t = np.arange(0, len(ag) * dt, dt).tolist()
        self.dt = dt


class NewmarkMethod(NumericalEvaluationDynamicResponse):

    def results(self):
        # Newmark Parameters Definitions
        GAMMA = 1 / 2
        # average acceleration
        BETA = 1 / 4
        # linear acceleration
        # BETA = 1/6
        mass = 1
        wn = 2 * np.pi / self.T
        c = self.ksi / (2 * mass * wn)
        k = wn ** 2 * mass
        ut = np.zeros(len(self.t))
        vt = np.zeros(len(self.t))
        at = np.zeros(len(self.t))
        const1 = mass / (BETA * (self.dt ** 2)) + (GAMMA * c) / (BETA * self.dt)
        const2 = mass / (BETA * self.dt) + ((GAMMA / BETA) - 1) * c
        const3 = (((1 / (2 * BETA)) - 1) * mass) + (self.dt * ((GAMMA / (2 * BETA)) - 1) * c)
        k_hat = k + const1
        pt = -mass * np.array(self.ag)
        at[0] = (pt[0] - c * vt[0] - k * ut[0]) / mass
        phat = np.array(np.zeros(len(self.t)))

        for i in range(1, len(self.t)):
            phat[i] = pt[i] + const1 * ut[i - 1] + const2 * vt[i - 1] + const3 * at[i - 1]
            ut[i] = phat[i] / k_hat
            vt[i] = (GAMMA / (BETA * self.dt)) * (ut[i] - ut[i - 1]) + (1 - (GAMMA / BETA)) * vt[i - 1] + (
                    self.dt * (1 - (GAMMA / (2 * BETA))) * at[i - 1])
            at[i] = (1 / (BETA * self.dt ** 2)) * (ut[i] - ut[i - 1]) - (1 / (BETA * self.dt)) * vt[i - 1] - (
                    (1 / (2 * BETA)) - 1) * at[i - 1]

        return [at, vt, ut]

    def PseudoSpectralValues(self, ut, omega):

        Sd = np.max(ut)
        Sv = omega * Sd
        Sa = (omega ** 2) * Sd

        return Sa, Sv, Sd

