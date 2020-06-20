import numpy as np
from numpy import linalg as LA
import math
import torch


class Track:

    def __init__(self,config):
        self.path = config['data_dir']

        self.N = config['n_track']
        self.X = np.loadtxt(self.path + "x_center.txt")[:, 0]
        self.Y = np.loadtxt(self.path + "x_center.txt")[:, 1]
        self.s = np.loadtxt(self.path + "s_center.txt")
        self.phi = np.loadtxt(self.path + "phi_center.txt")
        self.kappa = np.loadtxt(self.path + "kappa_center.txt")
        self.diff_s = np.mean(np.diff(self.s))

        self.d_upper = np.loadtxt(self.path + "con_inner.txt")
        self.d_lower = np.loadtxt(self.path + "con_outer.txt")
        self.d_upper[520:565] = np.clip(self.d_upper[520:565], 0.2, 1)
        # self.d_lower[520:565] = np.clip(self.d_lower[520:565], -1, -0.2)
        self.border_angle_upper = np.loadtxt(self.path + "con_angle_inner.txt")
        self.border_angle_lower = np.loadtxt(self.path + "con_angle_outer.txt")

    def posAtIndex(self, i):
        return np.array([self.X[i], self.Y[i]])

    def vecToPoint(self, index, x):
        return np.array([x[0] - self.X[index], x[1] - self.Y[index]])

    def vecTrack(self, index):
        if index >= self.N - 1:
            next_index = 0
        else:
            next_index = index + 1

        return np.array([self.X[next_index] - self.X[index], self.Y[next_index] - self.Y[index]])

    def interpol(self, name, index, rela_proj):
        if index == self.N:
            index = 0
            next_index = 1
        else:
            next_index = index + 1

        if name == "s":
            return self.s[index] + (rela_proj * (self.s[next_index] - self.s[index]))
        if name == "phi":
            return self.phi[index] + (rela_proj * (self.phi[next_index] - self.phi[index]))
        if name == "kappa":
            return self.kappa[index] + (rela_proj * (self.kappa[next_index] - self.kappa[index]))

    def fromStoPos(self, s):

        index = math.floor(s / self.diff_s)
        rela_proj = (s - self.s[index]) / self.diff_s
        pos = [self.X[index], self.Y[index]] + self.vecTrack(index) * rela_proj
        return pos

    def fromStoIndex(self, s):
        if s > self.s[-1]:
            s = s - self.s[-1]
        elif s < 0:
            s = s + self.s[-1]

        s = max(s, 0)
        s = min(s, self.s[-1] )

        index = math.floor(s / self.diff_s)
        rela_proj = (s - self.s[index]) / self.diff_s
        return [index, rela_proj]

    def fromLocaltoGlobal(self, x_local):
        s = x_local[0]
        d = x_local[1]
        mu = x_local[2]

        [index, rela_proj] = self.fromStoIndex(s)
        pos_center = [self.X[index], self.Y[index]] + self.vecTrack(index) * rela_proj
        phi = self.interpol("phi", index, rela_proj)

        pos_global = pos_center + d * np.array([-np.sin(phi), np.cos(phi)])
        heading = phi + mu
        return [pos_global[0], pos_global[1], heading]

    def wrapMu(self, mu):
        if mu < -np.pi:
            mu = mu + 2 * np.pi
        elif mu > np.pi:
            mu = mu - 2 * np.pi
        return mu

    def compLocalCoordinates(self, x):
        dist = np.zeros(self.N)
        for i in range(self.N):
            dist[i] = LA.norm(self.vecToPoint(i, x))

        min_index = np.argmin(dist)
        min_dist = dist[min_index]

        # if min_dist > 0.4:
        # print(min_index)

        if min_dist <= 1e-13:
            s = self.s[min_index]
            d = 0
            mu = x[2] - self.phi[min_index]
            kappa = self.kappa[min_index]
            phi = self.phi[min_index]
        else:
            a = self.vecToPoint(min_index, x)
            b = self.vecTrack(min_index)

            cos_theta = (np.dot(a, b) / (LA.norm(a) * LA.norm(b)))
            # print("cos(theta): ",cos_theta)

            if cos_theta < 0:
                min_index = min_index - 1
                if min_index < 0:
                    min_index = self.N - 1
                a = self.vecToPoint(min_index, x)
                b = self.vecTrack(min_index)

                cos_theta = (np.dot(a, b) / (LA.norm(a) * LA.norm(b)))
                # print("cos(theta): ",cos_theta)

            if cos_theta >= 1:
                cos_theta = 0.99999999

            rela_proj = LA.norm(a) * cos_theta / LA.norm(b)
            # print("realtive projection: ",rela_proj)
            rela_proj = max(min(rela_proj, 1), 0)
            # print("realtive projection: ",rela_proj)
            theta = math.acos(cos_theta)

            error_sign = -np.sign(a[0] * b[1] - a[1] * b[0])
            error = error_sign * LA.norm(a) * np.sin(theta)
            error_dist = error_sign * LA.norm(
                self.posAtIndex(min_index) + b * LA.norm(a) * cos_theta / LA.norm(b) - [x[0], x[1]])

            s = self.interpol("s", min_index, rela_proj)
            d = error
            mu = self.wrapMu(x[2] - self.interpol("phi", min_index, rela_proj))
            kappa = self.interpol("kappa", min_index, rela_proj)
            phi = self.interpol("phi", min_index, rela_proj)

        return np.array([s, d, mu, x[3], x[4], x[5], kappa, phi])



