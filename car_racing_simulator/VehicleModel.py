import torch
import car_racing_simulator.Track as Track
import numpy as np
import copy


class VehicleModel():
    def __init__(self,n_batch,device,config):

        self.device = device
        self.track = Track.Track(config)

        self.track_s = torch.from_numpy(self.track.s).type(torch.FloatTensor).to(self.device)
        self.track_kappa = torch.from_numpy(self.track.kappa).type(torch.FloatTensor).to(self.device)
        self.track_phi = torch.from_numpy(self.track.phi).type(torch.FloatTensor).to(self.device)
        self.track_X = torch.from_numpy(self.track.X).type(torch.FloatTensor).to(self.device)
        self.track_Y = torch.from_numpy(self.track.Y).type(torch.FloatTensor).to(self.device)

        self.track_d_upper = torch.from_numpy(self.track.d_upper).type(torch.FloatTensor).to(self.device)
        self.track_d_lower = torch.from_numpy(self.track.d_lower).type(torch.FloatTensor).to(self.device)
        self.track_angle_upper = torch.from_numpy(self.track.border_angle_upper).type(torch.FloatTensor).to(self.device)
        self.track_angle_lower = torch.from_numpy(self.track.border_angle_lower).type(torch.FloatTensor).to(self.device)


        self.n_full_state = config['n_state']
        self.n_control = config["n_control"]
        self.n_batch = n_batch

        # Model Parameters
        self.Cm1 = 0.287
        self.Cm2 = 0.054527
        self.Cr0 = 0.051891
        self.Cr2 = 0.000348

        self.B_r = 3.3852 / 1.2
        self.C_r = 1.2691
        self.D_r = 1. * 0.1737 * 1.2

        self.B_f = 2.579
        self.C_f = 1.2
        self.D_f = 1.05 * .192

        self.mass = 0.041
        self.mass_long = 0.041
        self.I_z = 27.8e-6
        self.l_f = 0.029
        self.l_r = 0.033

        self.L = 0.06
        self.W = 0.03

        self.tv_p = 0

        self.Ts = 0.03


    def dynModel(self, x, u):

        k1 = self.dx(x, u)
        k2 = self.dx(x + self.Ts / 2. * k1, u)
        k3 = self.dx(x + self.Ts / 2. * k2, u)
        k4 = self.dx(x + self.Ts * k3, u)

        x_next = x + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)

        return x_next

    def dx(self, x, u):

        f = torch.empty(self.n_batch, self.n_full_state,device=self.device)

        phi = x[:, 2]
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5]

        delta = u[:, 1]

        r_tar = delta * v_x / (self.l_f + self.l_r)

        [F_rx, F_ry, F_fy] = self.forceModel(x, u)

        f[:, 0] = v_x * torch.cos(phi) - v_y * torch.sin(phi)
        f[:, 1] = v_x * torch.sin(phi) + v_y * torch.cos(phi)
        f[:, 2] = r
        f[:, 3] = 1 / self.mass_long * (F_rx - F_fy * torch.sin(delta) + self.mass * v_y * r)
        f[:, 4] = 1 / self.mass * (F_ry + F_fy * torch.cos(delta) - self.mass * v_x * r)
        f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r + self.tv_p * (r_tar - r))
        return f

    def slipAngle(self, x, u):
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5]

        delta = u[:, 1]

        alpha_f = -torch.atan((self.l_f * r + v_y) / (v_x+1e-5)) + delta
        alpha_r = torch.atan((self.l_r * r - v_y) / (v_x+1e-5))

        return alpha_f, alpha_r

    def forceModel(self, x, u):
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5]

        D = u[:, 0]
        delta = u[:, 1]

        alpha_f = -torch.atan((self.l_f * r + v_y) / (v_x+1e-5)) + delta
        alpha_r = torch.atan((self.l_r * r - v_y) / (v_x+1e-5))

        F_rx = self.Cm1 * D - self.Cm2*v_x*D - self.Cr2*v_x**2 - self.Cr0
        F_ry = self.D_r * torch.sin(self.C_r * torch.atan(self.B_r * alpha_r))
        F_fy = self.D_f * torch.sin(self.C_f * torch.atan(self.B_f * alpha_f))

        return F_rx, F_ry, F_fy



    def compLocalCoordinates(self, x):

        x_local = torch.zeros(self.n_batch,7)

        dist = torch.zeros(self.n_batch,self.track.N)

        for i in range(self.track.N):
            dist[:,i] = (x[:,0] - self.track_X[i])**2 + (x[:,1] - self.track_Y[i])**2

        min_index = torch.argmin(dist, dim=1)
        # min_dist = torch.sqrt(dist[:,min_index])

        # if min_dist > 0.4:
        # print(min_index)
        for i in range(self.n_batch):
            # print(min_index[i])
            if dist[i,min_index[i]] <= 1e-13:
                s = self.track_s[min_index[i]]
                d = 0
                mu = x[2] - self.track_phi[min_index[i]]
                kappa = self.track_kappa[min_index[i]]

                x_local[i, :] = torch.tensor([s, d, mu, x[i, 3], x[i, 4], x[i, 5], kappa])
            else:
                a = torch.zeros(2)
                b = torch.zeros(2)
                a[0] = x[i, 0] - self.track_X[min_index[i]]
                a[1] = x[i, 1] - self.track_Y[min_index[i]]
                b[0] = self.track_X[min_index[i]+1] - self.track_X[min_index[i]]
                b[1] = self.track_Y[min_index[i]+1] - self.track_Y[min_index[i]]
                # a = self.vecToPoint(min_index, x)
                # b = self.vecTrack(min_index)a

                cos_theta = (torch.dot(a, b) / (torch.norm(a) * torch.norm(b)))
                # print("cos(theta): ",cos_theta)

                if cos_theta < 0:
                    min_index[i] = min_index[i] - 1
                    if min_index[i] < 0:
                        min_index[i] = self.track.N - 1

                    a[0] = x[i, 0] - self.track_X[min_index[i]]
                    a[1] = x[i, 1] - self.track_Y[min_index[i]]
                    b[0] = self.track_X[min_index[i] + 1] - self.track_X[min_index[i]]
                    b[1] = self.track_Y[min_index[i] + 1] - self.track_Y[min_index[i]]

                    cos_theta = (torch.dot(a, b) / (torch.norm(a) * torch.norm(b)))
                    # print("cos(theta): ",cos_theta)

                if cos_theta >= 1:
                    cos_theta = torch.tensor(0.9999999)

                rela_proj = torch.norm(a) * cos_theta / torch.norm(b)
                # print("realtive projection: ",rela_proj)
                rela_proj = max(min(rela_proj, 1), 0)
                # print("realtive projection: ",rela_proj)
                theta = torch.acos(cos_theta)

                error_sign = -torch.sign(a[0] * b[1] - a[1] * b[0])
                error = error_sign * torch.norm(a) * torch.sin(theta)
                if error != error:
                    error = 0.0

                # print(min_index[i])
                next_min_index = min_index[i] + 1
                if next_min_index > self.track.N:
                    next_min_index = 0

                s = self.track_s[min_index[i]] + (rela_proj * (-self.track_s[min_index[i]] + self.track_s[next_min_index]))
                d = error
                mu = self.wrapMu(x[i,2] - (self.track_phi[min_index[i]] + (rela_proj * (-self.track_phi[min_index[i]] + self.track_phi[next_min_index]))))
                kappa = self.track_kappa[min_index[i]] + (rela_proj * (-self.track_kappa[min_index[i]] + self.track_kappa[next_min_index]))
                if s!=s:
                    print(s)

                x_local[i,:] = torch.tensor([s, d, mu, x[i,3], x[i,4], x[i,5], kappa])

        return x_local

    def dynModelCurve(self, x, u):

        k1 = self.dxCurve(x, u).to(self.device)
        k2 = self.dxCurve(x + self.Ts / 2. * k1, u).to(self.device)
        k3 = self.dxCurve(x + self.Ts / 2. * k2, u).to(self.device)
        k4 = self.dxCurve(x + self.Ts * k3, u).to(self.device)

        x_next = x + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.).to(self.device)

        return x_next

    def dynModelBlend(self, x, u):

        blend_ratio = (x[:,3] - 0.3)/(0.2)

        lambda_blend = np.min([np.max([blend_ratio,0]),1])
        # blend_max = torch.max(torch.cat([blend_ratio.view(-1,1), torch.zeros(blend_ratio.size(0),1)],dim=1),dim=1)
        # blend_min = torch.min(torch.cat([blend_max.values.view(-1, 1), torch.ones(blend_max.values.size(0), 1)], dim=1), dim=1)
        # lambda_blend = blend_min.values

        if lambda_blend <1:
            v_x = x[:,3]
            v_y = x[:, 4]
            x_kin = torch.cat([x[:,0:3], torch.sqrt(v_x*v_x + v_y*v_y).reshape(-1,1)],dim =1)

            k1 = self.dxkin(x_kin, u).to(self.device)
            k2 = self.dxkin(x_kin + self.Ts / 2. * k1, u).to(self.device)
            k3 = self.dxkin(x_kin + self.Ts / 2. * k2, u).to(self.device)
            k4 = self.dxkin(x_kin + self.Ts * k3, u).to(self.device)

            x_kin_state = x_kin + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.).to(self.device)
            delta = u[:, 1]
            beta = torch.atan(self.l_r * torch.tan(delta) / (self.l_f + self.l_r))
            v_x_state = x_kin_state[:,3] * torch.cos(beta) # V*cos(beta)
            v_y_state = x_kin_state[:,3] * torch.sin(beta) # V*sin(beta)
            yawrate_state = v_x_state * torch.tan(delta)/(self.l_f + self.l_r)

            x_kin_full = torch.cat([x_kin_state[:,0:3],v_x_state.view(-1,1),v_y_state.view(-1,1), yawrate_state.view(-1,1)],dim =1)

            if lambda_blend ==0:
                return x_kin_full

        if lambda_blend >0:

            k1 = self.dxCurve(x, u).to(self.device)
            k2 = self.dxCurve(x + self.Ts / 2. * k1, u).to(self.device)
            k3 = self.dxCurve(x + self.Ts / 2. * k2, u).to(self.device)
            k4 = self.dxCurve(x + self.Ts * k3, u).to(self.device)

            x_dyn = x + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.).to(self.device)
            if lambda_blend ==1:
                return x_dyn

        return  x_dyn*lambda_blend + (1-lambda_blend)*x_kin_full

    def dynModelBlendBatch(self, x, u_unclipped):

        blend_ratio = (x[:,3] - 0.3)/(0.2)

        # lambda_blend = np.min([np.max([blend_ratio,0]),1])
        blend_max = torch.max(torch.cat([blend_ratio.view(-1,1), torch.zeros(blend_ratio.size(0),1)],dim=1),dim=1)
        blend_min = torch.min(torch.cat([blend_max.values.view(-1, 1), torch.ones(blend_max.values.size(0), 1)], dim=1), dim=1)
        lambda_blend = blend_min.values
        # print(lambda_blend)
        u = u_unclipped
        # u[:,0] = torch.clamp(u_unclipped[:,0],-0.2,1) #
        # u[:,1] = torch.clamp(u_unclipped[:,1],-0.35,0.35) # steering angle
        u[:,0] = torch.clamp(u_unclipped[:,0],-1,1) #
        u[:,1] = torch.clamp(u_unclipped[:,1],-1,1) # steering angle
        # u[:, 0] = u[:, 0]*1.2/2 + 0.4 #(-0.2,1)
        # u[:, 1] = u[:, 1] * 0.35 #(-0.35,035)

        v_x = x[:,3]
        v_y = x[:, 4]
        x_kin = torch.cat([x[:,0:3], torch.sqrt(v_x*v_x + v_y*v_y).reshape(-1,1)],dim =1)

        k1 = self.dxkin(x_kin, u).to(self.device)
        k2 = self.dxkin(x_kin + self.Ts / 2. * k1, u).to(self.device)
        k3 = self.dxkin(x_kin + self.Ts / 2. * k2, u).to(self.device)
        k4 = self.dxkin(x_kin + self.Ts * k3, u).to(self.device)

        x_kin_state = x_kin + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.).to(self.device)
        delta = u[:, 1]
        beta = torch.atan(self.l_r * torch.tan(delta) / (self.l_f + self.l_r))
        v_x_state = x_kin_state[:,3] * torch.cos(beta) # V*cos(beta)
        v_y_state = x_kin_state[:,3] * torch.sin(beta) # V*sin(beta)
        yawrate_state = v_x_state * torch.tan(delta)/(self.l_f + self.l_r)

        x_kin_full = torch.cat([x_kin_state[:,0:3],v_x_state.view(-1,1),v_y_state.view(-1,1), yawrate_state.view(-1,1)],dim =1)

        k1 = self.dxCurve(x, u).to(self.device)
        k2 = self.dxCurve(x + self.Ts / 2. * k1, u).to(self.device)
        k3 = self.dxCurve(x + self.Ts / 2. * k2, u).to(self.device)
        k4 = self.dxCurve(x + self.Ts * k3, u).to(self.device)

        x_dyn = x + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.).to(self.device)

        return  (x_dyn.transpose(0,1)*lambda_blend + x_kin_full.transpose(0,1)*(1-lambda_blend)).transpose(0,1)

    def dxkin(self, x, u):

        fkin = torch.empty(x.size(0), 4)

        s = x[:,0] #progress
        d = x[:,1] #horizontal displacement
        mu = x[:, 2] #orientation
        v = x[:, 3]

        delta = u[:, 1]

        kappa = self.getCurvature(s)

        beta = torch.atan(self.l_r*torch.tan(delta)/(self.l_f + self.l_r))

        fkin[:, 0] = (v*torch.cos(beta + mu))/(1.0 - kappa*d)   # s_dot
        fkin[:, 1] = v*torch.sin(beta + mu) # d_dot
        fkin[:, 2] = v*torch.sin(beta)/self.l_r - kappa*(v*torch.cos(beta + mu))/(1.0 - kappa*d)
        slow_ind =  v<=0.1
        D_0 = (self.Cr0 + self.Cr2*v*v)/(self.Cm1 - self.Cm2 * v)
        D_slow  = torch.max(D_0,u[:,0])
        D_fast = u[:,0]

        D = D_slow*slow_ind + D_fast*(~slow_ind)

        fkin[:, 3] = 1 / self.mass_long * (self.Cm1 * D - self.Cm2 * v * D - self.Cr0 - self.Cr2*v*v)

        return fkin

    def dxCurve_blend(self, x, u):

        f = torch.empty(self.n_batch, self.n_full_state)

        s = x[:,0] #progress
        d = x[:,1] #horizontal displacement
        mu = x[:, 2] #orientation
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5] #yawrate

        delta = u[:, 1]

        r_tar = delta * v_x / (self.l_f + self.l_r)

        blend_ratio = (v_x - 0.3)/(0.2)

        lambda_blend = np.min([np.max([blend_ratio,0]),1])
        kappa = self.getCurvature(s)

        if lambda_blend<1:
            fkin = torch.empty(self.n_batch, self.n_full_state)

            v = np.sqrt(v_x*v_x + v_y*v_y)
            beta = torch.tan(self.l_r*torch.atan(delta/(self.l_f + self.lr)))

            fkin[:, 0] = (v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d)   # s_dot
            fkin[:, 1] = v_x * torch.sin(mu) + v_y * torch.cos(mu) # d_dot
            fkin[:, 2] = v*torch.sin(beta)/self.l_r - kappa*((v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d))
            v_dot = 1 / self.mass_long * (self.Cm1 * u[:, 0] - self.Cm2 * v_x * u[:, 0])

            fkin[:, 3] = 1 / self.mass_long * (self.Cm1 * u[:, 0] - self.Cm2 * v_x * u[:, 0])
            fkin[:, 4] = delta * fkin[:, 3] * self.l_r / (self.l_r + self.l_f)
            fkin[:, 5] = delta * fkin[:, 3] / (self.l_r + self.l_f)
            if lambda_blend ==0:
                return fkin

        if lambda_blend>0:
            [F_rx, F_ry, F_fy] = self.forceModel(x, u)

            f[:, 0] = (v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d)
            f[:, 1] =  v_x * torch.sin(mu) + v_y * torch.cos(mu)
            f[:, 2] = r - kappa*((v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d))
            f[:, 3] = 1 / self.mass_long * (F_rx - F_fy * torch.sin(delta) + self.mass * v_y * r)
            f[:, 4] = 1 / self.mass * (F_ry + F_fy * torch.cos(delta) - self.mass * v_x * r)
            f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r + self.tv_p * (r_tar - r))
            if lambda_blend ==1:
                return f

        return f*lambda_blend + (1-lambda_blend)*fkin


    def dxCurve(self, x, u):

        f = torch.empty(x.size(0), self.n_full_state)

        s = x[:,0] #progress
        d = x[:,1] #horizontal displacement
        mu = x[:, 2] #orientation
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5] #yawrate

        delta = u[:, 1]

        r_tar = delta * v_x / (self.l_f + self.l_r)

        [F_rx, F_ry, F_fy] = self.forceModel(x, u)

        kappa = self.getCurvature(s)

        f[:, 0] = (v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d)
        f[:, 1] =  v_x * torch.sin(mu) + v_y * torch.cos(mu)
        f[:, 2] = r - kappa*((v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d))
        f[:, 3] = 1 / self.mass_long * (F_rx - F_fy * torch.sin(delta) + self.mass * v_y * r)
        f[:, 4] = 1 / self.mass * (F_ry + F_fy * torch.cos(delta) - self.mass * v_x * r)
        f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r + self.tv_p * (r_tar - r))
        return f

    def fromStoIndexBatch(self,s_in):

        s = s_in

        i_nan = (s != s)
        i_nan += (s >= 1e10) + (s <= -1e10)
        if torch.sum(i_nan) > 0:
            for i in range(self.n_batch):
                if i_nan[i]:
                    s[i] = 0
        # s[i_nan] = torch.zeros(torch.sum(i_nan))
        k = 0
        if torch.max(s) > self.track_s[-1] or torch.min(s) < 0:
            s = torch.fmod(s,self.track_s[-1])
            # i_wrapdown = (s > self.track_s[-1]).type(torch.FloatTensor)
            i_wrapup = (s < 0).type(torch.FloatTensor)

            s = s + i_wrapup * self.track_s[-1]

            if torch.max(s) > self.track_s[-1] or torch.min(s) < 0:
                s = torch.max(s, torch.zeros(self.n_batch))
                s = torch.min(s, self.track_s[-1] * torch.ones(self.n_batch))

            # print(s-s_in)

        index = (torch.floor(s / self.track.diff_s)).type(torch.LongTensor)
        if torch.min(index) < 0:
            print(index)

        rela_proj = (s - self.track_s[index]) / self.track.diff_s

        next_index = index + 1
        i_index_wrap = (next_index < self.track.N).type(torch.LongTensor)
        next_index = torch.fmod(next_index,self.track.N)# * i_index_wrap

        return index, next_index, rela_proj

    def getCurvature(self, s):
        index, next_index, rela_proj = self.fromStoIndexBatch(s)

        kappa = self.track_kappa[index] + rela_proj * (self.track_kappa[next_index] - self.track_kappa[index])

        return kappa

    def getTrackHeading(self,s):
        index, next_index, rela_proj = self.fromStoIndexBatch(s)
        phi = self.track_phi[index] + rela_proj * (self.track_phi[next_index] - self.track_phi[index])
        return phi

    def getLocalBounds(self,s):
        index, next_index, rela_proj = self.fromStoIndexBatch(s)
        d_upper = self.track_d_upper[index] + \
                  rela_proj * (self.track_d_upper[next_index] - self.track_d_upper[index])
        d_lower = self.track_d_lower[index] +\
                  rela_proj * (self.track_d_lower[next_index] - self.track_d_lower[index])

        angle_upper = self.track_angle_upper[index] + \
                      rela_proj * (self.track_angle_upper[next_index] - self.track_angle_upper[index])
        angle_lower = self.track_angle_lower[index] + \
                      rela_proj * (self.track_angle_lower[next_index] - self.track_angle_lower[index])

        return d_upper, d_lower,angle_upper,angle_lower



    def fromLocalToGlobal(self,state_local,phi_ref):
        s = state_local[:,0]
        d = state_local[:,1]
        mu = state_local[:,2]
        v_x = state_local[:, 3]
        v_y = state_local[:, 4]
        r = state_local[:, 5]
        index, next_index, rela_proj = self.fromStoIndexBatch(s)
        vec_track = torch.empty(self.n_batch,2)
        vec_track[:, 0] = (self.track_X[next_index] - self.track_X[index])* rela_proj
        vec_track[:, 1] = (self.track_Y[next_index] - self.track_Y[index])* rela_proj

        pos_index = torch.empty(self.n_batch,2)
        pos_index[:, 0] = self.track_X[index]
        pos_index[:, 1] = self.track_Y[index]

        pos_center = pos_index + vec_track

        phi_0 = self.track_phi[index]
        # phi_1 = self.track_phi[next_index]
        phi = phi_0
        # phi = self.getTrackHeading(s)#self.track_phi[index] + rela_proj * (self.track_phi[next_index] - self.track_phi[index])

        pos_global = torch.empty(self.n_batch,2)
        pos_global[:, 0] = pos_center[:, 0] - d * torch.sin(phi)
        pos_global[:, 1] = pos_center[:, 1] + d * torch.cos(phi)

        heading = phi + mu

        # heading = torch.fmod(heading,2*np.pi)

        upwrap_index = ((phi_ref - heading)>1.5*np.pi).type(torch.FloatTensor)
        downwrap_index = ((phi_ref - heading)<-1.5*np.pi).type(torch.FloatTensor)
        heading = heading - 2*np.pi*downwrap_index + 2*np.pi*upwrap_index

        upwrap_index = ((phi_ref - heading) > 1.5 * np.pi).type(torch.FloatTensor)
        downwrap_index = ((phi_ref - heading) < -1.5 * np.pi).type(torch.FloatTensor)
        heading = heading - 2 * np.pi * downwrap_index + 2 * np.pi * upwrap_index

        x_global = torch.empty(self.n_batch,self.n_full_state)
        x_global[:, 0] = pos_global[:, 0]
        x_global[:, 1] = pos_global[:, 1]
        x_global[:, 2] = heading
        x_global[:, 3] = v_x
        x_global[:, 4] = v_y
        x_global[:, 5] = r

        return x_global

    def fromStoIndex(self, s):

        s = torch.fmod(s,self.track_s[-1])
        # if s > self.track_kappa[-1]:
        #     s = s - self.track_kappa[-1]
        if s < 0:
            s = s + self.track_s[-1]
        elif s != s:
            s = torch.tensor(0.0)

        index = (torch.floor(s / self.track.diff_s)).type(torch.LongTensor)
        rela_proj = (s - self.track_s[index]) / self.track.diff_s
        return [index, rela_proj]

    def wrapMu(self, mu):
        if mu < -np.pi:
            mu = mu + 2 * np.pi
        elif mu > np.pi:
            mu = mu - 2 * np.pi
        return mu

