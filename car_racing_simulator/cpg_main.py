import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import json
import sys

import dpg_orca.VehicleModel as VehicleModel
import dpg_orca.Track as Track
from dpg_orca.pure_pursuit import State, cpg_controller, calc_target_index

if __name__ == '__main__':
    sys.stdout.write("Hello\n")
    config = json.load(open('config.json'))
    track1 = Track.Track(config)

    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cpu")

    vehicle_model = VehicleModel.VehicleModel(config["n_batch"], device, config)

    x0 = torch.zeros(config["n_batch"], config["n_state"])


    u0 = torch.zeros(config["n_batch"], config["n_control"])

    state = State(x=track1.X[3], y=track1.Y[3], yaw=0.0, v=0.0)

    global_coordinates = []
    curvilinear_coordinates = []
    inputs = []
    target_ind = calc_target_index(state, track1.X, track1.Y)
    print(target_ind)
    state_data =[]


    for i in range(1000):
        # print(i)
        # d = 0
        # delta = 0
        # if x0[0,3] < 0.2:
        #     d = 0.2
        # if x0[0,1] > 0.00001:
        #     delta =  -0.02 - 0.1*x0[0,1]
        # elif x0[0,1]<-0.00001:
        #     delta = 0.02 - 0.5*x0[0,1]

        d, delta, target_ind = cpg_controller(track1.X, track1.Y, state, target_ind)
        print(target_ind)
        try:
            torch.FloatTensor([[d, delta]])
        except:
            d = d[0]


        u0 = torch.FloatTensor([[d,delta]])
        x0 = vehicle_model.dynModelBlend(x0, u0)
        curvilinear_coordinates.append(x0.detach().numpy().reshape(6, ))
        global_coordinates.append(track1.fromLocaltoGlobal(x0.detach().numpy().reshape(6,)))
        inputs.append(np.array([d,delta]))
        state.x, state.y, state.yaw = track1.fromLocaltoGlobal(x0.detach().numpy().reshape(6,))
        state.v = x0[:,3].detach().numpy()
        state_data.append([state.x,state.y,state.yaw,state.v])

        if x0[0,0]>20:
            break

    curvilinear_np = np.array(curvilinear_coordinates)
    global_np = np.array(global_coordinates)
    global_lower = []
    global_upper = []
    global_center = []
    for i in range(999):
        global_lower.append(track1.fromLocaltoGlobal(np.array([track1.s[i],track1.d_lower[i],0])))
        global_upper.append(track1.fromLocaltoGlobal(np.array([track1.s[i], track1.d_upper[i], 0])))
        global_center.append(track1.fromLocaltoGlobal(np.array([track1.s[i], 0, 0])))
    global_l = np.array(global_lower)
    global_u = np.array(global_upper)
    global_c = np.array(global_center)


    plt.plot(global_np[:, 0], global_np[:, 1])
    plt.plot(global_l[:, 0], global_l[:, 1])
    plt.plot(global_u[:,0], global_u[:,1])
    plt.plot(global_c[:, 0], global_c[:, 1])
    plt.title('Track')
    plt.show()

    plt.plot(global_np[:,0],global_np[:,1])
    # plt.plot(track1.X,track1.Y)
    plt.title('global')
    plt.show()

    plt.plot(curvilinear_np[:,0],curvilinear_np[:,1])
    plt.title('curvilinear')
    plt.show()

    plt.plot(curvilinear_np[:,0])
    plt.title('progress')
    plt.show()

    plt.plot(curvilinear_np[:,1])
    plt.title('lateral d')
    plt.show()

    plt.plot(curvilinear_np[:,2])
    plt.title('rel_yaw')
    plt.show()

    plt.plot(curvilinear_np[:,3])
    plt.title('velocity')
    plt.show()

    inputs = np.array(inputs)
    plt.plot(inputs[:,0])
    plt.plot(inputs[:, 1])
    plt.title('inputs')
    plt.show()

    state_data = np.array(state_data)
    plt.plot(state_data[:,0])
    plt.plot(state_data[:, 1])
    plt.plot(state_data[:, 2])
    plt.plot(state_data[:, 3])
    plt.show()

    temp=1







