import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import json
import sys

import dpg_orca.VehicleModel as VehicleModel
import dpg_orca.Track

def main():

	sys.stdout.write("Hello\n")
	config = json.load(open('config.json'))

	# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
	device = torch.device("cpu")

	vehicle_model = VehicleModel.VehicleModel(config["n_batch"],device,config)

	x0 = torch.zeros(config["n_batch"], config["n_state"])
	x0[0, 3] = 0.0
	x0[1, 3] = 0.6

	u0 = torch.zeros(config["n_batch"], config["n_control"])


	for i in range(10):
		x0 = vehicle_model.dynModelBlendBatch(x0,u0)

		print(vehicle_model.getLocalBounds(x0[:,0]))

		# sys.stdout.write(str(x0))

		# sys.stdout.write("\n")


if __name__ == '__main__':
    main()

	
