# Competitive Policy Gradient (CoPG) Algorithm
This repository contains all code and experiments for competitive policy gradient (CoPG) algorithm. The paper for competitive policy gradient can be found [here](https://arxiv.org/abs/2006.10611),
The code for Trust Region Competitive Policy Optimization (TRCPO) algorithm can be found [here](https://github.com/manish-pra/trcopo).

## Experiment videos are available [here](https://sites.google.com/view/rl-copo)
## Dependencies
1. Code is tested on python 3.5.2.
2. Only Markov Soccer experiment requires [OpenSpiel library](https://github.com/deepmind/open_spiel), Other 5 experiments can be run directly. 
3. Require [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)

## Repository structure
    .
    ├── notebooks
    │   ├── LQ_game.ipynb
    │   ├── bilinear_game.ipynb
    │   ├── RockPaperScissors.ipynb
    │   ├── matching_pennies.ipynb
    │   ├── MarkovSoccer.ipynb
    │   ├── CarRacing.ipynb
    ├── game                            # Each game have a saparate folder with this structure
    │   ├── game.py                     
    │   ├── copg_game.py                
    │   ├── gda_game.py
    │   ├── network.py
    │   ├── pretrained_models.py       (if applicable)
    │   ├── results.py                 (if applicable)
    ├── copg_optim
    │   ├── copg.py 
    │   ├── critic_functions.py 
    │   ├── utils.py 
    ├── car_racing_simulator
    └── ...
1. [Jupyter notebooks](https://github.com/manish-pra/copg/tree/master/notebooks) are the best point to start. It contains demonstrations and results. 
2. Folder [copg_optim](https://github.com/manish-pra/copg/tree/master/copg_optim) contains optimization code

## How to start ?
Open jupyter notebook and run it to see results.

or

```
git clone "adress"
cd copg
cd RockPaperScissors
python3 copg_rps.py
cd ..
cd tensorboard
tensordboard --logdir .
```
You can check results in the tensorboard.

## Experiment Demonstration
### &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; GDA vs GDA   &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                  CoPG vs CoPG
### ORCA Car Racing
&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;![](https://github.com/manish-pra/copg/blob/master/car_racing/gifs/CarRacingGDAvsGDA.gif) &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; ![](https://github.com/manish-pra/copg/blob/master/car_racing/gifs/CarRacingCoPGvsCoPG.gif)
### Rock Paper Scissors
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;<img src="https://github.com/manish-pra/copg/blob/master/rps/gifs/RPS%20GDA%20vs%20GDA.gif" width="350" height="250">&nbsp; &nbsp;&nbsp;<img src="https://github.com/manish-pra/copg/blob/master/rps/gifs/RPS%20cropped%20counter.gif" width="70" height="250">&nbsp; &nbsp;&nbsp; <img src="https://github.com/manish-pra/copg/blob/master/rps/gifs/RPS%20CoPG%20vs%20CoPG.gif" width="350" height="250"> 

### Markov Soccer
&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;<img src="https://github.com/manish-pra/copg/blob/master/markov_soccer/gifs/Soccer%20GDA%20vs%20GDA.gif" width="350" height="250">&nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <img src="https://github.com/manish-pra/copg/blob/master/markov_soccer/gifs/Soccer%20CoPG%20vs%20CoPG.gif" width="350" height="250"> 
### Matching Pennies
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;<img src="https://github.com/manish-pra/copg/blob/master/matchingpennies/gifs/MP%20GDA%20vs%20GDA.gif" width="350" height="250">&nbsp; &nbsp;&nbsp; <img src="https://github.com/manish-pra/copg/blob/master/matchingpennies/gifs/MP%20cropped2%20counter.gif" width="70" height="250">&nbsp; &nbsp;&nbsp; <img src="https://github.com/manish-pra/copg/blob/master/matchingpennies/gifs/MP%20CoPG%20vs%20CoPG.gif" width="350" height="250"> 
