import numpy as np

def get_relative_state(time_step):
    state = time_step.observations["info_state"][0]
    # print(pretty_board(time_step))
    try:
        pos = np.nonzero(state[0:20])[0]
        y = int(pos/5)
        x = int(pos%5)
        a_pos = np.array([x,y])
        a_status = 0
    except:
        try:
            pos = np.nonzero(state[20:40])[0]
            y = int(pos/5)
            x = int(pos%5)
            a_pos = np.array([x,y])
            a_status =1
        except:
            a_pos = np.array([5,1.5])
            a_status = 1
    try:
        pos = np.nonzero(state[40:60])[0]
        # print(1,pos)
        y = int(pos/5)
        x = int(pos % 5)
        b_pos = np.array([x, y])
        b_status = 0
    except:
        try: # 2nd try becasue in some cases when the game is done it is possible that none of the player exists
            pos = np.nonzero(state[60:80])[0]
            # print(2,pos)
            y = int(pos / 5)
            x = int(pos % 5)
            b_pos = np.array([x, y])
            b_status = 1
        except:
            b_pos = np.array([-1,1.5])
            b_status = 1
    if a_status==1:
        o_pos = a_pos
    elif b_status==1:
        o_pos = b_pos
    else:
        pos = np.nonzero(state[80:100])[0]
        y = int(pos / 5)
        x = int(pos % 5)
        o_pos = np.array([x, y])
    Ba = o_pos - a_pos # ball relative to a position
    Bb = o_pos - b_pos # Ball relative to b position
    G1a = np.array([4,1]) - a_pos
    G2a = np.array([4,2]) - a_pos
    G1b = np.array([0,1]) - b_pos
    G2b = np.array([0,2]) - b_pos
    rel_state = np.array([Ba,Bb, G1a, G2a, G1b, G2b]).reshape(12,)
    return a_status, b_status, rel_state

def get_two_state(time_step):
    state = time_step.observations["info_state"][0]
    # print(pretty_board(time_step))
    try:
        pos = np.nonzero(state[0:20])[0]
        y = int(pos/5)
        x = int(pos%5)
        a_pos = np.array([x,y])
        a_status = 0
    except:
        try:
            pos = np.nonzero(state[20:40])[0]
            y = int(pos/5)
            x = int(pos%5)
            a_pos = np.array([x,y])
            a_status =1
        except:
            a_pos = np.array([5,1.5])
            a_status = 1
    try:
        pos = np.nonzero(state[40:60])[0]
        # print(1,pos)
        y = int(pos/5)
        x = int(pos % 5)
        b_pos = np.array([x, y])
        b_status = 0
    except:
        try: # 2nd try becasue in some cases when the game is done it is possible that none of the player exists
            pos = np.nonzero(state[60:80])[0]
            # print(2,pos)
            y = int(pos / 5)
            x = int(pos % 5)
            b_pos = np.array([x, y])
            b_status = 1
        except:
            b_pos = np.array([-1,1.5])
            b_status = 1
    if a_status==1:
        o_pos = a_pos
    elif b_status==1:
        o_pos = b_pos
    else:
        pos = np.nonzero(state[80:100])[0]
        y = int(pos / 5)
        x = int(pos % 5)
        o_pos = np.array([x, y])
    Ba = o_pos - a_pos # ball relative to a position
    Bb = o_pos - b_pos # Ball relative to b position
    G1a = np.array([4,1]) - a_pos
    G2a = np.array([4,2]) - a_pos
    G1b = np.array([0,1]) - b_pos
    G2b = np.array([0,2]) - b_pos
    state1 = np.array([Ba, Bb, G1a, G2a, G1b, G2b]).reshape(12,)
    state2 = np.array([Bb, Ba, G1b, G2b, G1a, G2a]).reshape(12,)
    return a_status, b_status, state1, state2