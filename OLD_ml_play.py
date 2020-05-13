"""
The template of the script for the machine learning process in game pingpong
"""

# Import the necessary modules and classes
import pickle
import random
import numpy as np
import os.path as path
from mlgame.communication import ml as comm
from sklearn.neighbors import KNeighborsRegressor
import os
  
def ml_loop(side: str):
    """
    The main loop for the machine learning process
    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```
    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    trainX = []
    trainY = []
    if path.exists(path.join(path.dirname(__file__),'final/model_1p.asv')) :
        model = pickle.load(open(path.join(path.dirname(__file__),"final/model_1p.asv"), 'rb'))
        trainX = pickle.load(open(path.join(path.dirname(__file__),"final/data_1p.pickle"), 'rb'))
        trainY = pickle.load(open(path.join(path.dirname(__file__),"final/target_1p.pickle"), 'rb'))

    ball_served = False
    datacount = 0
    data = []
    landingPoint = 0
    preSpeedy = 0
    speedy = 0
    landing = 100
    
    def move_to(player, pred) : #move platform to predicted position to catch ball 
        if player == '1P':
            if scene_info["platform_1P"][0]+20  > (pred-8) and scene_info["platform_1P"][0]+20 < (pred+8): return 0 # NONE
            elif scene_info["platform_1P"][0]+20 <= (pred-8) : 
                return 1 # goes right
            else : 
                return 2 # goes left
        else :
            if scene_info["platform_2P"][0]+20  > (pred) and scene_info["platform_2P"][0]+20 < (pred): return 0 # NONE
            elif scene_info["platform_2P"][0]+20 <= (pred) : return 1 # goes right
            else : return 2 # goes left

    def dump_files(model, data, target) :
        model = model.fit(data, target.ravel())
        # dump the files
        pickle.dump(model,open(path.join(path.dirname(__file__),'final/model_1p.asv'), 'wb'))
        pickle.dump(data,open(path.join(path.dirname(__file__),'final/data_1p.pickle'), 'wb'))
        pickle.dump(target,open(path.join(path.dirname(__file__),'final/target_1p.pickle'), 'wb'))
    
    def ml_loop_for_2P():  # as same as 1P
        if scene_info["ball_speed"][1] > 0 : 
            return move_to(player = '2P',pred = 100)
        else : 
            x = ( scene_info["platform_2P"][1]+30-scene_info["ball"][1] ) // scene_info["ball_speed"][1] 
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x) 
            bound = pred // 200 
            if (bound > 0):
                if (bound%2 == 0):
                    pred = pred - bound*200 
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) :
                if bound%2 ==1:
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            return move_to(player = '2P',pred = pred)

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()

    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()

        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False
            
            if path.exists(path.join(path.dirname(__file__),'final/model_1p.asv')) :
                model = pickle.load(open(path.join(path.dirname(__file__),"final/model_1p.asv"), 'rb'))
                dump_files(model, trainX, trainY)

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})

            speedy = 0
            data = []
            datacount = 0
            ball_served = True

        else:
            if side == "1P":
                preSpeedy = speedy
                speedy = scene_info['ball_speed'][1]

                ballx = scene_info['ball'][0]
                bally = scene_info['ball'][1]
                speedx = scene_info['ball_speed'][0]
                blocker = scene_info['blocker'][0]

                # the very initital of the game
                if not path.exists(path.join(path.dirname(__file__),'final/model_1p.asv')) :
                    # save the data
                    data = data + [[ballx, bally, speedx, speedy, blocker]]
                    datacount += 1
                    # test if the ball is at the bottom
                    if (410 - bally) < speedy and bally < 410:
                        landingPoint = ((410 - bally) / speedy * speedx) + ballx
                        trainY = np.full((datacount, 1), landingPoint)
                        trainX = np.array(data)
                        # set the model
                        model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')
                        # fit the model
                        model = model.fit(trainX, trainY.ravel())
                        # dump the files
                        pickle.dump(model,open(path.join(path.dirname(__file__),'final/model_1p.asv'), 'wb'))
                        pickle.dump(trainX,open(path.join(path.dirname(__file__),'final/data_1p.pickle'), 'wb'))
                        pickle.dump(trainY,open(path.join(path.dirname(__file__),'final/target_1p.pickle'), 'wb'))
                        datacount = 0
                        data = []
                        landing = landingPoint
                    else :
                        landing = random.randint(0,200)

                # check whether the pickle and model file exist
                else :
                    # add the data into data
                    data = data + [[ballx, bally, speedx, speedy, blocker]]
                    datacount += 1

                    # landing = predictx
                    if speedy > 0 and preSpeedy <= 0 :
                        landing = model.predict([[ballx, bally, speedx, speedy, blocker]])
                    if speedy < 0 and preSpeedy >= 0 :
                        landing = model.predict([[ballx, bally, speedx, speedy, blocker]])
                        # print(landing)
                        
                    # test if the ball is at the bottom
                    if (415 - bally) <= speedy and bally <= 415 and speedy > 0 :
                        landingPoint = ((415 - bally) / speedy * speedx) + ballx
                        if abs(landing - landingPoint) > 0.1 :
                            trainY = np.append(trainY, np.full((datacount, 1), landingPoint), axis=0)
                            trainY = np.array(trainY)
                            trainX = np.concatenate((trainX, data))
                        data = []
                        datacount = 0
                        
                command = move_to('1P', landing)
                
                # make ball able to cut
                # if (415 - bally) <= abs(speedy) and abs(scene_info['platform_1P'][0]+20 - landing) < 8 :
                    # command = random.randint(0, 3)
                
            else:
                command = ml_loop_for_2P()
            
            
            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})