# Import the necessary modules and classes
from mlgame.communication import ml as comm
import pickle
import numpy as np
import os 


def ml_loop(side: str):

    with open(os.path.join(os.path.dirname(__file__),'save','model.pickle'), 'rb') as f:
        model = pickle.load(f)

    ball_served = False
    def move_to(player, pred) : #move platform to predicted position to catch ball 
        if player == '1P':
            if scene_info["platform_1P"][0]+20  > (pred-10) and scene_info["platform_1P"][0]+20 < (pred+10): return 0 # NONE
            elif scene_info["platform_1P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left
        else :
            if scene_info["platform_2P"][0]+20  > (pred-10)and scene_info["platform_2P"][0]+20 < (pred+10):
                if scene_info["platform_2P"][1]+30-scene_info["ball_speed"][1] > scene_info["ball"][1] : #slice
                    return 0
                    #if scene_info["ball"][0] > scene_info["platform_2P"][0]+20 and scene_info["ball_speed"][0] > 0:
                    #    return 1
                    #elif scene_info["ball"][0] > scene_info["platform_2P"][0]+20 and scene_info["ball_speed"][0] < 0:
                    #    return 1
                else :
                    return 0 # NONE
            elif scene_info["platform_2P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left

    def ml_loop_for_1P(): 
        if scene_info["ball_speed"][1] > 0 : # 球正在向下 # ball goes down
            x = ( scene_info["platform_1P"][1]-scene_info["ball"][1] ) // scene_info["ball_speed"][1] # 幾個frame以後會需要接  # x means how many frames before catch the ball
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x)  # 預測最終位置 # pred means predict ball landing site 
            bound = pred // 200 # Determine if it is beyond the boundary
            if (bound > 0): # pred > 200 # fix landing position
                if (bound%2 == 0) : 
                    pred = pred - bound*200                    
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) : # pred < 0
                if (bound%2 ==1) :
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            return move_to(player = '1P',pred = pred)
        else : # 球正在向上 # ball goes up
            return move_to(player = '1P',pred = 100)


    def ml_loop_for_2P():  # as same as 1P
        # print ("blocker : %s, ball : %s"%(scene_info["blocker"],scene_info["ball"]))
        if scene_info["ball_speed"][1] > 0 :
            if scene_info["ball_speed"][0] > 0:
                direction = 0
            else :
                direction = 1
        else :
            if scene_info["ball_speed"][0] > 0:
                direction = 2
            else:
                direction = 3
        X = [scene_info["ball"][0], scene_info["ball"][1], direction, scene_info["blocker"][0],scene_info["ball_speed"][0],scene_info["ball_speed"][1]]
        X = np.array(X).reshape((1,-1))
        pred = model.predict(X)
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

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_RIGHT"})
            ball_served = True
        else:
            if side == "1P":
                command = ml_loop_for_1P()
            else:
                command = ml_loop_for_2P()

            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})