import time
from bahiart_gym.server.baseenvironment import BaseEnvironment
from bahiart_gym.manager import Manager
import numpy as np
import math

class GatherBallDataset(BaseEnvironment):
    def __init__(self, monitor_port: int = 3200, connection_timeout_seconds: int = 60):
        
        self.manager = Manager(sleep_between_processes=0.1)

        self.positions: list = []
        self.ball_motion: dict = []
        self.ball_is_flying: list = []
        self.is_ball_moving: bool = False
        self.last_ball_position: list = [0, 0, 0]
        self.ball_speed: float = 0.0
        self.moving_speed: float = 0.05
        self.just_thrusted: bool = False
        self.minimum_motion: int = 20
        self.ball_start_position: list = [-14.5, 0, 0.040]

        self.max_angle: float = 60
        self.current_angle: float = 0
        self.angle_step: float = 5 #5
    
        self.max_power: float = 23
        self.min_power: float = 3
        self.current_power: float = self.min_power
        self.power_range: float = self.max_power - self.min_power
        self.power_step: float = 5 #5

        self.step: int = 0
        self.total_steps: int = int(((self.power_range / self.power_step) + 1) * ((self.max_angle / self.angle_step) + 1))
 
        self.manager.kill_server()
        self.manager.start_server()

        super().__init__(monitor_port, connection_timeout_seconds)

    def on_update(self):

        if self.step > self.total_steps: #has finished gathering
            self.save()
            print("Finished gathering dataset. Exiting")
            exit(0)
        
        self.start_play_mode() #gets to play on 

        speed = np.linalg.norm((np.array(self.world.ballCurrentPos) - np.array(self.last_ball_position)) / 0.02) #calculates speed
        if self.is_ball_moving and speed != 0: #in rare occasions the ball position won't be updated throughout the motion, which results in a 0 speed and ends motion tracking
            self.ball_speed = speed #if valid, updates ball speed.

        if (self.ball_speed > self.moving_speed and self.is_ball_moving) or self.just_thrusted: #always enter here at least once after thrusting and until the motion ends.
            self.ball_motion.append(
                {
                    'position': self.world.ballCurrentPos, 
                    'is_flying': True if self.world.ballCurrentPos[2] > 0.042 + 1e-5 else False
                }
            )
            self.ball_is_flying.append(True if self.world.ballCurrentPos[2] > 0.042 + 1e-5 else False)
            self.just_thrusted = False
        elif self.ball_speed <= self.moving_speed and self.is_ball_moving and len(self.ball_motion) > self.minimum_motion: #stores this motion
            self.store_ball_positions()
        elif not self.is_ball_moving: #thrusts ball
            self.thrust_ball()
            
        # print(f'ball current pos -> {self.world.ballCurrentPos}')
        self.last_ball_position = self.world.ballCurrentPos 

    def on_server_not_responsive(self):
        ...

    def start_play_mode(self):
        if self.world.playMode != self.world.PlayModes.PlayOn.value:
            self.trainer.changePlayMode(self.world.PlayModes.PlayOn.name)

    def store_ball_positions(self):

        if len(self.ball_motion) > 0 and len(self.ball_motion[0]['position']) > 0:
                #removes wrong values and repetitions
                slice_index = len(self.ball_motion) - 1 - (next(i for i, sublist in enumerate(self.ball_motion[::-1]) if sublist['position'][0] == self.ball_start_position[0])) #inverts list, iterates it until initial ball position
                self.ball_motion = self.ball_motion[slice_index:]

                self.is_ball_moving = False
                motion = [(list(np.round(np.array(cycle['position']) - np.array(self.ball_motion[0]['position']), 3)), cycle['is_flying']) for cycle in self.ball_motion]
                self.positions.append(motion)

        flying_percentage = (self.ball_is_flying.count(True) / len(self.ball_is_flying)) * 100

        print(f'flying percentage -> {flying_percentage}')
        print('aerial ball' if flying_percentage > 12.5 else 'rolling ball')

        self.ball_motion.clear()
        self.ball_is_flying.clear()

    def save(self):
        print("Saving dataset.")
        np.save('ball_dataset.npy', np.array(self.positions, dtype=object), allow_pickle=True)

    def thrust_ball(self):

        vector = self.next_vector()

        self.trainer.beamBall(self.ball_start_position[0], self.ball_start_position[1], self.ball_start_position[2], 0, 0, 0)
        time.sleep(2)
        self.trainer.beamBall(self.ball_start_position[0], self.ball_start_position[1], self.ball_start_position[2], vector[0], 0, vector[1])
        self.is_ball_moving = True
        self.just_thrusted = True

    def next_vector(self):

        angle = self.current_angle

        power = self.current_power
        self.current_power += self.power_step

        if self.current_power > self.max_power:
            self.current_power = self.min_power
            self.current_angle += self.angle_step
            self.save()
        
        self.step += 1
        print(f"CURRENT STEP -> {self.step}/{self.total_steps}")
            
        angle_radians: float = (angle * math.pi / 180)

        vector = [math.cos(angle_radians), math.sin(angle_radians)]

        print(f'Angle -> {angle}; Power -> {power}')

        return np.array(vector) * power

def main():
    GatherBallDataset()


if __name__ == "__main__":
    main()