import gym
import numpy as np
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import FloatParameter, Result
from opentuner.measurement import MeasurementInterface


from multiprocessing import pool
pool = pool.Pool()


def nyan(cfg):
    def core():
        env = gym.make('BipedalWalker-v2')

        env.reset()

        steps = 0
        total_reward = 0
        done = False
        a = np.array([0.0, 0.0, 0.0, 0.0])
        STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
        SPEED = cfg['speed']  # Will fall forward on higher speed
        state = STAY_ON_ONE_LEG
        moving_leg = 0
        supporting_leg = 1 - moving_leg
        SUPPORT_KNEE_ANGLE = cfg['support_knee_angle']
        supporting_knee_angle = SUPPORT_KNEE_ANGLE

        while not done:
            s, r, done, info = env.step(a)

            if done and r == -100:
                return -100, steps

            total_reward += r
            # if steps % 20 == 0 or done:
            #     print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            #     print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            #     print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            #     print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
            steps += 1

            moving_s_base = 4 + 5*moving_leg
            supporting_s_base = 4 + 5*supporting_leg

            hip_targ  = [None,None]   # -0.8 .. +1.1
            knee_targ = [None,None]   # -0.6 .. +0.9
            hip_todo  = [0.0, 0.0]
            knee_todo = [0.0, 0.0]

            if state==STAY_ON_ONE_LEG:
                hip_targ[moving_leg]  = 1.1
                knee_targ[moving_leg] = -0.6
                supporting_knee_angle += 0.03
                if s[2] > SPEED: supporting_knee_angle += 0.03
                supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
                knee_targ[supporting_leg] = supporting_knee_angle
                if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                    state = PUT_OTHER_DOWN
            if state==PUT_OTHER_DOWN:
                hip_targ[moving_leg]  = +0.1
                knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
                knee_targ[supporting_leg] = supporting_knee_angle
                if s[moving_s_base+4]:
                    state = PUSH_OFF
                    supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
            if state==PUSH_OFF:
                knee_targ[moving_leg] = supporting_knee_angle
                knee_targ[supporting_leg] = +1.0
                if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                    state = STAY_ON_ONE_LEG
                    moving_leg = 1 - moving_leg
                    supporting_leg = 1 - moving_leg

            if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
            if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
            if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
            if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

            hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
            hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
            knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
            knee_todo[1] -= 15.0*s[3]

            a[0] = hip_todo[0]
            a[1] = knee_todo[0]
            a[2] = hip_todo[1]
            a[3] = knee_todo[1]
            a = np.clip(0.5*a, -1.0, 1.0)

        print('noninfinite')
        return total_reward, steps

    total_reward, total_steps = [], []
    for result in pool.imap_unordered(core, range(100)):
        if result[0] == 100:
            return Result(time=float('inf'), state='TIMEOUT')

        total_reward.append(result[0] / 100)
        total_steps.append(result[1] / 100)

    return Result(time=total_reward**1.618 + total_steps)


class Agent(MeasurementInterface):

    # def __init__(self, *args, **kwargs):
    #     self.env = gym.make('BipedalWalker-v2')
    #     super().__init__(*args, **kwargs)

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
                FloatParameter('speed', 0.27, 0.32)
                )
        manipulator.add_parameter(
                FloatParameter('support_knee_angle', 0.09, 0.11)
                )
        manipulator.add_parameter(
                FloatParameter('angle_delta', 0.02, 0.04)
                )
        return manipulator

    env = gym.make('BipedalWalker-v2')

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        env = self.env

        totals = [[], []]
        for _ in range(3):
            env.reset()

            steps = 0
            total_reward = 0
            done = False
            a = np.array([0.0, 0.0, 0.0, 0.0])
            STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
            SPEED = cfg['speed']  # Will fall forward on higher speed
            state = STAY_ON_ONE_LEG
            moving_leg = 0
            supporting_leg = 1 - moving_leg
            SUPPORT_KNEE_ANGLE = cfg['support_knee_angle']
            supporting_knee_angle = SUPPORT_KNEE_ANGLE

            while not done:
                s, r, done, info = env.step(a)

                if done and r == -100:
                    return Result(time=float('inf'), state='TIMEOUT')

                total_reward += r
                # if steps % 20 == 0 or done:
                #     print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #     print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
                #     print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
                #     print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
                steps += 1

                moving_s_base = 4 + 5*moving_leg
                supporting_s_base = 4 + 5*supporting_leg

                hip_targ  = [None,None]   # -0.8 .. +1.1
                knee_targ = [None,None]   # -0.6 .. +0.9
                hip_todo  = [0.0, 0.0]
                knee_todo = [0.0, 0.0]

                if state==STAY_ON_ONE_LEG:
                    hip_targ[moving_leg]  = 1.1
                    knee_targ[moving_leg] = -0.6
                    supporting_knee_angle += cfg['angle_delta']
                    if s[2] > SPEED: supporting_knee_angle += cfg['angle_delta']
                    supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
                    knee_targ[supporting_leg] = supporting_knee_angle
                    if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                        state = PUT_OTHER_DOWN
                if state==PUT_OTHER_DOWN:
                    hip_targ[moving_leg]  = +0.1
                    knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
                    knee_targ[supporting_leg] = supporting_knee_angle
                    if s[moving_s_base+4]:
                        state = PUSH_OFF
                        supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
                if state==PUSH_OFF:
                    knee_targ[moving_leg] = supporting_knee_angle
                    knee_targ[supporting_leg] = +1.0
                    if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                        state = STAY_ON_ONE_LEG
                        moving_leg = 1 - moving_leg
                        supporting_leg = 1 - moving_leg

                if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
                if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
                if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
                if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

                hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
                hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
                knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
                knee_todo[1] -= 15.0*s[3]

                a[0] = hip_todo[0]
                a[1] = knee_todo[0]
                a[2] = hip_todo[1]
                a[3] = knee_todo[1]
                a = np.clip(0.5*a, -1.0, 1.0)

            totals[0].append(total_reward / 50)
            totals[1].append(steps / 50)

        return Result(time=300/(sum(totals[0])**1.618 + sum(totals[1])))

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal block size written to mmm_final_config.json:", configuration.data)
        self.manipulator().save_to_file(configuration.data, 'mmm_final_config.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    Agent.main(argparser.parse_args())

