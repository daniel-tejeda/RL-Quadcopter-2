import numpy as np
import math
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None, task_name='vanilla'):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_size = 4

        self.action_low = 300
        self.action_high = 1000

        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.task_name = task_name
        self.ep_reward = 0
        self.pos_reach_treshold = 0.5


    def get_reward(self, pose_rep=[]):
        return eval("self.get_reward_"+self.task_name)(pose_rep)


    def get_reward_vanilla(self, pose_rep):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def get_reward_takeoff(self, pose_rep):

        #weights for x,y,z dims ---> z more important for takeoff
        xyz_w = [.3, .3, .4]

        time_reward = 1
        dir_reward = np.dot(xyz_w, self.get_direction_reward(pose_rep))
        pos_reward = np.dot(xyz_w, self.get_pos_reward(pose_rep))

        #weights of direction / position / time
        step_weights = [.15, .25, .6]
        step_reward = np.tanh(np.dot(step_weights,[dir_reward, pos_reward, time_reward]))

        terminal_reward = 0

        if self.sim.done and self.sim.time < self.runtime:
            terminal_reward = -10 #crash
        elif self.get_task_success(pose_rep):
            terminal_reward = 10 #target position reached


        #print([appr_reward, pos_reward, time_reward, crash_penalty])
        #print("reward:{} z_approx:{} z_posdiff:{} crash_penalty:{}".format(reward, z_approx, -z_posdiff, crash_penalty))
        return step_reward + terminal_reward

    def get_direction_reward(self, pose_rep):

        pos_log = np.array(pose_rep)[:,:3]
        diff_init = np.abs(np.array(pos_log - self.target_pos))
        #print("\n",diff_init)
        mrate = np.diff(diff_init, axis=0)
        #print("\n",speeds)
        dir_reward = -np.average(mrate, axis=0, weights=[.3,.7])
        #print("\n",appr_reward)
        return dir_reward

    def get_pos_reward(self, pose_rep):

        pos_diff = self.get_pos_diff(pose_rep[self.action_repeat-1])
        pos_reward = np.array([-pd if pd > self.pos_reach_treshold else 10 for pd in pos_diff])
        return pos_reward

    def get_pos_diff(self, pose):
        return np.abs(pose[:3] - self.target_pos)

    def get_task_success(self, pose_rep):
        pos_diff = self.get_pos_diff(pose_rep[self.action_repeat-1])
        pos_r = np.array([0 if pd > self.pos_reach_treshold else 1 for pd in pos_diff])
        return np.sum(pos_r) == 3


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            #reward += self.get_reward()
            pose_all.append(self.sim.pose)

        reward = self.get_reward(pose_all)
        self.ep_reward += reward

        #done if target position is reached
        done = done or self.get_task_success(pose_all)

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.ep_reward = 0
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state



def sigmoid(x):
    return 1 / (1 + math.exp(-x))
