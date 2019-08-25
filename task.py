import numpy as np
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

        action_ranges = {
            'takeoff': (405,600),
            'land': (300, 405),
            'hover': (406,406),
            'reach': (300, 9000),
            'vanilla': (0,900)
        }

        self.action_low = action_ranges[task_name][0]
        self.action_high = action_ranges[task_name][1]

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.task_done = False
        self.task_name = task_name


    def get_reward(self):
        return eval("self.get_reward_"+self.task_name)()


    def get_reward_vanilla(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward


    def get_reward_takeoff(self):
        """Uses current pose of sim to return reward."""

        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        """
        reward = 0
        reward += 1.-.25*(abs(self.sim.pose[0] - self.target_pos[0]))
        reward += 1.-.25*(abs(self.sim.pose[1] - self.target_pos[1]))
        reward += 1.-.5*(abs(self.sim.pose[2] - self.target_pos[2]))
        """

        if self.sim.done and self.sim.runtime > self.sim.time:
            """ crash """
            reward = -100
            self.task_done = True
        elif self.sim.pose[2] == self.target_pos[2]:
            """ Z reward """
            reward = 100
            self.task_done = True

        return np.tanh(reward)


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        rewards = []
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            rewards.append(self.get_reward())
            pose_all.append(self.sim.pose)

        done = done or self.task_done
        reward = max(rewards)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.task_done = False
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
