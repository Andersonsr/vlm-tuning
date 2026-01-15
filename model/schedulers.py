from math import ceil


class LinearScheduler:
    def __init__(self, initial_temp, final_temp, num_steps):
        '''
        :param initial_temp: initial temperature value
        :param final_temp: final temperature value
        :param num_steps: number of steps until final temperature is reached
        '''

        self.cooling_rate = (initial_temp - final_temp) / num_steps
        self.num_steps = num_steps
        self.initial_temperature = initial_temp
        self.final_temperature = final_temp
        self.step = 0


    def update(self):
        temperature = max(self.initial_temperature - (self.step * self.cooling_rate), self.final_temperature)
        self.step += 1
        return temperature


class StepScheduler(LinearScheduler):
    def update(self):
        delta = self.initial_temperature - self.final_temperature
        step_len = self.num_steps // (delta // 5)
        cur_step = self.step // step_len
        self.step += 1
        return max(self.initial_temperature - (cur_step * 5), self.final_temperature)

