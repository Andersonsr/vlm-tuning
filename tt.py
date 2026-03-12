import torch

class dummy:
    def __init__(self):
        self.cooling_steps = 8820
        self.target_temperature = 60.
        self.step = 0
        self.temp = 100

    def update_temperature(self):
            cooling_rate = (100.0 - self.target_temperature) / self.cooling_steps
            temperature = max(100.0 - (self.step * cooling_rate), self.target_temperature)
            new_temp = torch.nn.Parameter(torch.log(torch.ones(1) * temperature), requires_grad=False)
            self.temp = new_temp
            self.step += 1

if __name__ == '__main__':
    d = dummy()
    for i in range(10000):
        d.update_temperature()
        print(d.temp.exp())     