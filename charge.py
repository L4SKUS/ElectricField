import numpy as np
import scipy.constants as constants

c = constants.c


class OscillatingCharge:

    def __init__(self, start_position=(-2e-9, 0), direction=(1, 0), amplitude=4e-9, speed=c):
        self.start_position = np.array(start_position)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.amplitude = amplitude
        self.w = speed/amplitude

    def xpos(self, t):
        xpos = self.start_position[0] + self.direction[0]*self.amplitude*(1-np.cos(self.w*t))
        return xpos

    def ypos(self, t):
        ypos = self.start_position[1] + self.direction[1]*self.amplitude*(1-np.cos(self.w*t))
        return ypos

    def xvel(self, t):
        xvel = self.direction[0]*self.amplitude*self.w*np.sin(self.w*t)
        return xvel

    def yvel(self, t):
        yvel = self.direction[1]*self.amplitude*self.w*np.sin(self.w*t)
        return yvel

    def xacc(self, t):
        xacc = self.direction[0]*self.amplitude*self.w**2*np.cos(self.w*t)
        return xacc

    def yacc(self, t):
        yacc = self.direction[1]*self.amplitude*self.w**2*np.cos(self.w*t)
        return yacc

    def retarded_time(self, tr, t, x, y):
        # Eq. 10.55
        return ((x-self.xpos(tr))**2 + (y-self.ypos(tr))**2)**0.5 - c*(t-tr)
