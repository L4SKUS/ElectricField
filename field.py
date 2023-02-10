import numpy as np
import scipy.constants as constants
from scipy import optimize

eps = constants.epsilon_0
pi = constants.pi
e = constants.e
c = constants.c


class Field:

    def __init__(self, charge, h=1e-20):

        self.charge = charge
        # h (float, optional): Tolerance for Newton's Method optimization. Defaults to 1e-20.
        # Causes overflow errors if too small
        self.h = h

    def calculate_E(self, t, x, y):
        t_array = np.ones(x.shape)
        t_array[:, :] = t
        # retarded time
        tr = optimize.newton(func=self.charge.retarded_time, x0=t_array, args=(t_array, x, y), tol=self.h)

        # retarded position
        rx = x - self.charge.xpos(tr)
        ry = y - self.charge.ypos(tr)
        r_mag = (rx ** 2 + ry ** 2) ** 0.5
        # retarded velocity
        vx = self.charge.xvel(tr)
        vy = self.charge.yvel(tr)
        # retarded acceleration
        ax = self.charge.xacc(tr)
        ay = self.charge.yacc(tr)
        # Eq. 10.71
        ux = c * rx / r_mag - vx
        uy = c * ry / r_mag - vy
        r_dot_u = rx * ux + ry * uy
        r_dot_a = rx * ax + ry * ay
        vel_mag = (vx ** 2 + vy ** 2) ** 0.5
        # Eq. 10.72
        const = e / (4 * pi * eps) * r_mag / r_dot_u ** 3

        xvel_field = const * (c ** 2 - vel_mag ** 2) * ux
        yvel_field = const * (c ** 2 - vel_mag ** 2) * uy

        xacc_field = const * (r_dot_a * ux - r_dot_u * ax)
        yacc_field = const * (r_dot_a * uy - r_dot_u * ay)

        Ex = xvel_field + xacc_field
        Ey = yvel_field + yacc_field

        return Ex, Ey
