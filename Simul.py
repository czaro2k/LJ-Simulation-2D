import numpy as np
import matplotlib.pyplot as plt
import copy

class Simul:

    def __init__(self, N, T=10, dT=0.0001, box=8, radius=0.5, epsilon=1, sigma=1, temp=2.5):

        self.epsilon = epsilon
        self.sigma = sigma
        self.dT = dT
        self.steps = int(T / dT)
        self.N = N
        self.box = box
        self.radius = radius
        self.temp = temp
        self.r = self.initialize_position()
        self.v = self.initialize_velocities()
        self.Ep = []
        self.Ek = []
        self.r_cut = 2.5

    def initialize_velocities(self):

        v = np.random.random([self.N, 2])
        vx_mean = np.mean(v[:, 0])
        vy_mean = np.mean(v[:, 1])

        for i in v:
            i[0] -= vx_mean
            i[1] -= vy_mean

        E_start = np.mean(v[:, 0] ** 2) + np.mean(v[:, 1] ** 2)
        f = np.sqrt(self.temp / (E_start))

        return v * f

    def initialize_position(self):

        r = []
        sampling = True
        i = 0

        while sampling:
            noncongruent = True
            n_pos = np.random.random(2) * self.box
            if i == 0:
                r.append(n_pos.tolist())
            else:
                for j in range(len(r)):
                    if abs(np.linalg.norm(r[j] - n_pos)) <= self.radius * 2:
                        noncongruent = False
                if noncongruent:
                    r.append(n_pos.tolist())
            if len(r) == self.N:
                sampling = False
            i += 1

        r = np.array(r)

        return r

    def calc_dist(self):

        r_temp = copy.deepcopy(self.r)
        diff_matrix = np.zeros((self.N, self.N, 2))
        dist_matrix = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                for k in range(2):
                    diff_matrix[i][j][k] = r_temp[i][k] - r_temp[j][k]
                    if diff_matrix[i][j][k] > self.box / 2.:
                        diff_matrix[i][j][k] -= self.box
                    elif diff_matrix[i][j][k] < -self.box / 2.:
                        diff_matrix[i][j][k] += self.box

                dist_matrix[i][j] = np.linalg.norm(diff_matrix[i][j])

        return dist_matrix, diff_matrix

    def calc_pot(self):

        dist = self.calc_dist()[0]
        pot = np.where(dist != 0., 4.0 * self.epsilon * ((self.sigma / dist) ** 12 - (self.sigma / dist) ** 6), 0.)
        pot[dist > self.r_cut] = 0
        lower_triangle = np.matrix(np.tril(pot, k=-1))

        return np.sum(lower_triangle)

    def calc_kinetic(self):

        kin = (self.v[:, 0] ** 2 + self.v[:, 1] ** 2) / 2
        return np.sum(kin)

    def calc_force(self):

        dist, diff = self.calc_dist()

        force = np.zeros((self.N, self.N))
        force = np.where(dist != 0., 48 * self.epsilon * self.sigma ** 12 / dist ** 13 - 24 * self.epsilon * self.sigma ** 6 / dist ** 7, 0)
        force[dist > self.r_cut] = 0

        fx = force * diff[:, :, 0] / dist
        fy = force * diff[:, :, 1] / dist

        fx = np.nansum(fx, axis=1)
        fy = np.nansum(fy, axis=1)

        return fx, fy

    def make_step(self):

        fx, fy = self.calc_force()

        vx_new = self.v[:, 0] + fx * self.dT
        vy_new = self.v[:, 1] + fy * self.dT

        x_new = self.r[:, 0] + vx_new * self.dT
        y_new = self.r[:, 1] + vy_new * self.dT

        x_new = np.where(x_new >= self.box, x_new % self.box, x_new)
        x_new = np.where(x_new < 0, x_new % self.box, x_new)

        y_new = np.where(y_new >= self.box, y_new % self.box, y_new)
        y_new = np.where(y_new < 0, y_new % self.box, y_new)

        self.r[:, 0] = x_new
        self.r[:, 1] = y_new

        self.v[:, 0] = vx_new
        self.v[:, 1] = vy_new

    def plot(self):

        fig, ax = plt.subplots()
        plt.axis('square')
        plt.xlim(0, self.box)
        plt.ylim(0, self.box)

        for i in self.r:
            ax.add_patch(plt.Circle(i, radius=self.radius, color='red'))
        plt.show()
        plt.close()

    def plot_and_save(self, fname):
        fname = int(fname) // 300
        fname = str(fname)
        if len(fname) < 3:
            fname = (3 - len(fname)) * '0' + fname
        fig, ax = plt.subplots()
        plt.axis('square')
        plt.xlim(0, self.box)
        plt.ylim(0, self.box)

        for i in self.r:
            ax.add_patch(plt.Circle(i, radius=self.radius, color='red'))
        path = './pngs/' + fname + ".png"
        plt.savefig(path)
        plt.close()

    def plot_energies(self):

        min_lim = np.min([*self.Ep, *self.Ek]) - 2
        max_lim = np.max([*self.Ek, *self.Ek]) + 2

        plt.plot(np.arange(0, self.steps, 1), self.Ep)
        plt.ylim(min_lim, max_lim)
        plt.xlabel('time [s^-5]')
        plt.ylabel('potential energy [J]')
        plt.show()

        plt.plot(np.arange(0, self.steps, 1), self.Ek)
        plt.ylim(min_lim, max_lim)
        plt.xlabel('time [s^-5]')
        plt.ylabel('kinetic energy [J]')
        plt.show()

        plt.plot(np.arange(0, self.steps, 1), [p + k for p, k in zip(self.Ep, self.Ek)])
        plt.ylim(min_lim, max_lim)
        plt.xlabel('time [s^-5]')
        plt.ylabel('total energy [J]')
        plt.show()

    def run(self):
        for i in range(self.steps):
            self.Ep.append(self.calc_pot())
            self.Ek.append(self.calc_kinetic())
            self.make_step()
            if i % 300 == 0:
                self.plot()
        self.plot_energies()

    def run_and_save(self):
        for i in range(self.steps):
            self.Ep.append(self.calc_pot())
            self.Ek.append(self.calc_kinetic())
            self.make_step()
            if i % 300 == 0:
                self.plot_and_save(i)
        self.plot_energies()

simul = Simul(16)
simul.run_and_save()
