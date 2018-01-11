import os
import random
import sys
import imageio
import pylab as plt
from matplotlib.animation import ArtistAnimation
from scipy.integrate import odeint
import scipy as sp
import numpy as np
from tqdm import tqdm
import pickle


class TargetFormat:
    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"


def convertFile(inputpath, targetFormat):
    outputpath = os.path.splitext(inputpath)[0] + targetFormat
    print("converting {} to {}".format(inputpath, outputpath))

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i, im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")


class HodgkinHuxley:
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m = 1.0
    """membrane capacitance, in uF/cm^2"""

    g_Na = 120.0
    """Sodium (Na) maximum conductances, in mS/cm^2"""

    g_K = 36.0
    """Postassium (K) maximum conductances, in mS/cm^2"""

    g_L = 0.3
    """Leak maximum conductances, in mS/cm^2"""

    E_Na = 50.0
    """Sodium (Na) Nernst reversal potentials, in mV"""

    E_K = -77.0
    """Postassium (K) Nernst reversal potentials, in mV"""

    E_L = -54.387
    """Leak Nernst reversal potentials, in mV"""

    def __init__(self, ms):

        self.t = sp.arange(0, ms, 1)  # Time in ms
        self.current_periods = []
        self.noise = []

    def set_current(self, intensity, begin, end):
        self.current_periods.append((intensity, begin, end))

    @staticmethod
    def alpha_m(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

    @staticmethod
    def beta_m(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)

    @staticmethod
    def alpha_h(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)

    @staticmethod
    def beta_h(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

    @staticmethod
    def alpha_n(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    @staticmethod
    def beta_n(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K * n**4 * (V - self.E_K)

    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    @staticmethod
    def inject(intensity, begin, end, t):
        return intensity * (t > begin) - intensity * (t > end)

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        return sum(self.inject(intensity, begin, end, t) for intensity, begin, end in self.current_periods)

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n, a = X
        
        current_noise = 0#random.gauss(0, 0.00000000001)
        
        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dVdt += current_noise
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt, current_noise

    def main(self, normal=True, anim=False):

        """
        Main demo for the Hodgkin Huxley neuron model
        """

        if normal:
            self.normal_plot()

        if anim:
            self.animated_plot()

    def isi(self, V):

        times_where_fire = self.t[V > -0.5]
        isi = []

        for i, v  in enumerate(times_where_fire):
            if i == 0:
                isi.append(v)
            else:
                isi.append(v - times_where_fire[i - 1])

        pickle.dump(isi, open("isi.p", "wb"))
        pickle.dump(times_where_fire, open("times.p", "wb"))
        
        self.plot_isih(isi)
    
    def plot_isih(self, isi):
 
        plt.hist(isi, normed=1, alpha=.8)
        plt.title("Interval inter spike")
        plt.show()

    def normal_plot(self):

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32, 0], self.t, args=(self,))
        V = X[:, 0]
        m = X[:, 1]
        h = X[:, 2]
        n = X[:, 3]
        noise = X[:, 4]

        # calculate and save isi
        self.isi(V)

        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)

        plt.figure()

        plt.style.use('ggplot')

        plt.subplot(4, 1, 1)
        plt.title('Hodgkin-Huxley avec courant stochastique')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')

        plt.subplot(4, 1, 2)
        i_inj_values = [self.I_inj(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)

        plt.subplot(4, 1, 3)
        plt.plot(self.t, noise, label="bruit")
        plt.ylabel('Intensité du bruit')

        plt.show()

    def animated_plot(self):

        """

        creates two plots :
        one representing voltage/t
        the other one being input current/t

        """

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:, 0]
        m = X[:, 1]
        h = X[:, 2]
        n = X[:, 3]

        images = {
            "current_line": [],
            "voltage_line": [],
            "gating": []
        }

        # ----------------------------------- #

        fig1 = plt.figure()

        print("Saving animation voltage_line")

        for i in tqdm(self.t):

            if not i % 10:

                # plot a frame of the graph
                voltage_line = plt.plot(self.t[:i + 1], V[:i + 1], color="b")
                plt.title('Hodgkin-Huxley Neuron firing')
                plt.ylabel('V (mV)')
                plt.xlabel('t (ms)')

                images["voltage_line"].append(voltage_line)

        anim = ArtistAnimation(fig1, images["voltage_line"], interval=40, blit=True)
        anim.save("voltage_line.mp4", dpi=200, extra_args=['-vcodec', 'libx264'])

        plt.close()

        # ----------------------------------- #

        fig2 = plt.figure()

        print("Saving animation current_line")
        for i in tqdm(self.t):

            if not i % 10:

                current_line = plt.plot(
                    self.t[:i + 1],
                    list(map(self.I_inj, self.t[:i + 1])),
                    color="red"
                )

                plt.title('Injected current')
                plt.xlabel('t (ms)')
                plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
                plt.ylim(-1, 40)

                images["current_line"].append(current_line)

        anim = ArtistAnimation(fig2, images["current_line"], interval=40, blit=True)
        anim.save("current_line.mp4", dpi=200, extra_args=['-vcodec', 'libx264'])

        plt.close()

        # ----------------------------------- #

        fig3 = plt.figure()

        print("Saving animation gating channels")
        for i in tqdm(self.t):

            if not i % 10:

                m1, h1, n1 = plt.plot(
                    self.t[:i + 1], m[:i + 1], 'r',
                    self.t[:i + 1], h[:i + 1], 'g',
                    self.t[:i + 1], n[:i + 1], 'b'
                )

                plt.title('Channels activity')
                plt.ylabel('Gating Value')
                plt.xlabel('t (ms)')
                plt.legend()

                images["gating"].append((m1, h1, n1))

        anim = ArtistAnimation(fig3, images["gating"], interval=40, blit=True)
        anim.save("gating.mp4", dpi=200, extra_args=['-vcodec', 'libx264'])

        plt.close()

        for i in ["voltage_line.mp4", "current_line.mp4", "gating.mp4"]:
            convertFile(i, TargetFormat.GIF)


if __name__ == '__main__':

    # set total runtime (in milliseconds)
    time_in_ms = 1000
    average_current = 9

    runner = HodgkinHuxley(ms=time_in_ms)

    # set periods where we inject current (in ms) and intensity (in uA)
    for i in tqdm(range(time_in_ms - 1)):
        runner.set_current(intensity=np.random.poisson(9, 1)[0], begin=i, end=i+1)

    # set figures to animation mode
    runner.main(normal=True, anim=False)
