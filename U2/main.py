import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from own.data_analysis import Data

def gauss(params, x):
    A, mu, s = params.valuesdict().values()
    return A*np.exp(-(x-mu)**2/(2*s**2))

def cauchy(params, x):
    A, mu, s = params.valuesdict().values()
    return A / (1 + (x-mu)**2/s**2) * 1/(np.pi*s)

def main():
    gs = pd.read_csv('data\gate sweeps\gate_sweep_50v.csv')
    len_s = len(gs)//2
    GS_gauss = Data(gs.v_g[:len_s], gs.r[:len_s])
    GS_cauchy = Data(gs.v_g[:len_s], gs.r[:len_s])
    params_gauss = dict(A=500, mu=70, s=20) # initial guess
    params_cauchy = dict(A=500, mu=70, s=20) # initial guess
    GS_gauss.fit(gauss, params_gauss)
    GS_cauchy.fit(cauchy, params_cauchy)
    GS_gauss.report(plot=True, xylabels=['Gate voltage (V)', 'Resistance (Ohm)'],
                    legend=[r'Resistance ($\Omega$)', 'Fit Gauss'])

    GS_cauchy.report(plot=True, xylabels=['Gate voltage (V)', 'Resistance (Ohm)'],
                    legend=[r'Resistance ($\Omega$)', 'Fit Cauchy'])

    
    """
    fig, ax = plt.subplots()
    ax.plot(gs.v_g, 0.3/gs.i_ds)
    ax.set_xlabel('Gate voltage (V)')
    ax.set_ylabel('0.3/Drain current (A)')
    ax.set_title('Gate sweep')
    ax.grid()
    fig.tight_layout()
    fig.show()
    """

    """
    fig2, ax2 = plt.subplots()
    ax2.plot(gs.v_g[:len_s], gs.r[:len_s])
    ax2.set_xlabel('Gate voltage (V)')
    ax2.set_ylabel('Resistance (Ohm)')
    ax2.set_title('Gate sweep')
    ax2.grid()
    fig2.tight_layout()
    fig2.show()
    """

    """
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(gs.v_g, gs.i_ds, gs.r)
    ax3d.set_xlabel('Gate voltage (V)')
    ax3d.set_ylabel('Drain current (A)')
    ax3d.set_zlabel('Resistance (Ohm)')
    ax3d.set_title('Gate sweep')
    fig3d.tight_layout()
    fig3d.show()
    """


if __name__ == '__main__':

    main()

    """
    IV_s = pd.read_csv('data\IV\IV_(IV-II)_(c_f)_subida.csv', header=9)
    IV_b = pd.read_csv('data\IV\IV_(IV-II)_(c_f)_bajada.csv', header=9)
    fig3, ax3 = plt.subplots()
    ax3.plot(IV_s['Voltage [V]'], IV_s['Current [A]'], label='Subida')
    ax3.plot(IV_b['Voltage [V]'], IV_b['Current [A]'], label='Bajada')
    ax3.set_xlabel('Gate voltage (V)')
    ax3.set_ylabel('Drain current (A)')
    ax3.set_title('IV')
    ax3.grid()
    ax3.legend()
    fig3.tight_layout()
    fig3.show()
    """

