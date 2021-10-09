import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Se definen los caudales a medir y los datos para trabajar
Qo = np.arange(20, 55, 5) # 20, 25... 50
Qw = np.arange(1, 8, 1) # 1, 2... 7
Vc_aux = ['0.78', '0.88', '0.88', '0.60', '0.60', '0.78', '0.88']
data = np.array([scipy.io.loadmat(f'G4/Qo={q}.00_Qw=5.00_Vc=0.60_.mat') for q in Qo])
data2 = np.array([scipy.io.loadmat(f'G4/Qo=30.00_Qw={q}.00_Vc={Vc_aux[q-1]}_.mat') for q in Qw])
LxP = 1/841.54

# Se definen un par de funciones útiles para ahorrar en líneas de código
def getarray(propstr, datos=data):
    return np.array([datos[i][propstr] for i in range(len(Qo))], dtype=object)

def debracket(array):
    return np.array([array[i][0, 0] for i in range(len(array))])

def time(array, fps=50):
    return np.arange(0, len(array)/fps, 1/fps)

# Las variables a trabajar
alphas = debracket(getarray('alpha'))
alphas2 = debracket(getarray('alpha', data2))
delta_aux = getarray('delta')[:, 0]
delta_aux2 = getarray('delta', data2)[:, 0]
deltas = np.array([float(string[3:]) for string in delta_aux])
deltas2 = np.array([float(string[3:]) for string in delta_aux2])
Cas = debracket(getarray('Ca')) + 5*0.0017
Cas2 = debracket(getarray('Ca', data2)) + Qw*0.0017
XFit = getarray('XFitr')
# Vgs = np.array([data[q]['Vg'][:, 0] for q in range(len(Qo))], dtype=object) + 0.6

Rmean = np.zeros(len(Qo))
for i in range(len(Qo)):
    Areas = np.array([area[0, 0] for area in data[i]['Areas'][:, 0]])
    meanArea = Areas[~np.isnan(Areas)].mean()
    Rmean[i] = np.sqrt(meanArea/np.pi) * LxP

#Regresión lineal alpha vs delta sin Qo = 35
reg_alphas, reg_deltas = np.delete(alphas, 3), np.delete(deltas, 3)
res = linregress(reg_deltas, reg_alphas)

def plot_figures():
    fig1, ax1 = plt.subplots(num=1)
    ax1.plot(deltas, alphas, '.k', lw=1, ms=8)
    ax1.plot(reg_deltas, res.intercept + res.slope*reg_deltas, 'k--', label='Regresión $R=0.99$')
    # ax1.plot(deltas2, alphas2, 'o--', lw=1, ms=4)
    ax1.set_xlabel(r'$\delta$')
    ax1.set_ylabel(r'$\alpha$')
    ax1.grid()
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(2, 1, figsize=(6.4, 6), num=2)
    ax2[0].plot(Cas, alphas, 'o--k', lw=1, ms=4)
    ax2[1].plot(Cas, Rmean, 'o--k', lw=1, ms=4)
    plt.setp(ax2[0].get_xticklabels(), visible=False)
    ax2[1].set_xlabel('Ca')
    ax2[0].set_ylabel(r'$\alpha$')
    ax2[1].set_ylabel('$R_0$')
    ax2[0].grid()
    ax2[1].grid()
    fig2.tight_layout()

    fig1.show()
    fig2.show()

def plot_modos():
    modo0 = np.array([XFit[q][:, 0] for q in range(len(Qo))], dtype=object)
    modo1 = np.array([XFit[q][:, 1] for q in range(len(Qo))], dtype=object)
    modo2 = np.array([XFit[q][:, 2] for q in range(len(Qo))], dtype=object)
    modo3 = np.array([XFit[q][:, 3] for q in range(len(Qo))], dtype=object)

    modo0[0][modo0[0]<=0.982] = np.nan
    modo0[2][modo0[2]<=0.995] = np.nan
    modo0[4][modo0[4]<=0.995] = np.nan

    fig3, ax3 = plt.subplots(num=3)
    for q in range(len(Qo)):
        ax3.plot(time(modo0[q]), modo0[q], '.', lw=1, ms=4, label=f'Qo={Qo[q]}')
    ax3.set_xlabel(r't [s]')
    ax3.set_ylabel(r'Radio medio adim.')
    ax3.grid()
    ax3.legend(loc='lower right')
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(num=4)
    modo2[0][modo2[0]<0.2] = np.nan
    for q in range(len(Qo)):
        modo2[q][modo2[q]>=0.255] = np.nan
        ax4.plot(time(modo2[q]), modo2[q], '.', lw=1, ms=4, label=f'Qo={Qo[q]}')
    ax4.set_xlabel(r't [s]')
    ax4.set_ylabel(r'Excentricidad')
    ax4.grid()
    ax4.legend(loc='lower right')
    fig4.tight_layout()

    fig5, ax5 = plt.subplots(num=5)
    modo3[0][modo3[0]<=-0.04] = np.nan
    for q in range(len(Qo)):
        modo3[q][modo3[q]>=0.02] = np.nan
        ax5.plot(time(modo3[q]), modo3[q], '.', lw=1, ms=4, label=f'Qo={Qo[q]}')
    ax5.set_xlabel(r't [s]')
    ax5.set_ylabel(r'Puntas y bala')
    ax5.grid()
    ax5.legend(loc='lower right')
    fig5.tight_layout()
    
    fig3.show()
    fig4.show()
    fig5.show()

if __name__ == '__main__':
    # plot_figures()
    plot_modos()
    pass