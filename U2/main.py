import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as lf
from scipy.constants import epsilon_0

# Gráficos: columnas: gauss, cauchy | filas: subida, bajada, juntas

def get_fit(x, y, model : lf.Model, params=None, report=True, **kwargs):
    """
    x: array-like
        datos del eje x
    y: array-like
        datos del eje y
    model: lmfit.Model
        modelo a ajustar
    params: lmfit.Parameters
        parámetros del modelo
    kwargs: dict
        parámetros adicionales para la función lmfit.Model.fit
    """
    if params is None:
        params = model.guess(y, x=x)
    fit = model.fit(y, params, x=x, **kwargs)
    if report:
        print(fit.fit_report(show_correl=False))
    return fit


if __name__ == '__main__':
    gs = pd.read_csv('data\gate sweeps\gate_sweep_50v.csv')
    len_s = len(gs) // 2
    x_s = gs.v_g[:len_s].sort_values(ignore_index=True)
    y_s = gs.r[:len_s]
    x_b = gs.v_g[len_s:].reset_index(drop=True).sort_values(ignore_index=True, ascending=False)
    y_b = gs.r[len_s:].reset_index(drop=True)

    Gaussian = lf.models.GaussianModel()
    Cauchy = lf.models.LorentzianModel()
    Exp = lf.models.ExponentialModel()

    gauss_params = Gaussian.make_params(amplitude=136_579, center=79, sigma=62)
    cauchy_params = Cauchy.make_params(amplitude=145_404, center=58, sigma=58)
    exp_params = Exp.make_params(amplitude=313, center=0, decay=-40.52)

    gauss_fit = get_fit(x_s, y_s, Gaussian, gauss_params)
    cauchy_fit = get_fit(x_s, y_s, Cauchy, cauchy_params)
    exp_fit = get_fit(x_b, y_b, Exp, exp_params)

    fig, ax = plt.subplots()
    data_kws = dict(ms=2)
    #gauss_fit.plot_fit(ax=ax, datafmt='o', fitfmt='-', data_kws=data_kws)
    #cauchy_fit.plot_fit(ax=ax, datafmt='o', fitfmt='-', data_kws=data_kws)
    ax.plot(x_s, y_s, 'o', ms=2, label='Data')
    ax.plot(x_b, y_b, 'o', ms=2, label='Data')
    ax.plot(x_s, gauss_fit.init_fit, '-', label='Gaussian fit')
    ax.plot(x_s, cauchy_fit.init_fit, '-', label='Cauchy fit')
    ax.plot(x_b, exp_fit.init_fit, '-', label='Exponential fit')
    ax.set_xlabel('Gate voltage [V]')
    ax.set_ylabel(r'Resistance [$\Omega$]')
    ax.set_title('Voltaje de subida')
    # ax.legend()
    fig.tight_layout()
    fig.show()

    # Transconductancia

    IV_s = pd.read_csv('data\IV\IV_(IV-II)_(c_f)_subida.csv', header=9)
    IV_b = pd.read_csv('data\IV\IV_(IV-II)_(c_f)_bajada.csv', header=9)

    Linear = lf.models.LinearModel()
    linear_fit = get_fit(IV_s['Voltage [V]'], IV_s['Current [A]'], Linear)

    fig3, ax3 = plt.subplots()
    ax3.plot(IV_s['Voltage [V]'], IV_s['Current [A]'], label='Subida')
    ax3.plot(IV_b['Voltage [V]'], IV_b['Current [A]'], label='Bajada')
    ax3.plot(IV_s['Voltage [V]'], linear_fit.best_fit, label='Linear fit')
    ax3.set_xlabel('Gate voltage [V]')
    ax3.set_ylabel('Drain current [A]')
    ax3.set_title('IV')
    ax3.grid()
    ax3.legend()
    fig3.tight_layout()
    fig3.show()

    WL = 5/2                    # Width / Length
    R_ch = 1 / linear_fit.params['slope'].value
    R_s = WL * R_ch
    
    C_g = epsilon_0 * 3.9 / 1   # dividido en el grosor del óxido de silicio
    g_m = np.diff(0.3/gauss_fit.best_fit) / np.diff(x_s)
    mu = g_m / (WL * 0.3 * C_g)



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