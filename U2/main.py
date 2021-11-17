import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as lf
from scipy.constants import epsilon_0
# from own.timer import timeit

# Gráficos: columnas: gauss, cauchy | filas: subida, bajada, juntas

def get_fit(x, y, model : lf.Model, params=None, print_report=False, **kwargs):
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
    if print_report:
        print(fit.fit_report(
            show_correl=False,
            ))
    print(fit.model, fit.best_values)
    return fit


if __name__ == '__main__':
    gs = pd.read_csv('data\gate sweeps\gate_sweep_50v.csv')
    len_s = len(gs) // 2
    gs_s = gs[:len_s].sort_values('v_g', ignore_index=True)
    gs_b = gs[len_s:].sort_values('v_g', ignore_index=True, ascending=False)
    x_s, y_s = gs_s.v_g, gs_s.r
    x_b, y_b = gs_b.v_g, gs_b.r
    I_s, I_b = gs_s.i_ds, gs_b.i_ds

    Gaussian = lf.models.GaussianModel()
    Cauchy = lf.models.LorentzianModel()
    Exp = lf.models.ExponentialModel()      # no se usa


    # Subida
    gauss_params = Gaussian.make_params(amplitude=136_579, center=79, sigma=62)
    cauchy_params = Cauchy.make_params(amplitude=145_404, center=58, sigma=58)
    exp_params = Exp.make_params(amplitude=353, center=0, decay=-60.52)

    gauss_fit = get_fit(x_s, y_s, Gaussian, gauss_params)
    cauchy_fit = get_fit(x_s, y_s, Cauchy, cauchy_params)
    # exp_fit = get_fit(x_b, y_b, Exp, exp_params)

    fig, ax = plt.subplots()
    ax.plot(x_s, y_s, 'o', ms=2, label='Upsweep')
    ax.plot(x_s, gauss_fit.best_fit, '-', label='Gaussian fit')
    ax.plot(x_s, cauchy_fit.best_fit, '-', label='Cauchy fit')
    gauss_dev = gauss_fit.eval_uncertainty(sigma=3)
    cauchy_dev = cauchy_fit.eval_uncertainty(sigma=3)
    # ax.fill_between(x_s, gauss_fit.best_fit - gauss_dev, gauss_fit.best_fit + gauss_dev, alpha=0.5, label='Gaussian fit error')
    # ax.fill_between(x_s, cauchy_fit.best_fit - cauchy_dev, cauchy_fit.best_fit + cauchy_dev, alpha=0.5, label='Cauchy fit error')

    ax.set_xlabel('Gate voltage [V]')
    ax.set_ylabel(r'Resistance [$\Omega$]')
    ax.set_title('IV-II GFET Gate sweep')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.show()


    # Bajada
    gauss_params = Gaussian.make_params(amplitude=136_579, center=79, sigma=62)
    cauchy_params = Cauchy.make_params(amplitude=145_404, center=58, sigma=58)
    exp_params = Exp.make_params(amplitude=353, center=0, decay=-60.52)

    gauss_fit = get_fit(x_b, y_b, Gaussian, gauss_params)
    cauchy_fit = get_fit(x_b, y_b, Cauchy, cauchy_params)
    # exp_fit = get_fit(x_b, y_b, Exp, exp_params)

    fig, ax = plt.subplots()
    ax.plot(x_b, y_b, 'o', ms=2, label='Downsweep')
    ax.plot(x_b, gauss_fit.best_fit, '-', label='Gaussian fit')
    ax.plot(x_b, cauchy_fit.best_fit, '-', label='Cauchy fit')
    gauss_dev = gauss_fit.eval_uncertainty(sigma=3)
    cauchy_dev = cauchy_fit.eval_uncertainty(sigma=3)
    # ax.fill_between(x_b, gauss_fit.best_fit - gauss_dev, gauss_fit.best_fit + gauss_dev, alpha=0.5, label='Gaussian fit error')
    # ax.fill_between(x_b, cauchy_fit.best_fit - cauchy_dev, cauchy_fit.best_fit + cauchy_dev, alpha=0.5, label='Cauchy fit error')

    ax.set_xlabel('Gate voltage [V]')
    ax.set_ylabel(r'Resistance [$\Omega$]')
    ax.set_title('IV-II GFET Gate sweep')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.show()


    # Subida y bajada
    gauss_params = Gaussian.make_params(amplitude=136_579, center=79, sigma=62)
    cauchy_params = Cauchy.make_params(amplitude=145_404, center=58, sigma=58)
    exp_params = Exp.make_params(amplitude=353, center=0, decay=-60.52)

    gauss_fit = get_fit(gs.v_g, gs.r, Gaussian, gauss_params)
    cauchy_fit = get_fit(gs.v_g, gs.r, Cauchy, cauchy_params)

    fig, ax = plt.subplots()
    ax.plot(gs.v_g, gs.r, 'o', ms=2, label='Full Gate sweep')
    ax.plot(gs.v_g, gauss_fit.best_fit, '-', label='Gaussian fit')
    ax.plot(gs.v_g, cauchy_fit.best_fit, '-', label='Cauchy fit')
    gauss_dev = gauss_fit.eval_uncertainty(sigma=3)
    cauchy_dev = cauchy_fit.eval_uncertainty(sigma=3)
    # ax.fill_between(gs.v_g, gauss_fit.best_fit - gauss_dev, gauss_fit.best_fit + gauss_dev, alpha=0.5, label='Gaussian fit error')
    # ax.fill_between(gs.v_g, cauchy_fit.best_fit - cauchy_dev, cauchy_fit.best_fit + cauchy_dev, alpha=0.5, label='Cauchy fit error')

    ax.set_xlabel('Gate voltage [V]')
    ax.set_ylabel(r'Resistance [$\Omega$]')
    ax.set_title('IV-II GFET Gate sweep')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.show()
    

    # Transconductancia
    
    IV_s = pd.read_csv('data\IV\IV_(IV-II)_(c_f)_subida.csv', header=9)
    IV_b = pd.read_csv('data\IV\IV_(IV-II)_(c_f)_bajada.csv', header=9)

    Linear = lf.models.LinearModel()
    linear_fit = get_fit(IV_s['Voltage [V]'], IV_s['Current [A]'], Linear)

    fig3, ax3 = plt.subplots()
    ax3.plot(IV_s['Voltage [V]'], IV_s['Current [A]'], 'o', ms=3, label='Upsweep')
    # ax3.plot(IV_b['Voltage [V]'], IV_b['Current [A]'], label='Bajada')
    ax3.plot(IV_s['Voltage [V]'], linear_fit.best_fit, label='Linear fit')
    ax3.set_xlabel('Gate voltage [V]')
    ax3.set_ylabel('Drain current [A]')
    ax3.set_title('IV-II sheet Resistance')
    ax3.grid()
    ax3.legend()
    fig3.tight_layout()
    fig3.show()

    WL = 5/2                    # Width / Length
    R_ch = 1 / linear_fit.params['slope'].value
    R_s = WL * R_ch
    
    C_g = epsilon_0 * 3.9 / 90e-9
    g_m = - (np.diff(I_s) / np.diff(x_s)).mean()
    mu = g_m / (WL * 0.3 * C_g)
    print(f'R_s = {R_s:.2f} [Ohm]')


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