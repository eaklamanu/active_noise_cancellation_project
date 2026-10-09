"""Reproducible ideal-secondary-path ANC simulation; run: python anc_lms.py."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 8000
DURATION = 5
MUS = (0.001, 0.005, 0.01, 0.05)
TAPS = (8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)
# Known causal FIR primary path, including two samples of propagation delay.
PRIMARY = np.array([0, 0, .65, .25, -.18, .12, .08, -.06,
                    .04, .03, -.02, .015, .01, -.008, .005, .003])

def make_data(seed):
    rng = np.random.default_rng(seed)
    count = FS * DURATION
    # Burn-in removes low-pass filter startup; stationary low-frequency noise.
    sos = signal.butter(4, 600, fs=FS, output='sos')
    x = signal.sosfilt(sos, rng.standard_normal(count + FS))[FS:]
    x /= np.sqrt(np.mean(x*x))
    correlated = signal.lfilter(PRIMARY, [1.0], x)
    # Independent white component: RMS 5% of correlated disturbance RMS.
    v = rng.standard_normal(count)
    v *= .05 * np.sqrt(np.mean(correlated**2) / np.mean(v**2))
    return x, correlated + v, v

def lms(x, d, length, mu):
    """y=w.T@u, e=d-y, w <- w+2*mu*e*u (factor 2 retained)."""
    w = np.zeros(length)
    u = np.zeros(length)
    e = np.empty(len(x))
    y = np.empty(len(x))
    for n in range(len(x)):
        u[1:] = u[:-1]
        u[0] = x[n]
        y[n] = w @ u
        e[n] = d[n] - y[n]
        w += 2 * mu * e[n] * u
        if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > 1e6:
            raise FloatingPointError(f'LMS diverged: taps={length}, mu={mu}, n={n}')
    return y, e, w

def metrics(d, e, v):
    # Final second is evaluation window; full record includes startup transient.
    steady = slice(4*FS, 5*FS)
    mse = np.mean(e[steady]**2)
    return dict(mse_full=float(np.mean(e**2)), mse_final=float(mse),
                input_power_final=float(np.mean(d[steady]**2)),
                output_power_final=float(mse),
                input_rms_final=float(np.sqrt(np.mean(d[steady]**2))),
                output_rms_final=float(np.sqrt(mse)),
                power_reduction_final_pct=float(100*(1-mse/np.mean(d[steady]**2))),
                rms_reduction_final_pct=float(100*(1-np.sqrt(mse/np.mean(d[steady]**2)))),
                attenuation_full_db=float(10*np.log10(np.mean(d**2)/np.mean(e**2))),
                attenuation_final_db=float(10*np.log10(np.mean(d[steady]**2)/mse)),
                floor_final=float(np.mean(v[steady]**2)))

def save_plot(root, name):
    plt.tight_layout()
    plt.savefig(root/'plots'/f'{name}.png', dpi=180)
    plt.savefig(root/'plots'/f'{name}.pdf')
    plt.close()

def main():
    root = Path(__file__).resolve().parent
    (root/'plots').mkdir(exist_ok=True)
    rows = []
    baseline = None
    curves = {}
    for seed in SEEDS:
        x, d, v = make_data(seed)
        for length in TAPS:
            for mu in MUS:
                try:
                    y, e, w = lms(x, d, length, mu)
                except FloatingPointError:
                    rows.append(dict(seed=seed,taps=length,mu=mu,status='diverged',
                        **{key:None for key in metrics(d,d,v)}))
                    continue
                rows.append(dict(seed=seed, taps=length, mu=mu,status='finite', **metrics(d,e,v)))
                if seed == 42 and length == 32:
                    curves[mu] = e.copy()
                    if mu == .001:
                        baseline = (x,d,v,y,e,w)
    with (root/'results.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    x,d,v,y,e,w = baseline
    comparisons = []
    for window, sl in [('full_record',slice(None)),('final_second',slice(4*FS,5*FS))]:
        for name, arr in [('reference_x',x),('disturbance_before_d',d),
                          ('adaptive_estimate_y',y),('residual_after_e',e)]:
            comparisons.append(dict(window=window,signal=name,
                power=float(np.mean(arr[sl]**2)),rms=float(np.sqrt(np.mean(arr[sl]**2)))))
    with (root/'signal_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]))
        writer.writeheader();writer.writerows(comparisons)
    np.savez_compressed(root/'baseline_signals.npz',x=x,d=d,v=v,y=y,e=e,w=w,primary=PRIMARY,fs=FS)
    summary = dict(fs=FS,duration=DURATION,seeds=list(SEEDS),baseline_seed=42,
                   baseline_taps=32,baseline_mu=.001,update='w += 2*mu*e*u',
                   evaluation_seconds=[4,5],baseline=metrics(d,e,v),sweep=[])
    for length in TAPS:
        for mu in MUS:
            subset=[r for r in rows if r['taps']==length and r['mu']==mu]
            finite=[r for r in subset if r['status']=='finite']
            vals=np.array([r['attenuation_final_db'] for r in finite])
            summary['sweep'].append(dict(taps=length,mu=mu,
                finite_runs=len(finite),diverged_runs=len(subset)-len(finite),
                attenuation_mean_db=float(vals.mean()) if len(vals) else None,
                attenuation_sd_db=float(vals.std(ddof=1)) if len(vals)>1 else None,
                mse_mean=float(np.mean([r['mse_final'] for r in finite])) if finite else None))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    t=np.arange(len(x))/FS
    fig,axs=plt.subplots(2,1,figsize=(8,5))
    axs[0].plot(t,d,lw=.4,label='Disturbance');axs[0].plot(t,e,lw=.4,label='Residual')
    axs[0].set(xlabel='Time (s)',ylabel='Amplitude',title='LMS ANC: 32 taps, mu = 0.001, seed 42');axs[0].legend()
    sl=slice(4*FS,4*FS+400)
    axs[1].plot(t[sl],d[sl],label='Disturbance');axs[1].plot(t[sl],e[sl],label='Residual')
    axs[1].set(xlabel='Time (s)',ylabel='Amplitude',title='Final-second detail');axs[1].legend()
    save_plot(root,'waveforms')
    # All four signals on identical axes: comparisons must not hide scale changes.
    fig, axs = plt.subplots(2,2,figsize=(9,5),sharex=True,sharey=True)
    sl=slice(4*FS,4*FS+400)
    limit=1.1*max(np.max(np.abs(arr[sl])) for arr in (x,d,y,e))
    for ax,arr,title,color in zip(axs.flat,(x,d,y,e),
            ('Reference input x(n)','Disturbance before cancellation d(n)',
             'Adaptive filter output y(n)','System output after cancellation e(n)'),
            ('tab:gray','tab:blue','tab:green','tab:orange')):
        ax.plot(t[sl],arr[sl],color=color,lw=1)
        ax.set(title=title,xlabel='Time (s)',ylabel='Amplitude',ylim=(-limit,limit))
        ax.grid(alpha=.2)
    save_plot(root,'signal_comparison')
    plt.figure(figsize=(8,4))
    for mu,err in curves.items():
        block=np.mean(err.reshape(-1,400)**2,axis=1)
        plt.semilogy((np.arange(len(block))+.5)*.05,block,label=f'mu = {mu}')
    plt.axhline(np.mean(v*v),color='black',ls='--',label='Independent noise power')
    plt.xlabel('Time (s)');plt.ylabel('50 ms block mean squared error')
    plt.title('Single-realization learning curves, 32 taps');plt.legend()
    save_plot(root,'learning_curves')
    plt.figure(figsize=(8,4))
    for arr,label in [(d,'Disturbance'),(e,'Residual'),(v,'Independent component')]:
        freq,power=signal.welch(arr[4*FS:],FS,nperseg=1024)
        plt.plot(freq,10*np.log10(np.maximum(power,1e-30)),label=label)
    plt.xlabel('Frequency (Hz)');plt.ylabel('PSD (dB / Hz, relative to unit amplitude squared)')
    plt.title('Final-second Welch power spectral density');plt.legend()
    save_plot(root,'spectra')
    plt.figure(figsize=(8,4))
    for length in TAPS:
        vals=[r for r in summary['sweep'] if r['taps']==length and r['finite_runs']==len(SEEDS)]
        plt.errorbar([r['mu'] for r in vals],[r['attenuation_mean_db'] for r in vals],
                     yerr=[r['attenuation_sd_db'] for r in vals],marker='o',capsize=4,label=f'{length} taps')
    plt.xscale('log');plt.xlabel('LMS step size mu');plt.ylabel('Final-second attenuation (dB)')
    plt.title('Five seeds: mean and sample SD; divergent settings omitted');plt.legend()
    save_plot(root,'parameter_sweep')
    # Independent implementation check and exact-model sanity check.
    target=np.pad(PRIMARY,(0,32-len(PRIMARY)))
    exact=signal.lfilter(target,[1],x)
    assert np.allclose(d-exact,v,atol=1e-12)
    assert np.allclose(d-y,e,atol=1e-12)
    assert np.isclose(metrics(d,e,v)['output_power_final'],metrics(d,e,v)['output_rms_final']**2)
    assert metrics(d,e,v)['attenuation_final_db'] > 10
    xx=np.array([1.,2.,-1.,.5]);dd=np.array([.5,1.,-.5,.25])
    _,ee,ww=lms(xx,dd,2,.01)
    refw=np.zeros(2); refe=[]
    for n in range(4):
        u=np.array([xx[n],xx[n-1] if n else 0])
        er=dd[n]-np.dot(refw,u);refe.append(er);refw=refw+.02*er*u
    assert np.allclose(ee,refe) and np.allclose(ww,refw)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
