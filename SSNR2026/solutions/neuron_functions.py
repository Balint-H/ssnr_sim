import numpy as np


DEFAULT_LIF = dict(
    tau   = 20e-3,    # membrane time constant (s)
    el    = -60e-3,   # leak potential (V)
    vr    = -70e-3,   # reset potential (V)
    vth   = -50e-3,   # spike threshold (V)
    r     = 100e6,    # membrane resistance (ohm)
    t_ref = 10e-3,    # refractory period (s)
)

def generate_sinusoidal_input(t_range, i_0, T):
    return i_0 * (1 + np.sin(2 * np.pi * t_range / T))

def generate_constant_input(t_range, i_0):
    return np.full(len(t_range), i_0)


def build_mu_mapping(n_muscles, n_per_muscle, drive_scale=1.0, seed=11):
    n_neurons = n_muscles * n_per_muscle
    assignment = np.repeat(np.eye(n_muscles), n_per_muscle, axis=0)   

    rng = np.random.default_rng(seed)
    weights = np.sqrt(np.abs(rng.standard_normal((n_neurons, 1))))    
    mu_mapping = drive_scale * assignment * weights

    return assignment, mu_mapping


def init_pool_state(n_neurons, n_muscles, el=DEFAULT_LIF["el"], seed=42):
    """Initial state of the pool: all neurons at rest, no past spikes, zero excitation."""
      
    v_state    = np.full(n_neurons, el)
    last_spike = np.full(n_neurons, -np.inf)
    E_state    = np.zeros(n_muscles)
    return v_state, last_spike, E_state



def force_to_current(F_target, F_max=1.0, I_min=0.0, I_max=5e-10):
    drive = np.clip(F_target / F_max, 0.0, 1.0)
    return I_min + drive * (I_max - I_min)


   
def simulate_membrane(v, i, el=-60e-3, tau=20e-3, r=100e6, dt=1e-3):
    return v + dt / tau * (el - v + r * i)

def detect_spike(v, vth=-50e-3, vr=-70e-3):
    if v >= vth:
        return vr, 1
    return v, 0

def detect_spike_refractory(v, i, t, last_spike_time, t_ref=10e-3,
                             vth=-50e-3, vr=-70e-3, el=-60e-3, tau=20e-3, r=100e6, dt=1e-3):

    if t - last_spike_time < t_ref:
        return vr, 0, last_spike_time
    v = simulate_membrane(v, i, el=el, tau=tau, r=r, dt=dt)
    v, spiked = detect_spike(v, vth=vth, vr=vr)
    if spiked:
        last_spike_time = t
    return v, spiked, last_spike_time



def detect_spike_refractory_pool(v_state, last_spike, t_now, dt, I_per_neuron,
                            params=DEFAULT_LIF):
    tau, el, vr, vth, r, t_ref = (params["tau"], params["el"], params["vr"],
                                  params["vth"], params["r"], params["t_ref"])

    refractory = (t_now - last_spike) < t_ref
    v_int = simulate_membrane(v_state, I_per_neuron, el=el, tau=tau, r=r, dt=dt)
    v_state = np.where(refractory, vr, v_int)
    spiked = v_state >= vth
    v_state = np.where(spiked, vr, v_state)
    last_spike = np.where(spiked, t_now, last_spike)

    return spiked.astype(np.float32), v_state, last_spike



def update_excitation(spikes, E_prev, mu_mapping, dt, tau_exc=20e-3):
    musc_input = mu_mapping.T @ spikes         
    alpha = dt / tau_exc
    E_new = (1.0 - alpha) * E_prev + alpha * musc_input
    return np.maximum(E_new, 0.0)   



def build_pool(i_trace, N, noise_std,dt=1e-3, tau=20e-3,
                 el=-60e-3, vr=-70e-3, vth=-50e-3,
                 r=100e6, t_ref=10e-3):
    step_end = len(i_trace)
    spike_trains_pool = np.zeros((N, step_end))

    for n in range(N):
        v = el
        last_spike_time = -np.inf
        for step, i_common in enumerate(i_trace):
            t = step * dt
            i = np.clip(i_common + np.random.randn() * noise_std, 0, None)
            v, spiked, last_spike_time = detect_spike_refractory(v, i, t, last_spike_time, t_ref)
            spike_trains_pool[n, step] = spiked

    return spike_trains_pool


def compute_cst(spike_trains_pool, fs, cutoff=30, order=4):
    from scipy.signal import butter, filtfilt
    cst = spike_trains_pool.sum(axis=0)
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return cst, filtfilt(b, a, cst)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())