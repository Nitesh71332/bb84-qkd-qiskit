"""
bb84_qkd.py
===========
Performance Analysis of BB84 Quantum Key Distribution under Noise and
Eavesdropping using Qiskit.

IEG 313: Quantum Computing for Engineers - Team "Bell State Bandits", IIIT Kottayam

WHAT THIS FILE DOES (big picture)
---------------------------------
Alice and Bob want a shared secret key. BB84 builds it from qubits:

    1. Alice picks random bits + random bases and sends qubits.
    2. Bob measures each qubit in a random basis.
    3. SIFTING: they publicly compare bases (never bits) and keep matches.
    4. QBER ESTIMATION: they compare a random sample of the kept bits.
       The fraction that differs is the Quantum Bit Error Rate (QBER).
    5. DETECTOR: QBER <= 11%  -> ACCEPT key,   QBER > 11% -> REJECT key.

We simulate two things that raise the QBER:
    * Eve, an eavesdropper doing an intercept-resend attack (~25% QBER)
    * Channel noise (depolarizing), 0% - 20%

FILE LAYOUT
-----------
    0. Configuration
    1. Quantum building blocks   (noisy backend, one-qubit circuit, transmit)
    2. The BB84 protocol         (run_bb84, one step per commented block)
    3. Experiments               (noise sweep, Eve-fraction sweep)
    4. Plots
    5. Command-line entry point

Basis encoding used everywhere:  0 = Z basis (|0>,|1>),  1 = X basis (|+>,|->)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ----------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------
N_QUBITS = 2000          # qubits Alice sends per run
SAMPLE_FRAC = 0.25       # fraction of the sifted key publicly revealed to estimate QBER
QBER_THRESHOLD = 0.11    # accept key if estimated QBER <= 11% (standard BB84 bound)
NOISE_LEVELS = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
TRIALS = 5               # runs averaged per (noise, Eve) setting
RESULTS_DIR = "results"

rng = np.random.default_rng()


def set_seed(seed):
    """Make a run reproducible (same random bits/bases every time)."""
    global rng
    rng = np.random.default_rng(seed)


# ----------------------------------------------------------------------------
# 1. QUANTUM BUILDING BLOCKS
# ----------------------------------------------------------------------------
def make_backend(noise):
    """
    Build the Qiskit Aer simulator.

    Channel noise is modelled as a depolarizing error on the 'id' (identity)
    gate, which our circuit applies once while the qubit is "in the channel".
    A depolarizing error with parameter lam flips the measured bit with
    probability lam/2, so we pass lam = 2*noise: then `noise` is directly the
    probability that the channel flips the qubit (e.g. noise=0.05 -> ~5% flips).
    """
    noise_model = None
    if noise > 0:
        noise_model = NoiseModel()
        err = depolarizing_error(min(2 * noise, 1.0), 1)
        noise_model.add_all_qubit_quantum_error(err, ["id"])
    return AerSimulator(noise_model=noise_model)


def build_circuit(bit, prep_basis, meas_basis, noisy):
    """
    Circuit for ONE qubit's journey:  prepare -> channel -> measure.

    PREPARE (the sender)
        bit = 1          -> X gate        : |0> becomes |1>
        prep_basis = X   -> H gate        : moves the state into the X basis
        Result: Z basis gives |0>,|1>;  X basis gives |+>,|->.
    CHANNEL
        id gate (this is where Aer injects the depolarizing noise)
    MEASURE (the receiver)
        meas_basis = X   -> H gate first  : rotates X basis back to Z
        then measure in Z (all hardware measures in Z).
    If meas_basis == prep_basis the result is exactly `bit` (without noise);
    if they differ the result is a random 0/1.
    """
    qc = QuantumCircuit(1, 1)
    if bit:
        qc.x(0)
    if prep_basis:
        qc.h(0)
    qc.barrier()
    if noisy:
        qc.id(0)
    qc.barrier()
    if meas_basis:
        qc.h(0)
    qc.measure(0, 0)
    return qc


def transmit(bits, prep_bases, meas_bases, backend, noisy):
    """
    Send many qubits through Qiskit and return what was measured.

    Every qubit's circuit depends only on (bit, prep_basis, meas_basis), so
    there are just 8 distinct circuits. We run each one ONCE with
    shots = (number of qubits needing it) instead of building 2000 circuits.
    memory=True returns the individual shot results, so each qubit still gets
    its own independent random measurement outcome.
    """
    out = np.zeros(len(bits), dtype=int)
    keys = bits * 4 + prep_bases * 2 + meas_bases       # unique id per circuit type
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]                    # qubits using this circuit
        b, pb, mb = (k >> 2) & 1, (k >> 1) & 1, k & 1
        qc = transpile(build_circuit(b, pb, mb, noisy), backend, optimization_level=0)
        shots = backend.run(qc, shots=len(idx), memory=True).result().get_memory()
        out[idx] = np.array([int(s) for s in shots])
    return out


# ----------------------------------------------------------------------------
# 2. THE BB84 PROTOCOL
# ----------------------------------------------------------------------------
def run_bb84(n=N_QUBITS, noise=0.0, eve=False, eve_fraction=1.0, verbose=False):
    """
    One complete BB84 run. Returns a dict of measured quantities.

    n             : number of qubits Alice sends
    noise         : channel noise level (0 - 0.2), probability of a flip
    eve           : is an eavesdropper present?
    eve_fraction  : share of qubits Eve intercepts (1.0 = all of them)
    """
    backend = make_backend(noise)
    noisy = noise > 0

    # ---- STEP 1: Alice's random bits and random bases ---------------------
    a_bits = rng.integers(0, 2, n)      # the secret bits she wants to share
    a_bases = rng.integers(0, 2, n)     # 0 = Z, 1 = X : how she encodes each bit

    # ---- STEP 2: Bob's random measurement bases ---------------------------
    b_bases = rng.integers(0, 2, n)

    # ---- STEP 3: quantum transmission -------------------------------------
    if not eve:
        # Alice -> Bob directly.
        b_bits = transmit(a_bits, a_bases, b_bases, backend, noisy)
    else:
        # Eve intercepts a random subset of the qubits (all of them by default).
        hit = rng.random(n) < eve_fraction
        b_bits = np.zeros(n, dtype=int)

        # (a) qubits Eve does NOT touch: Alice -> Bob directly
        free = ~hit
        b_bits[free] = transmit(a_bits[free], a_bases[free], b_bases[free], backend, noisy)

        # (b) qubits Eve intercepts (intercept-resend attack):
        #     Eve guesses a random basis and MEASURES (she cannot copy a qubit:
        #     no-cloning theorem). Wrong basis -> her result is random and the
        #     qubit is disturbed. She then RE-PREPARES a new qubit with the bit
        #     she read, in the basis she used, and sends it to Bob.
        e_bases = rng.integers(0, 2, n)
        e_bits = transmit(a_bits[hit], a_bases[hit], e_bases[hit], backend, noisy=False)
        b_bits[hit] = transmit(e_bits, e_bases[hit], b_bases[hit], backend, noisy)

    # ---- STEP 4: SIFTING --------------------------------------------------
    # Over the public channel they announce BASES only. Keep positions where
    # the bases match; there Bob's bit should equal Alice's. Others are random.
    keep = a_bases == b_bases
    a_sift, b_sift = a_bits[keep], b_bits[keep]
    sifted_len = len(a_sift)

    # ---- STEP 5: QBER ESTIMATION ------------------------------------------
    # Reveal a random sample of the sifted bits and count mismatches.
    # Those revealed bits are public now, so they are discarded.
    sample_size = max(1, int(SAMPLE_FRAC * sifted_len))
    sample_idx = rng.choice(sifted_len, sample_size, replace=False)
    in_sample = np.zeros(sifted_len, dtype=bool)
    in_sample[sample_idx] = True
    qber_est = float(np.mean(a_sift[in_sample] != b_sift[in_sample]))   # what Alice & Bob see
    qber_true = float(np.mean(a_sift != b_sift))                        # over the whole sifted key

    # ---- STEP 6: THRESHOLD DETECTOR ---------------------------------------
    accepted = qber_est <= QBER_THRESHOLD

    # ---- STEP 7: FINAL KEY ------------------------------------------------
    # Remaining (unrevealed) sifted bits. Used only if the key is accepted.
    a_key, b_key = a_sift[~in_sample], b_sift[~in_sample]
    agreement = float(np.mean(a_key == b_key)) if len(a_key) else float("nan")

    if verbose:
        _print_report(n, a_bits, a_bases, b_bases, b_bits, sifted_len, sample_size,
                      qber_est, agreement, len(a_key), accepted, noise, eve, eve_fraction)

    return dict(noise=noise, eve=eve, eve_fraction=eve_fraction, sifted=sifted_len,
                qber_est=qber_est, qber_true=qber_true, agreement=agreement,
                raw_final_len=len(a_key), final_len=len(a_key) if accepted else 0,
                accepted=accepted)


def _print_report(n, a_bits, a_bases, b_bases, b_bits, sifted, sample, qber, agree,
                  final_len, accepted, noise, eve, eve_fraction):
    """Human-readable summary of one run (used for demos)."""
    name = {0: "Z", 1: "X"}
    print(f"Settings            : noise={noise:.0%}, Eve={'ON (intercepts %d%%)' % (eve_fraction*100) if eve else 'OFF'}")
    print("First 10 qubits     :")
    print("   #  Alice bit  A-basis  B-basis  Bob bit  match?")
    for i in range(10):
        m = "yes" if a_bases[i] == b_bases[i] else "no "
        print(f"  {i+1:2d}      {a_bits[i]}         {name[a_bases[i]]}        {name[b_bases[i]]}       {b_bits[i]}      {m}")
    print(f"Qubits sent         : {n}")
    print(f"Sifted key length   : {sifted}   (bases matched)")
    print(f"Sample revealed     : {sample}")
    print(f"Estimated QBER      : {qber:.2%}")
    print(f"Key agreement       : {agree:.2%}")
    print(f"Final key length    : {final_len}")
    print(f"Decision            : {'ACCEPT' if accepted else 'REJECT (disturbance detected)'}\n")


# ----------------------------------------------------------------------------
# 3. EXPERIMENTS
# ----------------------------------------------------------------------------
def noise_experiment(trials=TRIALS, n=N_QUBITS):
    """Main experiment: sweep noise 0-20%, with and without Eve, average trials."""
    rows = []
    for eve in (False, True):
        for p in NOISE_LEVELS:
            for _ in range(trials):
                rows.append(run_bb84(n=n, noise=p, eve=eve))
            print(f"  done: Eve={'ON ' if eve else 'OFF'} noise={p:.0%}")
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["eve", "noise"])
                  .agg(QBER=("qber_true", "mean"), QBER_std=("qber_true", "std"),
                       agreement=("agreement", "mean"), sifted=("sifted", "mean"),
                       final_key_len=("raw_final_len", "mean"),
                       accept_rate=("accepted", "mean"))
                  .reset_index())
    summary["eve"] = summary["eve"].map({False: "No Eve", True: "Eve (intercept-resend)"})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw.to_csv(f"{RESULTS_DIR}/raw_results.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/summary.csv", index=False)
    return summary


def eve_fraction_experiment(trials=TRIALS, n=N_QUBITS):
    """Extension: Eve intercepts only a fraction f of the qubits (no noise).
    Theory: QBER ~ 25% * f, so a careful Eve can stay under the threshold."""
    fractions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for f in fractions:
        for _ in range(trials):
            rows.append(run_bb84(n=n, noise=0.0, eve=f > 0, eve_fraction=f))
        print(f"  done: Eve intercepts {f:.0%}")
    df = pd.DataFrame(rows)
    out = (df.groupby("eve_fraction")
             .agg(QBER=("qber_true", "mean"), accept_rate=("accepted", "mean"))
             .reset_index())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(f"{RESULTS_DIR}/eve_fraction.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------------
COLORS = {"No Eve": "tab:blue", "Eve (intercept-resend)": "tab:red"}


def _finish(fname, show):
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{RESULTS_DIR}/{fname}", dpi=200)
    plt.show() if show else plt.close()


def plot_noise_results(summary, show=False):
    """4 graphs: QBER, key agreement, final key length, detector acceptance."""
    # QBER vs noise (with the 11% threshold line)
    plt.figure(figsize=(6, 4))
    for name, g in summary.groupby("eve"):
        plt.errorbar(g.noise * 100, g.QBER * 100, yerr=g.QBER_std * 100,
                     marker="o", capsize=3, label=name, color=COLORS[name])
    plt.axhline(QBER_THRESHOLD * 100, ls="--", color="k", label="11% threshold")
    plt.xlabel("Channel noise (%)"); plt.ylabel("QBER (%)")
    plt.title("QBER vs channel noise"); plt.legend(); plt.grid(alpha=.3)
    _finish("qber_vs_noise.png", show)

    for col, ylabel, title, fname, scale in [
        ("agreement", "Key agreement (%)", "Key agreement vs channel noise", "key_agreement_vs_noise.png", 100),
        ("final_key_len", "Final key length (bits)", "Final key length (after sifting + sampling)", "final_key_length.png", 1),
        ("accept_rate", "Key accepted (%)", "Threshold detector: acceptance rate", "detector_accept_rate.png", 100),
    ]:
        plt.figure(figsize=(6, 4))
        for name, g in summary.groupby("eve"):
            plt.plot(g.noise * 100, g[col] * scale, marker="o", label=name, color=COLORS[name])
        plt.xlabel("Channel noise (%)"); plt.ylabel(ylabel)
        plt.title(title); plt.legend(); plt.grid(alpha=.3)
        _finish(fname, show)


def plot_eve_fraction(df, show=False):
    """QBER vs fraction of qubits Eve intercepts, with theory line 25% * f."""
    plt.figure(figsize=(6, 4))
    plt.plot(df.eve_fraction * 100, df.QBER * 100, "o-", color="tab:red", label="Simulated")
    plt.plot(df.eve_fraction * 100, 25 * df.eve_fraction, "k:", label="Theory: 25% x fraction")
    plt.axhline(QBER_THRESHOLD * 100, ls="--", color="gray", label="11% threshold")
    plt.xlabel("Qubits intercepted by Eve (%)"); plt.ylabel("QBER (%)")
    plt.title("Partial eavesdropping"); plt.legend(); plt.grid(alpha=.3)
    _finish("qber_vs_eve_fraction.png", show)


# ----------------------------------------------------------------------------
# 5. COMMAND-LINE ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BB84 QKD simulation (Qiskit)")
    ap.add_argument("--demo", action="store_true", help="single verbose run, with and without Eve")
    ap.add_argument("--trials", type=int, default=TRIALS, help="runs averaged per setting")
    ap.add_argument("--n", type=int, default=N_QUBITS, help="qubits Alice sends per run")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args, _ = ap.parse_known_args()     # parse_known_args: also safe inside Jupyter

    if args.seed is not None:
        set_seed(args.seed)

    if args.demo:
        run_bb84(n=args.n, noise=0.05, eve=False, verbose=True)
        run_bb84(n=args.n, noise=0.05, eve=True, verbose=True)
    else:
        print("Noise experiment...")
        summary = noise_experiment(args.trials, args.n)
        print(summary.round(3).to_string(index=False))
        plot_noise_results(summary)
        print("Eve-fraction experiment...")
        plot_eve_fraction(eve_fraction_experiment(args.trials, args.n))
        print(f"\nSaved CSVs and graphs to ./{RESULTS_DIR}/")