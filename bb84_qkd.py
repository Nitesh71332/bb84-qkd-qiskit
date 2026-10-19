"""
BB84 Quantum Key Distribution simulation under channel noise and eavesdropping.

Educational Qiskit/Aer simulation. This is not production QKD: it does not
implement authenticated classical communication, error reconciliation, or
privacy amplification. The 11% decision threshold is illustrative, not a
universal security guarantee.
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 10_000
SAMPLE_FRAC = 0.25
QBER_THRESHOLD = 0.11  # Illustrative threshold only
NOISE_LEVELS = [i / 100 for i in range(0, 21, 2)]
TRIALS = 10
RESULTS_DIR = "results"
EVE_FRACTIONS = [i / 10 for i in range(11)]

rng = np.random.default_rng()


def set_seed(seed: Optional[int]) -> None:
    """Set random seed for reproducible classical choices."""
    global rng
    rng = np.random.default_rng(seed)


def validate_inputs(n: int, noise: float, eve_fraction: float) -> None:
    """Validate one BB84 simulation's parameters."""
    if not isinstance(n, (int, np.integer)) or n < 10:
        raise ValueError("n must be an integer >= 10.")
    if not np.isfinite(noise) or not 0.0 <= noise <= 0.20:
        raise ValueError("noise must be between 0.0 and 0.20.")
    if not np.isfinite(eve_fraction) or not 0.0 <= eve_fraction <= 1.0:
        raise ValueError("eve_fraction must be between 0.0 and 1.0.")


def validate_experiment(trials: int, n: int) -> None:
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be an integer >= 1.")
    if not isinstance(n, int) or n < 10:
        raise ValueError("n must be an integer >= 10.")


def make_backend(noise: float) -> AerSimulator:
    """Build Aer backend with depolarizing error on the channel marker gate."""
    noise_model = None
    if noise > 0:
        noise_model = NoiseModel()
        error = depolarizing_error(min(2 * noise, 1.0), 1)
        noise_model.add_all_qubit_quantum_error(error, ["id"])
    return AerSimulator(noise_model=noise_model)


def build_circuit(bit: int, prep_basis: int, meas_basis: int, noisy: bool) -> QuantumCircuit:
    """Prepare, transmit, and measure one BB84 qubit."""
    circuit = QuantumCircuit(1, 1)
    if bit:
        circuit.x(0)
    if prep_basis:  # X basis
        circuit.h(0)
    circuit.barrier()
    if noisy:
        circuit.id(0)
    circuit.barrier()
    if meas_basis:  # rotate X-basis measurement to Z basis
        circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def transmit(bits, prep_bases, meas_bases, backend, noisy: bool) -> np.ndarray:
    """Simulate qubits grouped into the eight possible circuit types."""
    bits = np.asarray(bits, dtype=int)
    prep_bases = np.asarray(prep_bases, dtype=int)
    meas_bases = np.asarray(meas_bases, dtype=int)
    out = np.zeros(len(bits), dtype=int)
    if len(bits) == 0:
        return out

    keys = bits * 4 + prep_bases * 2 + meas_bases
    for key in np.unique(keys):
        indices = np.where(keys == key)[0]
        bit = (int(key) >> 2) & 1
        prep_basis = (int(key) >> 1) & 1
        meas_basis = int(key) & 1
        circuit = transpile(
            build_circuit(bit, prep_basis, meas_basis, noisy),
            backend,
            optimization_level=0,
        )
        memory = backend.run(
            circuit, shots=len(indices), memory=True
        ).result().get_memory()
        out[indices] = np.array([int(value) for value in memory], dtype=int)
    return out


def wilson_interval(errors: int, sample_size: int, z: float = 1.96):
    """Approximate 95% Wilson interval for a binomial proportion."""
    if sample_size <= 0:
        return float("nan"), float("nan")
    p = errors / sample_size
    denominator = 1 + z**2 / sample_size
    center = (p + z**2 / (2 * sample_size)) / denominator
    margin = (
        z
        * np.sqrt(p * (1 - p) / sample_size + z**2 / (4 * sample_size**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def run_bb84(
    n: int = N_QUBITS,
    noise: float = 0.0,
    eve: bool = False,
    eve_fraction: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Run one BB84 simulation.

    qber_est is estimated from the publicly revealed sample.
    qber_true is simulator-only and uses all sifted bits; it is not available
    to Alice and Bob in a real protocol.
    """
    validate_inputs(n, noise, eve_fraction)
    backend = make_backend(noise)
    noisy = noise > 0

    alice_bits = rng.integers(0, 2, n)
    alice_bases = rng.integers(0, 2, n)
    bob_bases = rng.integers(0, 2, n)

    if not eve or eve_fraction == 0:
        bob_bits = transmit(alice_bits, alice_bases, bob_bases, backend, noisy)
    else:
        intercepted = rng.random(n) < eve_fraction
        bob_bits = np.zeros(n, dtype=int)
        untouched = ~intercepted
        bob_bits[untouched] = transmit(
            alice_bits[untouched],
            alice_bases[untouched],
            bob_bases[untouched],
            backend,
            noisy,
        )

        eve_bases = rng.integers(0, 2, n)
        # Eve measures Alice's qubit. The noisy channel is applied on Alice->Eve.
        eve_bits = transmit(
            alice_bits[intercepted],
            alice_bases[intercepted],
            eve_bases[intercepted],
            backend,
            noisy,
        )
        # Eve resends a freshly prepared qubit; channel noise also affects Eve->Bob.
        bob_bits[intercepted] = transmit(
            eve_bits,
            eve_bases[intercepted],
            bob_bases[intercepted],
            backend,
            noisy,
        )

    keep = alice_bases == bob_bases
    alice_sifted = alice_bits[keep]
    bob_sifted = bob_bits[keep]
    sifted_len = len(alice_sifted)

    if sifted_len == 0:
        result = {
            "noise": noise,
            "eve": bool(eve and eve_fraction > 0),
            "eve_fraction": eve_fraction if eve else 0.0,
            "sifted": 0,
            "sample_size": 0,
            "qber_est": float("nan"),
            "qber_ci_low": float("nan"),
            "qber_ci_high": float("nan"),
            "qber_true": float("nan"),
            "agreement": float("nan"),
            "candidate_key_len": 0,
            "accepted_key_len": 0,
            "accepted": False,
        }
        if verbose:
            print("No sifted bits were available; run rejected.")
        return result

    sample_size = min(sifted_len, max(1, int(SAMPLE_FRAC * sifted_len)))
    sample_indices = rng.choice(sifted_len, sample_size, replace=False)
    sample_mask = np.zeros(sifted_len, dtype=bool)
    sample_mask[sample_indices] = True
    sample_errors = int(np.count_nonzero(
        alice_sifted[sample_mask] != bob_sifted[sample_mask]
    ))
    qber_est = sample_errors / sample_size
    ci_low, ci_high = wilson_interval(sample_errors, sample_size)
    qber_true = float(np.mean(alice_sifted != bob_sifted))

    alice_key = alice_sifted[~sample_mask]
    bob_key = bob_sifted[~sample_mask]
    agreement = float(np.mean(alice_key == bob_key)) if len(alice_key) else float("nan")
    accepted = bool(qber_est <= QBER_THRESHOLD)
    candidate_key_len = len(alice_key)
    accepted_key_len = candidate_key_len if accepted else 0

    result = {
        "noise": noise,
        "eve": bool(eve and eve_fraction > 0),
        "eve_fraction": eve_fraction if eve else 0.0,
        "sifted": sifted_len,
        "sample_size": sample_size,
        "qber_est": qber_est,
        "qber_ci_low": ci_low,
        "qber_ci_high": ci_high,
        "qber_true": qber_true,
        "agreement": agreement,
        "candidate_key_len": candidate_key_len,
        "accepted_key_len": accepted_key_len,
        "accepted": accepted,
    }

    if verbose:
        print(f"Settings              : noise={noise:.1%}, Eve={'ON' if result['eve'] else 'OFF'}")
        print(f"Qubits sent           : {n}")
        print(f"Sifted bits           : {sifted_len}")
        print(f"Sample revealed       : {sample_size}")
        print(f"Estimated QBER        : {qber_est:.2%}")
        print(f"Approx. 95% QBER CI   : [{ci_low:.2%}, {ci_high:.2%}]")
        print(f"Simulator-only QBER   : {qber_true:.2%}")
        print(f"Remaining candidate bits: {candidate_key_len}")
        print(f"Accepted key length   : {accepted_key_len}")
        print(f"Decision              : {'ACCEPT' if accepted else 'REJECT'}")
        print("Note: accepted length is a teaching metric, not a secure final key.")

    return result


def noise_experiment(trials: int = TRIALS, n: int = N_QUBITS) -> pd.DataFrame:
    """Sweep channel noise with and without Eve, averaging repeated trials."""
    validate_experiment(trials, n)
    rows = []
    for eve in (False, True):
        for noise in NOISE_LEVELS:
            for _ in range(trials):
                rows.append(run_bb84(n=n, noise=noise, eve=eve))
            print(f"Completed: Eve={'ON' if eve else 'OFF'}, noise={noise:.0%}")

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["eve", "noise"], dropna=False)
        .agg(
            estimated_qber=("qber_est", "mean"),
            estimated_qber_std=("qber_est", "std"),
            simulator_true_qber=("qber_true", "mean"),
            simulator_true_qber_std=("qber_true", "std"),
            mean_sifted_bits=("sifted", "mean"),
            mean_sample_size=("sample_size", "mean"),
            mean_key_agreement=("agreement", "mean"),
            mean_candidate_key_len=("candidate_key_len", "mean"),
            mean_accepted_key_len=("accepted_key_len", "mean"),
            acceptance_rate=("accepted", "mean"),
        )
        .reset_index()
    )
    summary["eve_label"] = summary["eve"].map(
        {False: "No Eve", True: "Eve (intercept-resend)"}
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw.to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    return summary


def eve_fraction_experiment(
    trials: int = TRIALS, n: int = N_QUBITS
) -> pd.DataFrame:
    """Sweep the fraction intercepted by Eve with no channel noise."""
    validate_experiment(trials, n)
    rows = []
    for fraction in EVE_FRACTIONS:
        for _ in range(trials):
            rows.append(
                run_bb84(
                    n=n,
                    noise=0.0,
                    eve=fraction > 0,
                    eve_fraction=fraction,
                )
            )
        print(f"Completed: Eve intercepts {fraction:.0%}")

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("eve_fraction")
        .agg(
            estimated_qber=("qber_est", "mean"),
            simulator_true_qber=("qber_true", "mean"),
            mean_accepted_key_len=("accepted_key_len", "mean"),
            acceptance_rate=("accepted", "mean"),
        )
        .reset_index()
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw.to_csv(os.path.join(RESULTS_DIR, "eve_fraction_raw.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "eve_fraction.csv"), index=False)
    return summary


def finish_plot(filename: str, show: bool) -> None:
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_noise_results(summary: pd.DataFrame, show: bool = False) -> None:
    """Create QBER, accepted-key length, and acceptance-rate plots."""
    colors = {"No Eve": "tab:blue", "Eve (intercept-resend)": "tab:red"}
    for label, group in summary.groupby("eve_label"):
        group = group.sort_values("noise")
        plt.plot(
            group["noise"] * 100,
            group["estimated_qber"] * 100,
            marker="o",
            label=label,
            color=colors[label],
        )
    plt.axhline(QBER_THRESHOLD * 100, linestyle="--", color="black", label="Illustrative 11% threshold")
    plt.xlabel("Channel noise (%)")
    plt.ylabel("Estimated QBER (%)")
    plt.title("Estimated QBER vs Channel Noise")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_plot("qber_vs_noise.png", show)

    for column, ylabel, title, filename, scale in [
        ("mean_candidate_key_len", "Mean candidate key length (bits)", "Candidate Key Length vs Channel Noise", "candidate_key_length_vs_noise.png", 1),
        ("mean_accepted_key_len", "Mean accepted key length (bits)", "Accepted Key Length vs Channel Noise", "accepted_key_length_vs_noise.png", 1),
        ("acceptance_rate", "Acceptance rate (%)", "Key Acceptance Rate vs Channel Noise", "acceptance_rate_vs_noise.png", 100),
    ]:
        for label, group in summary.groupby("eve_label"):
            group = group.sort_values("noise")
            plt.plot(
                group["noise"] * 100,
                group[column] * scale,
                marker="o",
                label=label,
                color=colors[label],
            )
        plt.xlabel("Channel noise (%)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        finish_plot(filename, show)

    # Agreement among unrevealed candidate bits is a diagnostic metric only.
    if "mean_key_agreement" in summary.columns:
        for label, group in summary.groupby("eve_label"):
            group = group.sort_values("noise")
            plt.plot(
                group["noise"] * 100,
                group["mean_key_agreement"] * 100,
                marker="o",
                label=label,
                color=colors[label],
            )
        plt.xlabel("Channel noise (%)")
        plt.ylabel("Key agreement among candidate bits (%)")
        plt.title("Candidate Key Agreement vs Channel Noise")
        plt.grid(alpha=0.3)
        plt.legend()
        finish_plot("key_agreement_vs_noise.png", show)


def plot_eve_fraction(summary: pd.DataFrame, show: bool = False) -> None:
    plt.plot(
        summary["eve_fraction"] * 100,
        summary["simulator_true_qber"] * 100,
        "o-",
        label="Simulated QBER",
    )
    plt.plot(
        summary["eve_fraction"] * 100,
        25 * summary["eve_fraction"],
        "k:",
        label="Idealized theory: 25% × intercepted fraction",
    )
    plt.axhline(QBER_THRESHOLD * 100, linestyle="--", label="Illustrative 11% threshold")
    plt.xlabel("Qubits intercepted by Eve (%)")
    plt.ylabel("Simulator-only QBER (%)")
    plt.title("QBER vs Partial Intercept-Resend Eavesdropping")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_plot("qber_vs_eve_fraction.png", show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Educational BB84 QKD simulation using Qiskit Aer")
    parser.add_argument("--demo", action="store_true", help="run one demo with and without Eve")
    parser.add_argument("--trials", type=int, default=TRIALS, help="trials averaged per setting")
    parser.add_argument("--n", type=int, default=N_QUBITS, help="qubits sent per run")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args, _ = parser.parse_known_args()

    try:
        validate_experiment(args.trials, args.n)
    except ValueError as exc:
        parser.error(str(exc))
    set_seed(args.seed)

    if args.demo:
        run_bb84(n=args.n, noise=0.05, eve=False, verbose=True)
        run_bb84(n=args.n, noise=0.05, eve=True, verbose=True)
        return

    print("Running channel-noise experiment...")
    noise_summary = noise_experiment(args.trials, args.n)
    print(noise_summary.round(4).to_string(index=False))
    plot_noise_results(noise_summary, show=not args.no_show)

    print("Running partial-eavesdropping experiment...")
    eve_summary = eve_fraction_experiment(args.trials, args.n)
    print(eve_summary.round(4).to_string(index=False))
    plot_eve_fraction(eve_summary, show=not args.no_show)
    print(f"CSV files and plots saved in ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
