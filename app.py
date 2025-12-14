# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- Physical constants (SI) ---
hbar = 1.054_571_817e-34  # J·s
m_e  = 9.109_383_7015e-31 # kg
eV   = 1.602_176_634e-19  # J

st.set_page_config(page_title="Quantum Tunneling (Rectangular Barrier)", layout="wide")

st.title("Quantum Tunneling")
st.write(
    "Incident electron from the left. Region I: x<0, Region II: 0≤x≤a (barrier), Region III: x>a."
)

# --- Sidebar controls ---
st.sidebar.header("Parameters")

E_eV  = st.sidebar.slider("Electron energy E (eV)", 0.001, 20.0, 2.0, 0.001)
V0_eV = st.sidebar.slider("Barrier height V₀ (eV)", 0.0,  20.0, 5.0, 0.001)
a_nm  = st.sidebar.slider("Barrier width a (nm)",  0.01,  3.0, 1.0, 0.01)

Lmult = st.sidebar.slider("Plot padding (× a on each side)", 1.0, 10.0, 4.0, 0.5)
Npts  = st.sidebar.slider("Number of x points", 400, 5000, 1600, 100)

plot_mode = st.sidebar.selectbox(
    "Plot",
    ["Re ψ(x)", "|ψ(x)|²", "Re/Im ψ(x)", "Both (|ψ|² and Re/Im)"]
)

# --- Convert units ---
E  = E_eV * eV
V0 = V0_eV * eV
a  = a_nm * 1e-9  # m

if E <= 0:
    st.error("E must be > 0.")
    st.stop()

# Wave numbers
k = np.sqrt(2*m_e*E) / hbar  # outside barrier, real

# Inside barrier: q may be real (E>V0) or imaginary (E<V0), we keep it complex
q_sq = 2*m_e*(E - V0) / (hbar**2)
q = np.sqrt(q_sq + 0j)  # force complex

# --- Solve scattering via boundary matching ---
# Regions:
# I:  psi1 = exp(ikx) + r exp(-ikx)
# II: psi2 = A exp(iqx) + B exp(-iqx)
# III:psi3 = t exp(ikx)
#
# Match psi and dpsi/dx at x=0 and x=a.

ika = 1j * k * a
iqa = 1j * q * a

exp_iqa  = np.exp(iqa)
exp_miqa = np.exp(-iqa)
exp_ika  = np.exp(ika)

# Unknown vector: [r, A, B, t]
M = np.zeros((4, 4), dtype=complex)
b = np.zeros(4, dtype=complex)

# (1) 1 + r = A + B  -> r - A - B = -1
M[0, 0] = 1
M[0, 1] = -1
M[0, 2] = -1
M[0, 3] = 0
b[0]    = -1

# (2) ik(1 - r) = iq(A - B) -> -ik r - iq A + iq B = -ik
M[1, 0] = -1j * k
M[1, 1] = -1j * q
M[1, 2] =  1j * q
M[1, 3] = 0
b[1]    = -1j * k

# (3) A e^{iqa} + B e^{-iqa} = t e^{ika} -> A e^{iqa} + B e^{-iqa} - t e^{ika} = 0
M[2, 0] = 0
M[2, 1] = exp_iqa
M[2, 2] = exp_miqa
M[2, 3] = -exp_ika
b[2]    = 0

# (4) iq(A e^{iqa} - B e^{-iqa}) = ik t e^{ika}
# -> iq A e^{iqa} - iq B e^{-iqa} - ik t e^{ika} = 0
M[3, 0] = 0
M[3, 1] = 1j * q * exp_iqa
M[3, 2] = -1j * q * exp_miqa
M[3, 3] = -1j * k * exp_ika
b[3]    = 0

try:
    r, Acoef, Bcoef, t = np.linalg.solve(M, b)
except np.linalg.LinAlgError:
    st.error("Linear system became singular for these parameters (rare edge case). Try slightly different values.")
    st.stop()

# --- Reflection & Transmission ---
# Flux ratio: T = (k_right/k_left)*|t|^2 ; here k_right = k_left => T = |t|^2
R = float(np.abs(r)**2)
T = float(np.abs(t)**2)
RT_sum = R + T

# --- Build x grid and wavefunction in 3 zones ---
L = Lmult * a
x_min = -L
x_max = a + L
x = np.linspace(x_min, x_max, Npts)

psi = np.zeros_like(x, dtype=complex)

mask1 = x < 0
mask2 = (x >= 0) & (x <= a)
mask3 = x > a

psi[mask1] = np.exp(1j * k * x[mask1]) + r * np.exp(-1j * k * x[mask1])
psi[mask2] = Acoef * np.exp(1j * q * x[mask2]) + Bcoef * np.exp(-1j * q * x[mask2])
psi[mask3] = t * np.exp(1j * k * x[mask3])

prob = np.abs(psi)**2

# --- Layout: results + plots ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Results")
    st.metric("Transmission T", f"{T:.6f}")
    st.metric("Reflection R", f"{R:.6f}")
    st.metric("R + T", f"{RT_sum:.6f}")
    st.caption("Ideally R+T ≈ 1 (small numerical deviation may appear).")

    st.subheader("Computed coefficients")
    st.write(f"r = {r:.3g}")
    st.write(f"t = {t:.3g}")
    st.write(f"A = {Acoef:.3g}")
    st.write(f"B = {Bcoef:.3g}")

    st.subheader("Wave numbers")
    st.write(f"k (outside) = {k:.3e} 1/m")
    st.write(f"q (inside)  = {q:.3e} 1/m")
    if E_eV < V0_eV:
        st.info("E < V0: inside the barrier the wave decays (q is imaginary).")
    elif E_eV > V0_eV:
        st.info("E > V0: inside the barrier the wave oscillates (q is real).")
    else:
        st.warning("E ≈ V0: threshold case; expect more sensitivity / numerical quirks.")

with col2:
    st.subheader("Wavefunction visualization")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Barrier region markers
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axvline(a, linestyle="--", linewidth=1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Amplitude / Probability density")

    if plot_mode == "|ψ(x)|²":
        ax.plot(x, prob, label="|ψ(x)|²")
        ax.set_ylabel("|ψ(x)|²")
        ax.legend()

    elif plot_mode == "Re ψ(x)":
        ax.plot(x, np.real(psi), label="Re[ψ(x)]")
        ax.set_ylabel("Re ψ(x)")
        ax.legend()
        
    elif plot_mode == "Re/Im ψ(x)":
        ax.plot(x, np.real(psi), label="Re[ψ(x)]")
        ax.plot(x, -np.imag(psi), label="-Im[ψ(x)]")
        ax.set_ylabel("Re/Im ψ(x)")
        ax.legend()

    else:
        # Normalize curves for a clean combined plot (optional but helpful)
        # Keep scaling stable even if prob is huge near interference peaks.
        prob_scale = np.max(prob) if np.max(prob) > 0 else 1.0
        reim_scale = np.max(np.abs(psi)) if np.max(np.abs(psi)) > 0 else 1.0

        ax.plot(x, prob / prob_scale, label="|ψ(x)|² (normalized)")
        ax.plot(x, np.real(psi) / reim_scale, label="Re[ψ(x)] (normalized)")
        ax.plot(x, -np.imag(psi) / reim_scale, label="-Im[ψ(x)] (normalized)")
        ax.set_ylabel("Normalized curves")
        ax.legend()

    ax.set_title(f"Rectangular barrier: V0={V0_eV:.3f} eV, a={a_nm:.3f} nm, E={E_eV:.3f} eV")
    st.pyplot(fig)

