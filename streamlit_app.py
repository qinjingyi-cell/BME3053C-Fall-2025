from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Fourier Transform Learning Lab", layout="wide")

# -----------------------------
# Helper Data and Defaults
# -----------------------------
DEFAULT_FS = 2048
DEFAULT_DURATION = 1.5
DEFAULT_SAMPLES = 2048
MAX_COMPONENTS = 8
MIN_COMPONENTS = 1

PRESETS = {
    "None": {
        "description": "Start from scratch. Only the components you enable in the sidebar contribute to the signal.",
        "latex": "0",
        "generator": lambda t: np.zeros_like(t)
    },
    "Square Wave": {
        "description": "Square waves switch abruptly between high and low values, so their spectra contain strong odd harmonics that decay slowly.",
        "latex": r"\text{square}(t)",
        "generator": lambda t: signal.square(2 * np.pi * 5 * t)
    },
    "Triangle Wave": {
        "description": "Triangle waves change linearly, producing only odd harmonics whose amplitudes decay with 1/f^2, leading to a smoother spectrum than a square wave.",
        "latex": r"\text{triangle}(t)",
        "generator": lambda t: signal.sawtooth(2 * np.pi * 5 * t, width=0.5)
    },
    "Sawtooth Wave": {
        "description": "Sawtooth waves ramp up and drop sharply, introducing both even and odd harmonics that decay with 1/f, so the spectrum is dense.",
        "latex": r"\text{sawtooth}(t)",
        "generator": lambda t: signal.sawtooth(2 * np.pi * 5 * t)
    },
    "Gaussian Pulse": {
        "description": "Gaussian pulses are smooth bumps in time whose spectra are also Gaussian—broad pulses in time have narrow spectra and vice versa.",
        "latex": r"e^{-t^2}",
        "generator": lambda t: signal.gausspulse(t - DEFAULT_DURATION / 2, fc=15)
    },
    "Step": {
        "description": "Step functions jump from one value to another, creating a 1/f-type spectral roll-off that reflects the sharp edge.",
        "latex": r"u(t)",
        "generator": lambda t: np.heaviside(t - DEFAULT_DURATION / 2, 1.0)
    },
    "Impulse/Spike": {
        "description": "Short impulses are extremely narrow in time, so their spectra spread energy across almost all frequencies uniformly.",
        "latex": r"\delta(t)",
        "generator": lambda t: np.exp(-((t - DEFAULT_DURATION / 2) ** 2) / 0.0005)
    },
    "Chirp": {
        "description": "Chirps change frequency over time, producing diagonal spectral ridges that trace the instantaneous frequency trajectory.",
        "latex": r"\text{chirp}(t)",
        "generator": lambda t: signal.chirp(t, f0=2, f1=40, t1=DEFAULT_DURATION, method="linear")
    }
}

WINDOW_METADATA = {
    "Rectangular": {"main_lobe": "2 bins", "sidelobe": "-13 dB"},
    "Hann": {"main_lobe": "4 bins", "sidelobe": "-31 dB"},
    "Hamming": {"main_lobe": "4 bins", "sidelobe": "-43 dB"},
    "Blackman": {"main_lobe": "6 bins", "sidelobe": "-58 dB"}
}


def estimate_kaiser_metrics(beta: float) -> dict:
    sidelobe = -20 * np.log10(np.i0(beta)) if beta > 0 else -13
    main_lobe = f"{2 + int(beta // 3)} bins"
    return {"main_lobe": main_lobe, "sidelobe": f"{sidelobe:.1f} dB"}


# -----------------------------
# Helper Functions
# -----------------------------
def generate_time_vector(fs: float, duration: float, num_samples: int | None = None) -> np.ndarray:
    """Return an evenly spaced time vector."""
    if num_samples is None or num_samples <= 0:
        num_samples = max(int(fs * duration), 2)
    return np.linspace(0, duration, num_samples, endpoint=False)


def build_signal(components: list[dict], preset: str, t: np.ndarray) -> tuple[np.ndarray, str, list[np.ndarray], np.ndarray]:
    base = np.zeros_like(t)
    comp_signals: list[np.ndarray] = []
    latex_terms: list[str] = []

    for comp in components:
        amp = comp["amplitude"]
        freq = comp["frequency"]
        phase = comp["phase"]
        enabled = comp["enabled"]
        waveform = comp["waveform"]
        oscillator = np.sin if waveform == "sine" else np.cos
        func_symbol = "\\sin" if waveform == "sine" else "\\cos"
        signal_piece = amp * oscillator(2 * np.pi * freq * t + phase)
        comp_signals.append(signal_piece if enabled else np.zeros_like(t))
        if enabled:
            base += signal_piece
            latex_terms.append(rf"{amp:.2f}\,{func_symbol}(2\pi\cdot{freq:.2f} t + {phase:.2f})")

    preset_signal = PRESETS[preset]["generator"](t)
    base += preset_signal
    if preset != "None":
        latex_terms.append(PRESETS[preset]["latex"])

    latex_expression = "x(t) = " + " + ".join(latex_terms) if latex_terms else "x(t) = 0"
    return base, latex_expression, comp_signals, preset_signal


def compute_window(name: str, N: int, beta: float = 8.6) -> tuple[np.ndarray, dict]:
    if name == "Rectangular":
        win = np.ones(N)
    elif name == "Hann":
        win = np.hanning(N)
    elif name == "Hamming":
        win = np.hamming(N)
    elif name == "Blackman":
        win = np.blackman(N)
    elif name == "Kaiser":
        win = np.kaiser(N, beta)
    else:
        win = np.ones(N)

    if name == "Kaiser":
        meta = estimate_kaiser_metrics(beta)
    else:
        meta = WINDOW_METADATA.get(name, {"main_lobe": "-", "sidelobe": "-"})
    return win, meta


def fft_spectrum(x: np.ndarray, fs: float, window: np.ndarray | None = None, zero_pad_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    N = len(x)
    window_vals = window if window is not None else np.ones(N)
    xw = x * window_vals
    if zero_pad_factor > 1:
        pad = int((zero_pad_factor - 1) * N)
        xw = np.pad(xw, (0, pad))
    X = np.fft.fft(xw)
    freqs = np.fft.fftfreq(len(X), 1 / fs)
    return freqs, X


def reconstruct_band(X: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    X_band = np.zeros_like(X)
    X_band[mask] = X[mask]
    return np.fft.ifft(X_band).real


def describe_aliasing(components: list[dict], fs: float) -> list[str]:
    nyquist = fs / 2
    explanations = []
    for idx, comp in enumerate(components):
        freq = comp["frequency"]
        if freq > nyquist:
            alias = abs(((freq + nyquist) % fs) - nyquist)
            explanations.append(
                f"Component {idx + 1} at {freq:.1f} Hz exceeds Nyquist ({nyquist:.1f} Hz) and appears as {alias:.1f} Hz in the sampled data."
            )
    return explanations


def ensure_session_state():
    if "components" not in st.session_state:
        st.session_state.components = [
            {"amplitude": 1.0, "frequency": 5.0 + i * 2, "phase": 0.0, "waveform": "sine", "enabled": True}
            for i in range(5)
        ]
    if "highlight_component" not in st.session_state:
        st.session_state.highlight_component = None
    if "event_spectrum" not in st.session_state:
        st.session_state.event_spectrum = None
    if "band_bounds" not in st.session_state:
        st.session_state.band_bounds = [-20.0, 20.0]


def component_template() -> dict:
    return {"amplitude": 0.5, "frequency": 10.0, "phase": 0.0, "waveform": "sine", "enabled": True}


ensure_session_state()

# -----------------------------
# Sidebar Signal Builder
# -----------------------------
with st.sidebar:
    st.header("Signal Builder")
    st.info(
        "Use these controls to compose a custom signal. Every slider includes a short explanation so you understand how it shapes the waveform and eventually its spectrum."
    )

    preset_choice = st.selectbox(
        "Quick-add preset",
        options=list(PRESETS.keys()),
        help="Load a canonical waveform to compare against your custom sum of sinusoids."
    )
    st.caption(PRESETS[preset_choice]["description"])

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Add component", use_container_width=True, disabled=len(st.session_state.components) >= MAX_COMPONENTS):
            st.session_state.components.append(component_template())
    with col_btn2:
        if st.button("Remove component", use_container_width=True, disabled=len(st.session_state.components) <= MIN_COMPONENTS):
            st.session_state.components.pop()
    st.caption("Maintain at least one sinusoid, but feel free to explore up to eight layers of frequency content.")

    for idx, comp in enumerate(st.session_state.components):
        with st.expander(f"Component {idx + 1}", expanded=(idx == 0)):
            comp["enabled"] = st.checkbox(
                "Active",
                value=comp["enabled"],
                key=f"enabled_{idx}",
                help="Turn components on/off to hear how each sinusoid contributes energy.",
            )
            st.caption("Use the Active checkbox to temporarily mute this sinusoid while keeping its settings.")

            comp["amplitude"] = st.slider(
                "Amplitude",
                min_value=0.0,
                max_value=3.0,
                value=float(comp["amplitude"]),
                step=0.05,
                key=f"amp_{idx}",
            )
            st.caption("Amplitude controls the height of the wave. Larger amplitudes inject more energy at this frequency in the spectrum.")

            comp["frequency"] = st.slider(
                "Frequency (Hz)",
                min_value=0.1,
                max_value=80.0,
                value=float(comp["frequency"]),
                step=0.1,
                key=f"freq_{idx}",
            )
            st.caption("Frequency determines how many oscillations fit into one second. Higher frequencies push spectral peaks to the right.")

            comp["phase"] = st.slider(
                "Phase (rad)",
                min_value=-np.pi,
                max_value=np.pi,
                value=float(comp["phase"]),
                step=0.1,
                key=f"phase_{idx}",
            )
            st.caption("Phase shifts the waveform left/right in time. It does not change magnitude but affects alignment when signals add.")

            comp["waveform"] = st.radio(
                "Waveform",
                options=["sine", "cosine"],
                index=0 if comp["waveform"] == "sine" else 1,
                key=f"wave_{idx}",
                horizontal=True,
            )
            st.caption("Choose sine or cosine to decide whether the component starts at zero or its peak, reinforcing the geometry behind Fourier sums.")

# -----------------------------
# Shared Base Signal
# -----------------------------
base_time = generate_time_vector(DEFAULT_FS, DEFAULT_DURATION, DEFAULT_SAMPLES)
full_signal, latex_expression, component_signals, preset_signal = build_signal(
    st.session_state.components, preset_choice, base_time
)
window_rect = np.ones_like(base_time)
freqs_main, spectrum_main = fft_spectrum(full_signal, DEFAULT_FS, window_rect)
if st.session_state.highlight_component is not None:
    st.session_state.highlight_component = min(
        st.session_state.highlight_component, len(st.session_state.components) - 1
    )

# -----------------------------
# Tabs
# -----------------------------
tab_signal, tab_sampling, tab_window, tab_band = st.tabs(
    ["Signal Builder", "Sampling & Aliasing", "Windowing & Leakage", "Band Reconstruction"]
)


# -----------------------------
# Tab 1: Signal Builder & Linked Views
# -----------------------------
with tab_signal:
    st.info(
        "Purpose: build intuition about how time-domain ingredients translate into spectral peaks. Learn to read analytical expressions, interpret linked plots, and reconstruct signals by isolating bands."
    )

    latex_col, explain_col = st.columns([2, 1])
    with latex_col:
        st.latex(latex_expression)
        st.caption("This LaTeX expression lists every active sinusoid plus the optional preset. Each term follows A·sin(2πft + φ) or A·cos(2πft + φ).")
    with explain_col:
        st.success(
            "Symbols recap: A is amplitude, f is frequency in Hz, t is time in seconds, and φ is the phase shift. Summing these terms constructs the composite waveform you see below."
        )

    col_time, col_freq = st.columns(2)

    highlight_idx = st.session_state.get("highlight_component")

    with col_time:
        fig_time = go.Figure()
        fig_time.add_trace(
            go.Scatter(
                x=base_time,
                y=full_signal,
                name="Composite signal",
                line=dict(color="#1f77b4", width=2),
            )
        )
        for idx, comp_sig in enumerate(component_signals):
            fig_time.add_trace(
                go.Scatter(
                    x=base_time,
                    y=comp_sig,
                    name=f"Component {idx + 1}",
                    line=dict(width=1, dash="dot"),
                    opacity=0.2 if highlight_idx is not None and highlight_idx != idx else 0.5,
                )
            )
        if highlight_idx is not None:
            fig_time.add_trace(
                go.Scatter(
                    x=base_time,
                    y=component_signals[highlight_idx],
                    name=f"Highlighted component {highlight_idx + 1}",
                    line=dict(color="#d62728", width=3),
                )
            )
        fig_time.update_layout(
            title="Time-domain signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode="x",
        )
        time_events = plotly_events(
            fig_time,
            hover_event=True,
            select_event=False,
            click_event=False,
            key="time_events",
            override_height=420,
        )
        st.caption(
            "Hover anywhere on the time plot to probe local events. Sharp spikes trigger wide spectral spreads, while smooth oscillations stay narrow."
        )
        st.info("Sharp events contain many frequencies. Hover to see their spectral footprint.")
        if time_events:
            hover_time = time_events[0].get("x", 0.0)
            window_len = max(int(0.05 * DEFAULT_FS), 8)
            center_idx = np.argmin(np.abs(base_time - hover_time))
            start = max(center_idx - window_len, 0)
            stop = min(center_idx + window_len, len(base_time))
            local_segment = np.zeros_like(full_signal)
            local_segment[start:stop] = full_signal[start:stop]
            seg_freqs, seg_spec = fft_spectrum(local_segment, DEFAULT_FS, window_rect)
            st.session_state.event_spectrum = (seg_freqs, np.abs(seg_spec))
            st.caption(
                "Sharp events contain many frequencies. The orange trace in the spectrum panel shows the wide spectral footprint of the region you hovered."
            )

    with col_freq:
        freq_fig = make_subplots(rows=2, cols=1, shared_x=True, subplot_titles=("Magnitude", "Phase"))
        mask = freqs_main >= 0
        freq_fig.add_trace(
            go.Scatter(
                x=freqs_main[mask],
                y=np.abs(spectrum_main)[mask],
                name="|X(f)|",
                line=dict(color="#1f77b4"),
            ),
            row=1,
            col=1,
        )
        if st.session_state.event_spectrum is not None:
            seg_freqs, seg_mag = st.session_state.event_spectrum
            mask_seg = seg_freqs >= 0
            freq_fig.add_trace(
                go.Scatter(
                    x=seg_freqs[mask_seg],
                    y=seg_mag[mask_seg],
                    name="Local spectral footprint",
                    line=dict(color="#ff7f0e", dash="dot"),
                ),
                row=1,
                col=1,
            )
        freq_fig.add_trace(
            go.Scatter(
                x=freqs_main[mask],
                y=np.angle(spectrum_main)[mask],
                name="∠X(f)",
                line=dict(color="#2ca02c"),
            ),
            row=2,
            col=1,
        )
        freq_fig.update_layout(
            title="Frequency-domain magnitude & phase",
            xaxis2_title="Frequency (Hz)",
            hovermode="x",
            dragmode="select",
        )
        freq_events = plotly_events(
            freq_fig,
            hover_event=True,
            select_event=True,
            click_event=False,
            key="freq_events",
            override_height=420,
        )
        st.caption(
            "Hover on a frequency peak to highlight the matching sinusoid in the time plot. Drag-select a band to earmark it for reconstruction in the Band tab."
        )
        st.info("Hovering on a frequency component helps visualize which part of the time signal this component contributes to.")
        if freq_events:
            freq_val = freq_events[0].get("x", 0.0)
            closest_idx = np.argmin(
                [abs(freq_val - comp["frequency"]) for comp in st.session_state.components]
            )
            st.session_state.highlight_component = closest_idx
            if len(freq_events) > 1:
                xs = [pt.get("x", 0.0) for pt in freq_events]
                st.session_state.band_bounds = [float(max(0.0, min(xs))), float(max(xs))]


# -----------------------------
# Tab 2: Sampling & Aliasing
# -----------------------------
with tab_sampling:
    st.info(
        "Purpose: connect sampling parameters to Nyquist, aliasing, and spectral resolution. Adjust fs, duration, and sample count to see how discrete representations change the FFT."
    )

    col_params, col_plots = st.columns([1, 2])
    with col_params:
        fs = st.slider(
            "Sampling rate fs (Hz)",
            min_value=100,
            max_value=4000,
            value=1000,
            step=50,
            help="fs specifies how many samples you collect per second. Nyquist = fs/2 sets the highest recoverable frequency.",
        )
        st.caption("Higher fs pushes the Nyquist limit outward, reducing aliasing.")

        duration = st.slider(
            "Total duration T (s)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Duration controls how long you observe the signal. Longer windows improve frequency resolution (closer FFT bins).",
        )
        st.caption("Longer T narrows spectral bins because Δf = 1/T.")

        samples = st.slider(
            "Number of samples N",
            min_value=128,
            max_value=4096,
            value=1024,
            step=128,
            help="N determines how many discrete points represent the waveform.",
        )
        st.caption("Changing N while keeping T adjusts the sample spacing and influences FFT bin spacing.")

        zero_pad = st.checkbox("Enable zero padding", value=False)
        st.caption("Zero padding interpolates the FFT for smoother visuals but does not add new frequency information.")
        pad_factor = st.slider(
            "Pad factor",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            disabled=not zero_pad,
        )
        st.caption("Use larger pad factors for a smoother-looking spectrum. Remember it's visual interpolation only.")

    with col_plots:
        discrete_time = generate_time_vector(fs, duration, samples)
        discrete_signal, _, _, _ = build_signal(st.session_state.components, preset_choice, discrete_time)
        sampled_indices = np.linspace(0, len(discrete_time) - 1, min(len(discrete_time), 120), dtype=int)

        fig_sampling = go.Figure()
        fig_sampling.add_trace(go.Scatter(x=discrete_time, y=discrete_signal, name="Continuous-ish", line=dict(width=2)))
        fig_sampling.add_trace(
            go.Scatter(
                x=discrete_time[sampled_indices],
                y=discrete_signal[sampled_indices],
                mode="markers",
                marker=dict(size=6, color="#d62728"),
                name="Sampled points",
            )
        )
        fig_sampling.update_layout(title="Sampled signal", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_sampling, use_container_width=True)
        st.caption("Red markers show the discrete samples that the FFT uses. Too few samples per cycle cause aliasing artifacts.")

        window_vals = np.ones_like(discrete_signal)
        freqs_disc, spec_disc = fft_spectrum(discrete_signal, fs, window_vals, pad_factor if zero_pad else 1.0)
        positive = freqs_disc >= 0
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Bar(x=freqs_disc[positive], y=np.abs(spec_disc)[positive], name="|X(f)|"))
        fig_spec.update_layout(title="Sampled spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
        st.plotly_chart(fig_spec, use_container_width=True)
        st.caption("Watch peaks fold back toward zero whenever they exceed Nyquist (fs/2).")

        alias_notes = describe_aliasing(st.session_state.components, fs)
        if alias_notes:
            st.error("\n".join(alias_notes))
        else:
            st.success("All component frequencies sit below Nyquist, so no aliasing occurs.")


# -----------------------------
# Tab 3: Windowing & Leakage
# -----------------------------
with tab_window:
    st.info(
        "Purpose: understand how window shapes tame edge discontinuities and why misaligned tones produce spectral leakage."
    )

    col_controls, col_window_plot, col_spectrum = st.columns([1.1, 1, 1.2])

    with col_controls:
        window_name = st.selectbox(
            "Window type",
            options=["Rectangular", "Hann", "Hamming", "Blackman", "Kaiser"],
            index=1,
            help="Different windows trade off main-lobe width (resolution) and sidelobe level (noise suppression).",
        )
        st.caption("Rectangular keeps the narrowest main lobe but suffers high sidelobes. Tapered windows widen the main lobe but suppress leakage.")

        beta = st.slider(
            "Kaiser β",
            min_value=0.0,
            max_value=14.0,
            value=8.6,
            step=0.2,
            disabled=window_name != "Kaiser",
            help="β tunes the Kaiser window. Higher β increases sidelobe suppression at the cost of wider main lobes.",
        )
        st.caption("Adjust β only when the Kaiser window is selected to observe custom leakage tradeoffs.")

        misalign = st.checkbox(
            "Misalign frequency to bin",
            value=True,
            help="When active, the primary component is shifted by a non-integer multiple of Δf, producing leakage.",
        )
        st.caption("Perfect bin alignment concentrates energy in one bin; misalignment smears energy across neighbors.")

    with col_window_plot:
        N = DEFAULT_SAMPLES
        window_vals, window_meta = compute_window(window_name, N, beta)
        fig_window = go.Figure()
        fig_window.add_trace(go.Scatter(x=base_time, y=window_vals, name=f"{window_name} window"))
        fig_window.update_layout(title="Window in time", xaxis_title="Time (s)", yaxis_title="Weight")
        st.plotly_chart(fig_window, use_container_width=True)
        st.caption("Windowing tapers the edges so the FFT sees fewer discontinuities.")
        st.write(f"Main-lobe width estimate: **{window_meta['main_lobe']}**")
        st.write(f"Peak sidelobe level: **{window_meta['sidelobe']}**")
        st.caption("Main-lobe width limits how close two tones can be before they blur, while sidelobe level controls leakage of strong neighbors.")

    with col_spectrum:
        leakage_signal = full_signal.copy()
        if misalign:
            offset = (DEFAULT_FS / DEFAULT_SAMPLES) * 0.37
            leakage_signal += 0.8 * np.sin(2 * np.pi * (12.0 + offset) * base_time)
        leakage_windowed = leakage_signal * window_vals
        freqs_leak, spec_leak = fft_spectrum(leakage_windowed, DEFAULT_FS, zero_pad_factor=2)
        pos = freqs_leak >= 0
        fig_leak = go.Figure()
        fig_leak.add_trace(
            go.Scatter(x=freqs_leak[pos], y=20 * np.log10(np.abs(spec_leak[pos]) + 1e-9), name="Windowed spectrum"),
        )
        fig_leak.update_layout(title="Spectral leakage demo", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
        st.plotly_chart(fig_leak, use_container_width=True)
        st.caption("Toggle misalignment to see energy smear across bins. Better windows cut down sidelobe spread.")


# -----------------------------
# Tab 4: Band Reconstruction
# -----------------------------
with tab_band:
    st.info(
        "Purpose: practice filtering by selecting frequency bands and watching the reconstructed time-domain contribution."
    )

    default_low = max(0.0, float(st.session_state.band_bounds[0]))
    default_high = max(default_low + 0.5, float(st.session_state.band_bounds[1]))
    band_slider = st.slider(
        "Manual band selection (Hz)",
        min_value=0.0,
        max_value=float(DEFAULT_FS / 2),
        value=(default_low, default_high),
        help="Use this slider or drag directly on the spectrum in the Signal Builder tab to define the band to keep.",
    )
    st.caption("Band-pass thinking: everything inside stays, everything outside is muted before the inverse FFT.")

    freqs_band, spectrum_band = fft_spectrum(full_signal, DEFAULT_FS, window_rect, zero_pad_factor=1)
    low, high = band_slider
    st.session_state.band_bounds = [low, high]
    reconstructed = reconstruct_band(spectrum_band, freqs_band, (low, high))

    fig_band = go.Figure()
    mask_pos = freqs_band >= 0
    fig_band.add_trace(go.Bar(x=freqs_band[mask_pos], y=np.abs(spectrum_band)[mask_pos], name="Full spectrum"))
    band_mask = (freqs_band >= low) & (freqs_band <= high)
    fig_band.add_trace(
        go.Bar(
            x=freqs_band[band_mask],
            y=np.abs(spectrum_band)[band_mask],
            marker_color="#d62728",
            name="Selected band",
        )
    )
    fig_band.update_layout(title="Band selection", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    st.plotly_chart(fig_band, use_container_width=True)
    st.caption("Red bars show the band that will survive the reconstruction.")

    fig_recon = go.Figure()
    fig_recon.add_trace(go.Scatter(x=base_time, y=full_signal, name="Original"))
    fig_recon.add_trace(go.Scatter(x=base_time, y=reconstructed[: len(base_time)], name="Reconstructed band", line=dict(color="#d62728")))
    fig_recon.update_layout(title="Time-domain comparison", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_recon, use_container_width=True)
    st.caption("Overlay shows which structures remain after filtering. Narrow bands isolate single tones; wide bands keep broader features.")

    st.warning(
        "Filtering concept: selecting a band equals multiplying the spectrum by an ideal rectangular window. The inverse FFT reveals which time-domain structures rely on that frequency support."
    )
