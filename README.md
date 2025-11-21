# BME3053C
Course materials for BME3053C

## Fourier Transform Learning Lab

This repository now includes an interactive Streamlit app (`streamlit_app.py`) that turns Lesson 09 into a hands-on Fourier transform studio. Students can build signals, view linked time–frequency plots, explore sampling theory, study windowing/leakage, and practice band reconstruction.

### Getting Started

1. **Install dependencies** (preferably inside a virtual environment):
	```bash
	pip install -r requirements.txt
	```
2. **Run the lab**:
	```bash
	streamlit run streamlit_app.py
	```
3. Open the URL shown in the terminal. Use a wide browser window so the paired plots display side by side.

### Feature Highlights

- **Signal Builder Sidebar** – Configure up to eight sinusoidal components with amplitude, frequency, phase, waveform, and on/off toggles. Quick-add presets (square, triangle, sawtooth, Gaussian pulse, step, impulse, chirp) illustrate classic spectral fingerprints. A live LaTeX expression keeps the math front and center.
- **Tab 1: Linked Views** – Detailed explainer text plus Plotly charts. Hovering over spectrum peaks highlights the matching sinusoid in time, while hovering in time reveals the wide spectral footprint of sharp events. Drag-select a frequency band for later reconstruction.
- **Tab 2: Sampling & Aliasing Playground** – Sliders for sampling rate, duration, sample count, and zero padding, each annotated with theory reminders. Sample markers overlay the waveform, and automated Nyquist/aliasing notes appear when folding occurs.
- **Tab 3: Windowing & Leakage Lab** – Compare Rectangular, Hann, Hamming, Blackman, and Kaiser windows (adjustable β). Visualize the window shape, review main-lobe/sidelobe estimates, and witness how misaligned tones smear energy across bins.
- **Tab 4: Band Reconstruction Studio** – Combine slider or brushed selections to isolate spectral bands, then overlay the inverse-transform result on the original signal to illustrate filtering concepts.

### Implementation Notes

- Uses helper functions (`generate_time_vector`, `build_signal`, `fft_spectrum`, `reconstruct_band`, `compute_window`) for clarity and reusability.
- Relies on `streamlit-plotly-events` to capture Plotly hover/selection data, enabling the linked-view interactions and band brushing workflow.
- Every slider, toggle, dropdown, and plot is accompanied by explanatory captions so the interface doubles as an educational worksheet.

