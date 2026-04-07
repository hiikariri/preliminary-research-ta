import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add lib directory to path for custom modules
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if lib_path not in sys.path:
    sys.path.append(lib_path)

# Import DWT functions
try:
    from dwt import discrete_wavelet_transform, apply_dwt
except ImportError:
    st.error("Could not import dwt module. Make sure dwt.py is in the lib folder.")

# Page configuration
st.set_page_config(
    page_title="Signal Quality Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# FUNCTION DEFINITIONS
# ===========================

def assess_ppg_quality(signal):
    """
    Assess PPG signal quality
    TODO: Implement actual SQA algorithm
    
    Returns:
        quality (str): Quality label (Excellent, Good, Fair, Poor)
        score (float): Quality score (0-100)
        details (str): Detailed assessment information
    """
    # PLACEHOLDER: Simple statistical analysis
    # Replace this with actual SQA algorithm later
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    snr_estimate = mean_val / std_val if std_val > 0 else 0
    
    # Simple quality scoring (placeholder)
    score = min(100, snr_estimate * 10)
    
    if score >= 80:
        quality = "Excellent"
    elif score >= 60:
        quality = "Good"
    elif score >= 40:
        quality = "Fair"
    else:
        quality = "Poor"
    
    details = f"Quality: {quality}\n"
    details += f"Score: {score:.2f}/100\n"
    details += f"Mean: {mean_val:.4f}\n"
    details += f"Std Dev: {std_val:.4f}\n"
    details += f"SNR Est: {snr_estimate:.2f}\n"
    details += f"Samples: {len(signal)}"
    
    return quality, score, details

def assess_ecg_quality(signal, return_peaks=False):
    """
    Assess ECG signal quality using the algorithm from the paper.
    
    Process:
    1. Feasibility Conditions:
       - Flat line/saturation detection
       - Pure noise detection (Gaussian noise)
    2. QRS-based conditions (using DWT R-peak detection):
       - Heart rate feasibility (24-240 BPM)
       - No abrupt QRS amplitude changes
    3. Quality scoring algorithms:
       - Beats' Average Correlation Algorithm (iScore)
       - Beats' Clustering Algorithm (aScore)
    
    Args:
        signal: ECG signal array
        return_peaks: If True, return detected R-peaks along with quality metrics
    
    Returns:
        quality (str): Quality label (Bad quality, HR quality, Diag quality)
        score (float): Quality score (0-100)
        details (str): Detailed assessment information
        r_peaks (array): If return_peaks=True, indices of detected R peaks
    """
    
    details_list = []
    
    # Step 1: Feasibility Conditions
    feasibility_passed, feasibility_details = check_feasibility_conditions(signal)
    details_list.append("=== Feasibility Conditions ===")
    details_list.append(feasibility_details)
    
    if not feasibility_passed:
        quality = "Bad quality"
        score = 0
        details = "\n".join(details_list)
        details += "\n\nResult: Unacceptable Signal"
        if return_peaks:
            return quality, score, details, np.array([])
        return quality, score, details
    
    # Step 2: QRS-based conditions
    # TODO: Replace with your DWT-based R peak detection
    qrs_passed, qrs_details, r_peaks = check_qrs_conditions_dwt(signal)
    details_list.append("\n=== QRS Conditions ===")
    if not qrs_passed:
        quality = "Bad quality"
        score = 0
        details = "\n".join(details_list)
        details += "\n\nResult: Unacceptable Signal"
        if return_peaks:
            return quality, score, details, r_peaks
        return quality, score, details
    
    # Step 3: Beats' Average Correlation Algorithm (iScore)
    iscore = calculate_iscore(signal, r_peaks)
    details_list.append(f"\n=== iScore ===")
    details_list.append(f"iScore: {iscore:.2f}")
    
    if iscore < 1:
        quality = "Bad quality"
        score = iscore * 100 / 3  # Normalize to 0-33%
        details = "\n".join(details_list)
        details += "\n\nResult: Unacceptable Signal"
        if return_peaks:
            return quality, score, details, r_peaks
        return quality, score, details
    
    # Step 4: Beats' Clustering Algorithm (aScore)
    ascore = calculate_ascore(signal, r_peaks)
    details_list.append(f"\n=== aScore ===")
    details_list.append(f"aScore: {ascore:.2f}")
    
    # Final quality determination
    if ascore < 3:
        quality = "Bad quality"
        score = 33 + (ascore * 100 / 9)  # 33-44%
    elif ascore >= 3 and ascore < 2:
        quality = "HR quality"
        score = 44 + ((ascore - 3) * 100 / 9)  # 44-55%
    else:  # ascore >= 2
        quality = "Diag quality"
        score = 70 + min(30, (ascore - 2) * 15)  # 70-100%
    
    details = "\n".join(details_list)
    details += f"\n\nFinal Quality: {quality}"
    details += "\nResult: Acceptable Signal" if quality != "Bad quality" else "\nResult: Unacceptable Signal"
    
    if return_peaks:
        return quality, score, details, r_peaks
    return quality, score, details

def check_feasibility_conditions(signal):
    """
    Check feasibility conditions: flat line/saturation and pure noise detection.
    
    Returns:
        passed (bool): True if all conditions passed
        details (str): Description of checks performed
    """
    details_lines = []
    
    # Normalize signal to [0, 1000] for flat line detection
    signal_norm = ((signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 1000)
    
    # Flat line/Saturation detection
    # Check for CONSECUTIVE flat samples (not scattered small differences)
    diff = np.diff(signal_norm)
    is_flat = np.abs(diff) <= 0.1  # Very small threshold for truly flat segments
    
    # Count maximum consecutive flat samples
    # Using a run-length encoding approach
    flat_runs = []
    current_run = 0
    for flat in is_flat:
        if flat:
            current_run += 1
        else:
            if current_run > 0:
                flat_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        flat_runs.append(current_run)
    
    # Get maximum consecutive flat duration
    max_flat_samples = max(flat_runs) if flat_runs else 0
    
    # Assuming 125 Hz sampling rate (typical for ECG)
    sampling_rate = 125
    flat_duration = max_flat_samples / sampling_rate
    
    flat_detected = flat_duration > 1.5  # More than 1.5 seconds of consecutive flat line
    
    details_lines.append(f"Flat line check:")
    details_lines.append(f"  Max consecutive flat samples: {max_flat_samples}")
    details_lines.append(f"  Max flat duration: {flat_duration:.2f}s")
    details_lines.append(f"  Status: {'FAILED' if flat_detected else 'PASSED'}")
    
    if flat_detected:
        return False, "\n".join(details_lines)
    
    # Pure Noise Detection (Gaussian noise using zero crossings)
    # High-frequency noise has excessive zero crossings relative to signal duration
    signal_scaled = 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
    zero_crossings = np.sum(np.diff(np.sign(signal_scaled)) != 0)
    
    # Calculate zero crossing rate per second (threshold: 200 crossings per second indicates noise)
    # For a normal ECG at 125 Hz, we expect much fewer crossings
    sampling_rate = 125
    signal_duration = len(signal) / sampling_rate  # in seconds
    zero_crossing_rate = zero_crossings / signal_duration  # crossings per second
    
    # Threshold: More than 200 crossings per second suggests pure noise
    # A normal ECG might have 10-30 crossings per second
    gaussian_noise_detected = zero_crossing_rate >= 200
    
    details_lines.append(f"\nGaussian noise check:")
    details_lines.append(f"  Zero crossings: {zero_crossings}")
    details_lines.append(f"  Signal duration: {signal_duration:.2f}s")
    details_lines.append(f"  Zero crossing rate: {zero_crossing_rate:.2f}/s")
    details_lines.append(f"  Threshold: 200/s")
    details_lines.append(f"  Status: {'FAILED' if gaussian_noise_detected else 'PASSED'}")
    
    if gaussian_noise_detected:
        return False, "\n".join(details_lines)
    
    return True, "\n".join(details_lines)

def check_qrs_conditions_dwt(signal):
    """
    Check QRS-based conditions using DWT R-peak detection.
    TODO: Implement DWT-based R-peak detection algorithm
    
    Conditions:
    1. Heart rate between 24-240 BPM
    2. No abrupt QRS amplitude changes (max change < 0.9)
    
    Returns:
        passed (bool): True if conditions passed
        details (str): Description of checks performed
        r_peaks (array): Indices of detected R peaks
    """
    details_lines = []
    
    # PLACEHOLDER: Replace with actual DWT-based R-peak detection
    # For now, using a simple threshold-based detection as placeholder
    r_peaks = detect_r_peaks_placeholder(signal)
    
    details_lines.append(f"R-peak detection:")
    details_lines.append(f"  Detected peaks: {len(r_peaks)}")
    
    if len(r_peaks) < 2:
        details_lines.append(f"  Status: FAILED (insufficient peaks)")
        return False, "\n".join(details_lines), r_peaks
    
    # Calculate heart rate
    # Assuming 125 Hz sampling rate
    sampling_rate = 125
    rr_intervals = np.diff(r_peaks) / sampling_rate  # in seconds
    hr = 60 / np.mean(rr_intervals)  # beats per minute
    
    hr_valid = 24 <= hr <= 240
    
    details_lines.append(f"\nHeart rate check:")
    details_lines.append(f"  HR: {hr:.1f} BPM")
    details_lines.append(f"  Valid range: 24-240 BPM")
    details_lines.append(f"  Status: {'PASSED' if hr_valid else 'FAILED'}")
    
    if not hr_valid:
        return False, "\n".join(details_lines), r_peaks
    
    # Check QRS amplitude variations
    r_amplitudes = signal[r_peaks]
    r_amplitudes_norm = (r_amplitudes - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    if len(r_amplitudes_norm) > 1:
        qrs_amp_changes = np.abs(np.diff(r_amplitudes_norm))
        max_qrs_amp_change = np.max(qrs_amp_changes)
    else:
        max_qrs_amp_change = 0
    
    qrs_amp_valid = max_qrs_amp_change < 0.9
    
    details_lines.append(f"\nQRS amplitude check:")
    details_lines.append(f"  Max amplitude change: {max_qrs_amp_change:.3f}")
    details_lines.append(f"  Threshold: 0.9")
    details_lines.append(f"  Status: {'PASSED' if qrs_amp_valid else 'FAILED'}")
    
    if not qrs_amp_valid:
        return False, "\n".join(details_lines), r_peaks
    
    return True, "\n".join(details_lines), r_peaks

def detect_r_peaks_placeholder(signal):
    """
    DWT-based R-peak detection using zero crossing detection.
    Based on the paper: optimal gradient limits for scales 2^1 to 2^8
    
    Scale ranges and gradient thresholds:
    - Scale 2^1 to 2^3: gradient threshold = -0.7
    - Scale 2^4 to 2^6: gradient threshold = -0.5
    - Scale 2^7: gradient threshold = -0.4
    - Scale 2^8: gradient threshold = -0.1
    """
    try:
        # Get DWT coefficients
        h, g, qj, delay = discrete_wavelet_transform()
        
        # Sampling rate (assumed 125 Hz for BIDMC dataset)
        sampling_rate = 125
        total_time = len(signal) / sampling_rate
        
        # Apply DWT to signal
        w2fb = apply_dwt_fixed(delay, qj, total_time, sampling_rate, signal)
        
        # Detect zero crossings at multiple scales using gradient method
        r_peaks_all = []
        
        # Define gradient thresholds for each scale
        gradient_thresholds = {
            1: -0.7, 2: -0.7, 3: -0.7,  # Scale 2^1 to 2^3
            4: -0.5, 5: -0.5, 6: -0.5,  # Scale 2^4 to 2^6
            7: -0.4,                     # Scale 2^7
            8: -0.1                      # Scale 2^8
        }
        
        # Process scales 1-8 (2^1 to 2^8)
        for scale in range(1, 9):
            threshold = gradient_thresholds[scale]
            
            # Get wavelet coefficients for this scale
            coeffs = w2fb[scale][:len(signal)]
            
            # Calculate gradient
            gradient = np.gradient(coeffs)
            
            # Find zero crossings with gradient threshold
            zero_crossings = detect_zero_crossings_with_gradient(coeffs, gradient, threshold)
            
            if len(zero_crossings) > 0:
                r_peaks_all.extend(zero_crossings)
        
        # Remove duplicates and sort
        r_peaks_all = sorted(list(set(r_peaks_all)))
        
        # Filter peaks that are too close (minimum RR interval ~0.3s = 37 samples at 125Hz)
        r_peaks = filter_close_peaks(r_peaks_all, min_distance=37)
        
        return np.array(r_peaks)
        
    except Exception as e:
        # Fallback to simple threshold-based detection
        st.warning(f"DWT detection failed: {str(e)}. Using fallback method.")
        return detect_r_peaks_fallback(signal)

def apply_dwt_fixed(delay, qj, total_time, sampling_rate, signal):
    """
    Fixed version of apply_dwt that properly computes wavelet transform at each scale.
    """
    length = int(total_time * sampling_rate)
    w2fb = [[0 for _ in range(length + max(delay))] for _ in range(9)]
    
    for n in range(length):
        for j in range(1, 9):
            # Calculate convolution for scale j
            a = -(round(2**j) + round(2**(j-1)) - 2)
            b = -(a - round(2**(j-1)))
            
            for k in range(a, b+1):
                qj_idx = k + abs(a)
                sig_idx = n - (k + abs(a))
                
                if 0 <= qj_idx < len(qj[j]) and 0 <= sig_idx < len(signal):
                    w2fb[j][n + delay[j-1]] += qj[j][qj_idx] * signal[sig_idx]
    
    return w2fb

def detect_zero_crossings_with_gradient(coeffs, gradient, threshold):
    """
    Detect zero crossings with gradient threshold for QRS detection.
    
    Args:
        coeffs: Wavelet coefficients
        gradient: Gradient of coefficients
        threshold: Gradient threshold value (negative)
    
    Returns:
        List of indices where zero crossings occur with gradient below threshold
    """
    zero_crossings = []
    
    # Detect sign changes (zero crossings)
    signs = np.sign(coeffs)
    sign_changes = np.diff(signs)
    
    # Find indices where sign changes from positive to negative
    pos_to_neg = np.where(sign_changes < 0)[0]
    
    # Filter by gradient threshold
    for idx in pos_to_neg:
        if idx < len(gradient) and gradient[idx] < threshold:
            zero_crossings.append(idx)
    
    return zero_crossings

def filter_close_peaks(peaks, min_distance=37):
    """
    Filter out peaks that are too close together.
    Keep the peak with highest amplitude in each cluster.
    
    Args:
        peaks: List of peak indices
        min_distance: Minimum allowed distance between peaks (samples)
    
    Returns:
        Filtered list of peaks
    """
    if len(peaks) == 0:
        return peaks
    
    filtered_peaks = [peaks[0]]
    
    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] >= min_distance:
            filtered_peaks.append(peak)
    
    return filtered_peaks

def detect_r_peaks_fallback(signal):
    """
    Fallback simple threshold-based R-peak detection.
    Used when DWT method fails.
    """
    # Normalize signal
    signal_norm = (signal - np.mean(signal)) / np.std(signal)
    
    # Find peaks above threshold
    threshold = 0.5
    potential_peaks = np.where(signal_norm > threshold)[0]
    
    # Remove peaks too close together (minimum 0.3s = 37 samples at 125Hz)
    min_distance = 37
    r_peaks = []
    
    if len(potential_peaks) > 0:
        r_peaks.append(potential_peaks[0])
        for peak in potential_peaks[1:]:
            if peak - r_peaks[-1] >= min_distance:
                r_peaks.append(peak)
    
    return np.array(r_peaks)

def calculate_iscore(signal, r_peaks):
    """
    Calculate iScore using Beats' Average Correlation Algorithm.
    TODO: Implement the full algorithm as per the paper
    
    Returns:
        iscore (float): Score in range [L1, L2] where L1<1<L2
    """
    # PLACEHOLDER: Simplified correlation-based scoring
    if len(r_peaks) < 3:
        return 0
    
    # Extract beats (simplified)
    beats = []
    for i in range(len(r_peaks) - 1):
        beat = signal[r_peaks[i]:r_peaks[i+1]]
        if len(beat) > 0:
            beats.append(beat)
    
    if len(beats) < 2:
        return 0
    
    # Calculate average correlation (simplified)
    # In full implementation, this would use template matching
    correlations = []
    for i in range(len(beats) - 1):
        # Resample to same length for correlation
        len_min = min(len(beats[i]), len(beats[i+1]))
        b1 = beats[i][:len_min]
        b2 = beats[i+1][:len_min]
        
        if len(b1) > 1:
            corr = np.corrcoef(b1, b2)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    if len(correlations) == 0:
        return 0
    
    avg_corr = np.mean(correlations)
    # Map correlation to iScore range [0, 3]
    iscore = avg_corr * 3
    
    return iscore

def calculate_ascore(signal, r_peaks):
    """
    Calculate aScore using Beats' Clustering Algorithm.
    TODO: Implement the full algorithm as per the paper
    
    Returns:
        ascore (float): Score in range [L3, L2] where L3<3<L2
    """
    # PLACEHOLDER: Simplified clustering-based scoring
    if len(r_peaks) < 3:
        return 0
    
    # Extract beats
    beats = []
    for i in range(len(r_peaks) - 1):
        beat = signal[r_peaks[i]:r_peaks[i+1]]
        if len(beat) > 0:
            beats.append(beat)
    
    if len(beats) < 2:
        return 0
    
    # Calculate beat consistency (simplified)
    # In full implementation, this would use clustering
    beat_lengths = [len(b) for b in beats]
    length_std = np.std(beat_lengths) / np.mean(beat_lengths) if np.mean(beat_lengths) > 0 else 1
    
    # Map consistency to aScore range [0, 5]
    ascore = max(0, 5 - length_std * 5)
    
    return ascore

def get_quality_color(quality):
    """Get color based on quality level"""
    colors = {
        "Diag quality": "#00ff00",      # Green - Diagnostic quality
        "HR quality": "#ffaa00",        # Orange - Heart rate quality
        "Bad quality": "#ff4444",       # Red - Bad quality
        "Excellent": "#00ff00",         # For PPG
        "Good": "#7fff00",
        "Fair": "#ffaa00",
        "Poor": "#ff4444"
    }
    return colors.get(quality, "gray")

# ===========================
# STREAMLIT APP STARTS HERE
# ===========================

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ppg_quality' not in st.session_state:
    st.session_state.ppg_quality = None
if 'ecg_quality' not in st.session_state:
    st.session_state.ecg_quality = None
if 'assessment_done' not in st.session_state:
    st.session_state.assessment_done = False

# Title
st.title("Signal Quality Assessment")
st.markdown("### RESP, ECG, and PPG Signal Analysis")

# Sidebar
with st.sidebar:
    st.header("Data Import")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV and strip whitespace from column names
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.df.columns = st.session_state.df.columns.str.strip()
            st.success(f"Loaded: {uploaded_file.name}")
            
            # Display file info
            num_samples = len(st.session_state.df)
            duration = st.session_state.df['Time [s]'].iloc[-1] if 'Time [s]' in st.session_state.df.columns else 0
            
            st.metric("Total Samples", f"{num_samples:,}")
            st.metric("Duration", f"{duration:.2f} s")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    st.divider()
    
    # Assessment button
    if st.session_state.df is not None:
        if st.button("Assess Signal Quality", use_container_width=True, type="primary"):
            st.session_state.assessment_done = True
            st.rerun()
    else:
        st.info("Upload a CSV file to begin")

# Main content area
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Plot signals
    st.subheader("Signal Visualization")
    
    try:
        # Get time data
        if 'Time [s]' in df.columns:
            time = df['Time [s]']
        else:
            # Create time index if column doesn't exist
            time = np.arange(len(df))
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Respiratory Signal', 'ECG Lead II Signal', 'Photoplethysmogram (PPG) Signal'),
            vertical_spacing=0.08
        )
        
        # Plot RESP (Respiratory Signal)
        if 'RESP' in df.columns:
            fig.add_trace(
                go.Scatter(x=time, y=df['RESP'], mode='lines', 
                          name='RESP', line=dict(color='#1f77b4', width=1)),
                row=1, col=1
            )
            fig.update_yaxes(title_text="RESP", row=1, col=1)
        
        # Plot II (ECG Lead II)
        if 'II' in df.columns:
            fig.add_trace(
                go.Scatter(x=time, y=df['II'], mode='lines', 
                          name='ECG II', line=dict(color='#ff7f0e', width=1)),
                row=2, col=1
            )
            fig.update_yaxes(title_text="ECG II", row=2, col=1)
        
        # Plot PLETH (PPG Signal)
        if 'PLETH' in df.columns:
            fig.add_trace(
                go.Scatter(x=time, y=df['PLETH'], mode='lines', 
                          name='PPG', line=dict(color='#2ca02c', width=1)),
                row=3, col=1
            )
            fig.update_yaxes(title_text="PPG", row=3, col=1)
        
        # Update x-axis label for bottom plot
        fig.update_xaxes(title_text="Time [s]", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting signals: {str(e)}")
        st.write("DataFrame columns:", df.columns.tolist())
        st.write("DataFrame shape:", df.shape)
    
    # Quality Assessment Section
    if st.session_state.assessment_done:
        st.divider()
        st.subheader("Quality Assessment Results")
        
        col1, col2 = st.columns(2)
        
        # Assess PPG Quality
        if 'PLETH' in df.columns:
            ppg_signal = df['PLETH'].values
            ppg_quality, ppg_score, ppg_details = assess_ppg_quality(ppg_signal)
            
            with col1:
                st.markdown("#### PPG Signal Quality")
                
                # Quality badge with color
                quality_color = get_quality_color(ppg_quality)
                st.markdown(f"<h2 style='color: {quality_color};'>{ppg_quality}</h2>", unsafe_allow_html=True)
                
                st.metric("Quality Score", f"{ppg_score:.2f}/100")
        # Assess ECG Quality
        if 'II' in df.columns:
            ecg_signal = df['II'].values
            ecg_quality, ecg_score, ecg_details, r_peaks = assess_ecg_quality(ecg_signal, return_peaks=True)
            
            with col2:
                st.markdown("#### ECG Signal Quality")
                
                # Quality badge with color
                quality_color = get_quality_color(ecg_quality)
                st.markdown(f"<h2 style='color: {quality_color};'>{ecg_quality}</h2>", unsafe_allow_html=True)
                
                st.metric("Quality Score", f"{ecg_score:.2f}/100")
                
                with st.expander("View Details"):
                    st.text(ecg_details)
        
        # Visualization of R-peak detection
        if 'II' in df.columns and len(r_peaks) > 0:
            st.divider()
            st.subheader("R-Peak Detection Visualization")
            
            # Create ECG plot with detected R-peaks
            fig_peaks = go.Figure()
            
            # Get time data
            if 'Time [s]' in df.columns:
                time = df['Time [s]'].values
            else:
                time = np.arange(len(df)) / 125  # Assuming 125 Hz
            
            # Plot ECG signal
            fig_peaks.add_trace(go.Scatter(
                x=time,
                y=df['II'],
                mode='lines',
                name='ECG II',
                line=dict(color='#ff7f0e', width=1)
            ))
            
            # Plot detected R-peaks
            fig_peaks.add_trace(go.Scatter(
                x=time[r_peaks],
                y=df['II'].values[r_peaks],
                mode='markers',
                name='R-peaks',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x',
                    line=dict(width=2)
                )
            ))
            
            # Update layout
            fig_peaks.update_layout(
                title='ECG Signal with Detected R-Peaks',
                xaxis_title='Time [s]',
                yaxis_title='Amplitude',
                height=400,
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig_peaks, use_container_width=True)
            
            # Display RR interval statistics
            if len(r_peaks) > 1:
                st.subheader("Heart Rate Variability Analysis")
                
                col_hrv1, col_hrv2, col_hrv3 = st.columns(3)
                
                # Calculate RR intervals
                sampling_rate = 125
                rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # in milliseconds
                heart_rates = 60000 / rr_intervals  # in BPM
                
                with col_hrv1:
                    st.metric("Mean RR Interval", f"{np.mean(rr_intervals):.0f} ms")
                    st.metric("Std RR Interval", f"{np.std(rr_intervals):.0f} ms")
                
                with col_hrv2:
                    st.metric("Mean Heart Rate", f"{np.mean(heart_rates):.1f} BPM")
                    st.metric("HR Range", f"{np.min(heart_rates):.0f}-{np.max(heart_rates):.0f} BPM")
                
                with col_hrv3:
                    st.metric("Total Beats", f"{len(r_peaks)}")
                    st.metric("RMSSD", f"{np.sqrt(np.mean(np.diff(rr_intervals)**2)):.1f} ms")
                
                # Plot RR interval tachogram
                fig_rr = go.Figure()
                
                fig_rr.add_trace(go.Scatter(
                    x=time[r_peaks[1:]],
                    y=rr_intervals,
                    mode='lines+markers',
                    name='RR Intervals',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=6)
                ))
                
                fig_rr.update_layout(
                    title='RR Interval Tachogram',
                    xaxis_title='Time [s]',
                    yaxis_title='RR Interval [ms]',
                    height=300,
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig_rr, use_container_width=True)
                
                # Plot heart rate over time
                fig_hr = go.Figure()
                
                fig_hr.add_trace(go.Scatter(
                    x=time[r_peaks[1:]],
                    y=heart_rates,
                    mode='lines+markers',
                    name='Heart Rate',
                    line=dict(color='#d62728', width=2),
                    marker=dict(size=6)
                ))
                
                fig_hr.update_layout(
                    title='Instantaneous Heart Rate',
                    xaxis_title='Time [s]',
                    yaxis_title='Heart Rate [BPM]',
                    height=300,
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig_hr, use_container_width=True)
                st.metric("Quality Score", f"{ecg_score:.2f}/100")
                
                with st.expander("View Details"):
                    st.text(ecg_details)
        
else:
    # Welcome message
    st.info("Upload a CSV file from the sidebar to begin signal analysis")
    
    st.markdown("""
    ### How to use this application:
    
    1. **Upload Data**: Click on "Browse files" in the sidebar to upload your CSV file
    2. **View Signals**: The application will automatically plot RESP, ECG, and PPG signals
    3. **Assess Quality**: Click "Assess Signal Quality" to analyze signal quality
    4. **Review Results**: Check the quality scores and detailed metrics
    
    #### Expected CSV Format:
    - `Time [s]`: Time column in seconds
    - `RESP`: Respiratory signal
    - `II`: ECG Lead II signal
    - `PLETH`: PPG signal
    """)

