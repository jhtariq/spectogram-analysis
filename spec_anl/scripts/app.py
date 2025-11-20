# import numpy as np
# import streamlit as st
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt

# def compute_linear_spec(y, sr, n_fft, hop_length, window, to_db):
#     D = librosa.stft(
#         y,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         window=window,
#         center=True,
#     )
#     S = np.abs(D) ** 2
#     if to_db:
#         S = librosa.power_to_db(S, ref=np.max)
#     return S

# def compute_mel_spec(y, sr, n_fft, hop_length, window, n_mels, to_db):
#     S = librosa.feature.melspectrogram(
#         y=y,
#         sr=sr,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         window=window,
#         n_mels=n_mels,
#         power=2.0,
#     )
#     if to_db:
#         S = librosa.power_to_db(S, ref=np.max)
#     return S

# st.title("Interactive Spectrogram Explorer")

# uploaded_file = st.file_uploader(
#     "Upload an audio file",
#     type=["wav", "mp3", "flac", "ogg"],
# )

# # Global settings
# sr_target = st.sidebar.number_input(
#     "Target sample rate",
#     min_value=8000,
#     max_value=48000,
#     value=16000,
#     step=1000,
# )

# spec_type = st.sidebar.selectbox(
#     "Spectrogram type",
#     ["Mel", "Linear"],
# )

# # Common STFT parameters
# n_fft = st.sidebar.select_slider(
#     "Window size n_fft",
#     options=[512, 1024, 2048, 4096],
#     value=2048,
# )

# hop_length = st.sidebar.select_slider(
#     "Hop length",
#     options=[128, 256, 512, 1024],
#     value=512,
# )

# window = st.sidebar.selectbox(
#     "Window function",
#     ["hann", "hamming", "blackman"],
# )

# to_db = st.sidebar.checkbox("Convert to dB scale", value=True)

# # Mel specific options
# if spec_type == "Mel":
#     n_mels = st.sidebar.selectbox(
#         "Number of mel bands",
#         [64, 128, 256, 512],
#         index=1,
#     )
# else:
#     n_mels = None

# # interpolation = st.sidebar.selectbox(
# #     "Image interpolation",
# #     ["nearest", "bilinear", "bicubic"],
# #     index=0,
# # )

# log_freq_axis = (
#     spec_type == "Linear"
#     and st.sidebar.checkbox("Log frequency axis", value=True)
# )

# if uploaded_file is not None:
#     # Load audio
#     y, sr = librosa.load(uploaded_file, sr=sr_target, mono=True)

#     st.audio(uploaded_file, format="audio/wav")

#     # Compute spectrogram
#     if spec_type == "Mel":
#         S = compute_mel_spec(
#             y=y,
#             sr=sr,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             window=window,
#             n_mels=n_mels,
#             to_db=to_db,
#         )
#     else:
#         S = compute_linear_spec(
#             y=y,
#             sr=sr,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             window=window,
#             to_db=to_db,
#         )

#     fig, ax = plt.subplots(figsize=(9, 4))

#     if spec_type == "Mel":
#         img = librosa.display.specshow(
#             S,
#             sr=sr,
#             hop_length=hop_length,
#             x_axis="time",
#             y_axis="mel",
#             ax=ax,
#         )
#         ax.set_ylabel("Mel frequency")
#     else:
#         if log_freq_axis:
#             img = librosa.display.specshow(
#                 S,
#                 sr=sr,
#                 hop_length=hop_length,
#                 x_axis="time",
#                 y_axis="log",
#                 ax=ax,
#             )
#             ax.set_ylabel("Hz")
#         else:
#             img = librosa.display.specshow(
#                 S,
#                 sr=sr,
#                 hop_length=hop_length,
#                 x_axis="time",
#                 y_axis="linear",
#                 ax=ax,
#             )
#             ax.set_ylabel("Hz")

#     ax.set_title(f"{spec_type} spectrogram")
#     ax.set_xlabel("Time in seconds")

#     cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB" if to_db else None)
#     cbar.set_label("dB" if to_db else "Amplitude")

#     # Apply interpolation style after specshow
#     # img.set_interpolation(interpolation)

#     st.pyplot(fig)
# else:
#     st.info("Upload an audio file to see its spectrogram.")






import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import soundfile as sf
import librosa.feature as lf

def compute_linear_spec(
    y,
    sr,
    n_fft,
    hop_length,
    window,
    to_db,
    center,
    power,
    top_db,
):
    # STFT
    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
    )

    if power == 2.0:
        S = np.abs(D) ** 2
        if to_db:
            S = librosa.power_to_db(S, ref=np.max, top_db=top_db)
    else:
        S = np.abs(D)
        if to_db:
            S = librosa.amplitude_to_db(S, ref=np.max, top_db=top_db)

    return S


def compute_mel_spec(
    y,
    sr,
    n_fft,
    hop_length,
    window,
    n_mels,
    fmin,
    fmax,
    to_db,
    power,
    top_db,
):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )

    if to_db:
        S = librosa.power_to_db(S, ref=np.max, top_db=top_db)

    return S


# st.set_page_config(layout="wide")
st.title("Interactive Spectrogram Explorer")

# ===== Sidebar controls =====

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg"],
)

sr_target = st.sidebar.number_input(
    "Target sample rate",
    min_value=8000,
    max_value=48000,
    value=16000,
    step=1000,
)

spec_type = st.sidebar.selectbox(
    "Spectrogram type",
    ["Mel", "Linear"],
)

n_fft = st.sidebar.select_slider(
    "Window size n_fft",
    options=[512, 1024, 2048, 4096],
    value=2048,
)

hop_length = st.sidebar.select_slider(
    "Hop length",
    options=[128, 256, 512, 1024],
    value=512,
)

window = st.sidebar.selectbox(
    "Window function",
    ["hann", "hamming", "blackman"],
)

power_choice = st.sidebar.selectbox(
    "Spectrogram power",
    ["Power (|X|^2)", "Magnitude (|X|)"],
    index=0,
)
power = 2.0 if power_choice.startswith("Power") else 1.0

to_db = st.sidebar.checkbox("Convert to dB scale", value=True)

top_db = st.sidebar.slider(
    "Dynamic range top_db",
    min_value=40,
    max_value=120,
    value=80,
    step=5,
)

center = st.sidebar.checkbox(
    "Center frames (pad signal)",
    value=True,
)

# Frequency limits use the target sample rate for bounds
fmin = st.sidebar.number_input(
    "Minimum frequency Hz",
    min_value=0,
    max_value=sr_target // 2,
    value=0,
    step=10,
)

fmax = st.sidebar.number_input(
    "Maximum frequency Hz",
    min_value=1000,
    max_value=sr_target,
    value=sr_target // 2,
    step=100,
)

if fmax <= fmin:
    st.sidebar.error("Maximum frequency must be greater than minimum frequency")

# Mel specific options
if spec_type == "Mel":
    n_mels = st.sidebar.selectbox(
        "Number of mel bands",
        [64, 128, 256, 512],
        index=2,
    )
else:
    n_mels = None

log_freq_axis = (
    spec_type == "Linear"
    and st.sidebar.checkbox("Log frequency axis", value=True)
)

cmap = st.sidebar.selectbox(
    "Color map",
    ["magma", "viridis", "plasma", "inferno", "cividis"],
    index=0,
)

# ===== Main content =====

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=sr_target, mono=True)
    total_duration = len(y) / sr

    st.sidebar.markdown("### Time crop in seconds")

    start_time, end_time = st.sidebar.slider(
        "Select time range",
        min_value=0.0,
        max_value=float(np.round(total_duration, 2)),
        value=(0.0, float(np.round(total_duration, 2))),
        step=0.01,
    )

    if end_time <= start_time:
        st.sidebar.error("End time must be greater than start time")


    # st.audio(uploaded_file, format="audio/wav")
    # Slice waveform for the selected time range
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_seg = y[start_sample:end_sample]

    # Audio player for the cropped region
    # Streamlit audio expects bytes, so we reencode


    buf = io.BytesIO()
    sf.write(buf, y_seg, sr, format="WAV")
    st.audio(buf.getvalue(), format="audio/wav")


    if spec_type == "Mel":
        S = compute_mel_spec(
            y=y_seg,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            to_db=to_db,
            power=power,
            top_db=top_db,
        )
    else:
        S = compute_linear_spec(
            y=y_seg,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            to_db=to_db,
            center=center,
            power=power,
            top_db=top_db,
        )

    # fig, ax = plt.subplots(figsize=(9, 4))
    # Use the cropped audio segment if you added the time slider
    # otherwise set y_seg = y above
    y_seg = y_seg if "y_seg" in locals() else y

    # Two rows: waveform on top, spectrogram below
    fig, (ax_wave, ax_spec) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    # Waveform
    times_seg = np.linspace(0, len(y_seg) / sr, num=len(y_seg))
    ax_wave.plot(times_seg, y_seg)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform")

    # RMS energy curve on waveform
    rms = lf.rms(y=y_seg, frame_length=n_fft, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    rms_norm = rms / rms.max() if rms.max() > 0 else rms
    ax_wave.plot(rms_times, rms_norm, color="orange", alpha=0.8)
    ax_wave.legend(["Waveform", "RMS (norm)"])


    if spec_type == "Mel":
        img = librosa.display.specshow(
            S,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=fmin,
            fmax=fmax,
            cmap=cmap,
            ax=ax_spec,
        )
        ax_spec.set_ylabel("Mel frequency")
    else:
        if log_freq_axis:
            img = librosa.display.specshow(
                S,
                sr=sr,
                hop_length=hop_length,
                x_axis="time",
                y_axis="log",
                fmin=fmin,
                fmax=fmax,
                cmap=cmap,
                ax=ax_spec,
            )
        else:
            img = librosa.display.specshow(
                S,
                sr=sr,
                hop_length=hop_length,
                x_axis="time",
                y_axis="linear",
                fmin=fmin,
                fmax=fmax,
                cmap=cmap,
                ax=ax_spec,
            )
        ax_spec.set_ylabel("Hz")

    centroid = lf.spectral_centroid(
        y=y_seg,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]
    cent_times = librosa.frames_to_time(
        np.arange(len(centroid)), sr=sr, hop_length=hop_length
    )

    ax_spec.plot(cent_times, centroid, color="white", linewidth=1, alpha=0.7)

    # ax.set_title(f"{spec_type} spectrogram")
    ax_spec.set_title(f"{spec_type} spectrogram  from {start_time:.2f}s to {end_time:.2f}s")
    ax_spec.set_xlabel("Time in seconds")

    cbar = fig.colorbar(
        img,
        ax=ax_spec,
        format="%+2.0f dB" if to_db else None,
    )
    cbar.set_label("dB" if to_db else "Amplitude")

    st.pyplot(fig)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)

    st.download_button(
        label="Download current spectrogram as PNG",
        data=img_buf,
        file_name="spectrogram.png",
        mime="image/png",
    )
else:
    st.info("Upload an audio file to see its spectrogram.")
