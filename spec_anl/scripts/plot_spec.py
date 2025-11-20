import sys
import matplotlib.pyplot as plt
import librosa.display  # optional helper
from spec_core import compute_spectrogram

import matplotlib.pyplot as plt
import librosa.display
from spec_core import compute_spectrogram

def plot_spectrogram(
    audio_path,
    use_mel=True,
    sr_target=16000,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    window="hann",
    to_db=True,
):
    S, freqs, times, sr = compute_spectrogram(
        audio_path=audio_path,
        sr_target=sr_target,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        use_mel=use_mel,
        n_mels=n_mels,
        to_db=to_db,
    )

    plt.figure(figsize=(10, 4))

    if use_mel:
        librosa.display.specshow(
            S,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            fmax=sr / 2,
        )
        plt.ylabel("Mel frequency")
    else:
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        img = plt.imshow(
            S,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        plt.ylabel("Frequency in Hz")
        plt.colorbar(img, label="dB" if to_db else "Amplitude")

    plt.xlabel("Time in seconds")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_spec.py path_to_audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Try a mel spectrogram by default
    plot_spectrogram(
        audio_path,
        sr_target=16000,
        n_fft=2048,
        hop_length=512,
        window="hann",
        use_mel=True,
        n_mels=128,
        to_db=True,
    )


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_reference_style_spectrogram(
    audio_path,
    sr_target=16000,
    n_fft=2048,
    hop_length=512,
):
    # load audio
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)

    # complex STFT
    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )

    # power spectrogram
    S_power = np.abs(D) ** 2

    # convert to dB
    S_db = librosa.power_to_db(S_power, ref=np.max)

    plt.figure(figsize=(8, 4))

    # log frequency axis, time on x
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
    )

    plt.title("Reference power spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Hz")

    # color bar on the right with dB ticks
    cbar = plt.colorbar(img, format="%+2.0f dB")
    cbar.set_label("dB")

    # frequency ticks like in the reference image
    plt.yticks([64, 128, 256, 512, 1024, 2048, 4096, 8192])

    plt.tight_layout()
    plt.show()
