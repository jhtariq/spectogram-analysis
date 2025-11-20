import numpy as np
import librosa

import numpy as np
import librosa

def compute_spectrogram(
    audio_path,
    sr_target=16000,
    n_fft=1024,
    hop_length=256,
    window="hann",
    use_mel=False,
    n_mels=128,
    to_db=True,
):
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)

    if use_mel:
        # Mel power spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            n_mels=n_mels,
            power=2.0,
        )
        if to_db:
            S = librosa.power_to_db(S, ref=np.max)
        # For mel we can just keep bin indices and let the plotting handle the scale
        freqs = None
    else:
        # Linear magnitude spectrogram
        stft = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
        )
        S = np.abs(stft)
        if to_db:
            S = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    times = librosa.frames_to_time(
        np.arange(S.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return S, freqs, times, sr
