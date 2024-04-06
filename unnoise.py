from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import librosa
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Load default model
    model, df_state, _ = init_df()
    # Download and open some audio file. You use your audio files here
    audio_path = download_file(
        "https://github.com/Rikorose/DeepFilterNet/raw/e031053/assets/noisy_snr0.wav",
        download_dir=".",
    )
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    # Denoise the audio
    enhanced = enhance(model, df_state, audio)
    # Save for listening
    save_audio("enhanced.wav", enhanced, df_state.sr())


#     # Load the audio file (in this example, 'your_audio_file.mp3')
# audio_path = "C:\\Users\\behra\\OneDrive\\Desktop\\Sem.6\\Signal ha o systemha\\Project\\noisy_snr0.wav"
# # audio_path2="C:\\Users\\behra\\OneDrive\\Desktop\\Sem.6\\Signal ha o systemha\\Project\\enchanced.wav"
# y, sr = librosa.load(audio_path)
# # y, sr = librosa.load(audio_path2)


# # Create a time axis in seconds
# time = np.arange(0, len(y)) / sr

# # Plot the waveform
# plt.figure(figsize=(14, 5))
# plt.plot(time, y, linewidth=0.3, color='b')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Waveform')
# plt.show()
    



   # Load the first audio file
audio_path_1 = "C:\\Users\\behra\\OneDrive\\Desktop\\Sem.6\\Signal ha o systemha\\Project\\noisy_snr0.wav"
y1, sr1 = librosa.load(audio_path_1)

# Load the second audio file
audio_path_2 = "C:\\Users\\behra\\OneDrive\\Desktop\\Sem.6\\Signal ha o systemha\\Project\\enhanced.wav"
y2, sr2 = librosa.load(audio_path_2)

# Create time axes in seconds for both audio files
time1 = np.arange(0, len(y1)) / sr1
time2 = np.arange(0, len(y2)) / sr2

# Plot the waveforms of both audio files
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(time1, y1, linewidth=0.3, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform 1')

plt.subplot(1, 2, 2)
plt.plot(time2, y2, linewidth=0.3, color='r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform 2')

# plt.tight_layout()
plt.show()
