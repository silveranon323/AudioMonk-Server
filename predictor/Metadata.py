# def getmetadata(filename):
#     import librosa
#     import numpy as np


#     y, sr = librosa.load(filename)
#     #fetching tempo

#     onset_env = librosa.onset.onset_strength(y, sr)
#     tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

#     #fetching beats

#     y_harmonic, y_percussive = librosa.effects.hpss(y)
#     tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)

#     #chroma_stft

#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

#     #rmse

#     rmse = librosa.feature.rms(y=y)

#     #fetching spectral centroid

#     spec_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0]

#     #spectral bandwidth

#     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

#     #fetching spectral rolloff

#     spec_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]

#     #zero crossing rate

#     zero_crossing = librosa.feature.zero_crossing_rate(y)

#     #mfcc

#     mfcc = librosa.feature.mfcc(y=y, sr=sr)

#     #metadata dictionary

#     metadata_dict = {'tempo':tempo,'chroma_stft':np.mean(chroma_stft),'rmse':np.mean(rmse),
#                      'spectral_centroid':np.mean(spec_centroid),'spectral_bandwidth':np.mean(spec_bw), 
#                      'rolloff':np.mean(spec_rolloff), 'zero_crossing_rates':np.mean(zero_crossing)}

#     for i in range(1,21):
#         metadata_dict.update({'mfcc'+str(i):np.mean(mfcc[i-1])})

#     return list(metadata_dict.values())



import librosa
import numpy as np

def getmetadata(filename):
    # Load the audio file
    y, sr = librosa.load(filename)

    # Fetching tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]  # Extract scalar value

    # Fetching beats
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    _, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)  # Ignore returned tempo

    # Chromagram
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # RMSE (Root Mean Square Energy)
    rmse = librosa.feature.rms(y=y)

    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y+0.01, sr=sr)  # Avoid log(0) errors

    # Zero Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Metadata dictionary
    metadata_dict = {
        'tempo': tempo,
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spec_centroid),
        'spectral_bandwidth': np.mean(spec_bw),
        'rolloff': np.mean(spec_rolloff),
        'zero_crossing_rates': np.mean(zero_crossing),
    }

    # Add MFCCs (20 coefficients)
    for i in range(20):
        metadata_dict[f'mfcc{i+1}'] = np.mean(mfcc[i])

    return list(metadata_dict.values())
