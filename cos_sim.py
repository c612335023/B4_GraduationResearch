import librosa
import numpy as np
import csv
from multiprocessing import Pool

def extract_chroma(file_path):
    y, sr = librosa.load(file_path)
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    return chroma

def pitch_percentage(chroma):
    chroma_sum = np.sum(chroma, axis=1)
    return chroma_sum/np.sum(chroma_sum)

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def process(i):
    file_path1 = "dataset/wav_tra/tra"+str(i)+".wav"
    file_path2 = "dataset/wav_conventional/conventional"+str(i)+".wav"
    file_path3 = "dataset/wav_proposed1/proposed"+str(i)+".wav"

    chroma1 = extract_chroma(file_path1)
    chroma2 = extract_chroma(file_path2)
    chroma3 = extract_chroma(file_path3)

    chroma1 = pitch_percentage(chroma1)
    chroma2 = pitch_percentage(chroma2)
    chroma3 = pitch_percentage(chroma3)

    chroma1 = normalize(chroma1)
    chroma2 = normalize(chroma2)
    chroma3 = normalize(chroma3)

    dos1 = cosine_similarity(chroma1, chroma2)
    dos2 = cosine_similarity(chroma1, chroma3)

    return [i, dos1, dos2]

if __name__ == "__main__":
    with Pool(processes=4) as p:
        rows = list(p.map(process, range(254)))
    with open("cos_sim.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)