import torch
import numpy as np


def load_model():
  ''' Loads the model from the github repo and obtains just the feature encoder. '''

  repo = 'OxWearables/ssl-wearables'
  # class_num não interessa para extrair features; mas o hub pede este arg
  model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
  model.eval()

  # Passo crucial: ficar só com a parte auto-supervisionada
  # O README diz que há um 'feature_extractor' (pré-treinado) e um 'classifier' (não treinado). :contentReference[oaicite:14]{index=14}
  feature_encoder = model.feature_extractor
  feature_encoder.to("cpu")
  feature_encoder.eval()

  return feature_encoder


def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """
    acc_xyz: np.ndarray shape (N, 3) em m/s^2 (ou g), amostrado a fs_in_hz (float)
    devolve:
      acc_resampled: np.ndarray shape (M, 3) já a 30 Hz
      fs_target: 30.0
    """
    fs_target = 30.0
    win_size = 5 # in seconds
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz
    t_out = np.arange(0, win_size, 1.0/fs_target)

    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    return acc_resampled, fs_target


def acc_segmentation(data_np, window_sec=5, fs_in_hz=51.5):
    """
    Segmenta dados do acelerómetro em janelas de 5s.
    Agora devolve:
        - segments: lista de arrays (Nx3)
        - labels: atividade associada à janela
        - starts: índice de início da janela no dataset original
    """
    acc_x = data_np[:, 1]
    acc_y = data_np[:, 2]
    acc_z = data_np[:, 3]
    activities = data_np[:, -1]

    fs = fs_in_hz
    win_size = int(window_sec * fs)

    segments = []
    labels = []
    starts = []

    i = 0
    while i + win_size <= len(acc_x):
        act_window = activities[i:i+win_size]
        # janela válida se a atividade for constante
        if np.all(act_window == act_window[0]):
            seg = np.vstack([
                acc_x[i:i+win_size],
                acc_y[i:i+win_size],
                acc_z[i:i+win_size]
            ]).T  # (N,3)

            segments.append(seg)
            labels.append(int(act_window[0]))
            starts.append(i)

        i += win_size

    return segments, np.array(labels), np.array(starts)
