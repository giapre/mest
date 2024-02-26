'''Code taken from Wang et al. 2023 Science Translational Medicine'''
import numpy as np
import matplotlib.pyplot as plt
import re
def seeg_ch_name_split(nm):
    """
    Split an sEEG channel name into its electrode name and index
    >>> seeg_ch_name_split('GPH10')
    ('GPH', 10)
    """
    try:
        elec, idx = re.match(r"([A-Za-z']+)(\d+)", nm).groups()
    except AttributeError as exc:
        return None
    return elec, int(idx)

def bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names,is_minus=True):
    #from icdc import seeg_ch_name_split
    split_names = [seeg_ch_name_split(el) for el in seeg_xyz_names]
    bip_gain_rows = []
    bip_xyz = []
    bip_names = []
    for i in range(len(split_names) - 1):
        try:
            name, idx = split_names[i]
            next_name, next_idx = split_names[i + 1]
            if name == next_name:
                if is_minus:
                    bip_gain_rows.append(gain[i + 1] - gain[i])
                else:
                    bip_gain_rows.append((gain[i + 1] + gain[i]) / 2.0)
                bip_xyz.append(
                    [(p + q) / 2.0 for p, q in zip(seeg_xyz[i][1], seeg_xyz[i + 1][1])]
                )
                bip_names.append("%s%d-%d" % (name, idx, next_idx))
        except Exception as exc:
            print(exc)
    # abs val, envelope/power always postive
    bip_gain = np.abs(np.array(bip_gain_rows))
    bip_xyz = np.array(bip_xyz)
    return bip_gain, bip_xyz, bip_names

# Plot time series
def plot_ts(t, y):
    # Normalize the time series to have nice plots
    y /= (np.max(y, 0) - np.min(y, 0))
    y -= np.mean(y, 0)

    plt.figure(figsize=(10,10))
    plt.plot(t[:], y[:, 0, :, 0] + 6, 'C3', label='x1')
    plt.plot(t[:], y[:, 1, :, 0] + 4.5, 'C1', label='y1')
    plt.plot(t[:], y[:, 2, :, 0] + 3, 'C4', label='x2')
    plt.plot(t[:], y[:, 3, :, 0] + 1.5, 'C7', label='y2')
    plt.plot(t[:], y[:, 4, :, 0], 'C2', label='z')
    plt.title("Epileptors time series", fontsize=15)
    plt.xlabel('Time [ms]', fontsize=15)
    plt.legend()
    #yticks(np.arange(len(labels)), labels, fontsize=15)
    plt.show()

def plot_ts_3d(t, y):
    # Normalize the time series to have nice plots
    y /= (np.max(y, 0) - np.min(y, 0))
    y -= np.mean(y, 0)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[:, 2, 0, 0]-y[:, 0, 0, 0], y[:, 1, 0, 0], y[:, 4, 0, 0], linewidth = 0.4)
    ax.set_xlabel('x2-10*x1', size=7)
    ax.set_ylabel('y2', size=7)
    ax.set_zlabel('z', size=7)
    ax.set_title('Epileptor time series in 3D')
    plt.show()