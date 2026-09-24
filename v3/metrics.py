import numpy as np
import torch


def zscore_numpy(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def zscore_tensor(x):
    x = x.float()
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True, unbiased=False) + 1e-8)


def paired_sums(real, fake):
    if real.shape != fake.shape or real.ndim not in (2, 3) or not len(real):
        raise ValueError('Expected matching nonempty arrays')
    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError('Nonfinite evaluation data')
    error = zscore_numpy(real) - zscore_numpy(fake)
    return float(np.abs(error).sum()), float(np.square(error).sum()), error.size


def finish_metrics(absolute, squared, count):
    if count <= 0:
        raise ValueError('Cannot evaluate an empty dataset')
    return {'MAE': float(absolute / count), 'RMSE': float((squared / count) ** 0.5)}


def calculate_fd_for_small_sample(gt, fake, pca_dim=64, eps=1e-4, n_trials=1):
    from scipy.linalg import sqrtm
    from sklearn.decomposition import PCA

    fds = []
    for _ in range(n_trials):
        gt_flat = gt.reshape(gt.shape[0], -1)
        fake_flat = fake.reshape(fake.shape[0], -1)
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim, svd_solver='auto', whiten=False)
            combined = np.concatenate([gt_flat, fake_flat], axis=0)
            reduced = pca.fit_transform(combined)
            gt_flat = reduced[:gt.shape[0]]
            fake_flat = reduced[gt.shape[0]:]
        mu_gt, mu_fake = gt_flat.mean(axis=0), fake_flat.mean(axis=0)
        cov_gt, cov_fake = np.cov(gt_flat, rowvar=False), np.cov(fake_flat, rowvar=False)
        dim = cov_gt.shape[0]
        cov_gt = cov_gt + eps * np.eye(dim)
        cov_fake = cov_fake + eps * np.eye(dim)
        ssdiff = np.sum((mu_gt - mu_fake) ** 2.)
        covmean, _ = sqrtm(cov_gt.dot(cov_fake), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fds.append(float(ssdiff + np.trace(cov_gt) + np.trace(cov_fake) - 2. * np.trace(covmean)))
    return float(np.mean(fds)), float(np.std(fds))


def get_Rpeaks_ECG(filtered, sampling_rate):
    from biosppy.signals import ecg as ecg_func

    rpeaks, = ecg_func.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
    rpeaks, = ecg_func.correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05)
    templates, rpeaks = ecg_func.extract_heartbeats(signal=filtered, rpeaks=rpeaks,
                                                 sampling_rate=sampling_rate, before=0.2, after=0.4)
    return rpeaks, np.diff(rpeaks)


def heartbeats_ecg(filtered, sampling_rate):
    from biosppy.signals import tools

    rpeaks, rr_intervals = get_Rpeaks_ECG(filtered, sampling_rate)
    if rr_intervals.size != 0:
        hr_idx, hr = tools.get_heart_rate(beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3)
        if len(hr) == 0:
            hr_idx, hr = [-1], [-1]
    else:
        hr_idx, hr = [-1], [-1]
    return hr_idx, hr


def ecg_bpm_array(ecg_signal, sampling_rate=128, window=4, filter=False):
    import neurokit2 as nk

    final_bpm = []
    for k in ecg_signal:
        if isinstance(k, np.ndarray) and k.ndim > 1:
            k = np.squeeze(k)
        if filter is True:
            k = nk.ecg_clean(k, sampling_rate=128, method='pantompkins1985')
        hr_idx, hr = heartbeats_ecg(k, sampling_rate)
        final_bpm.append(np.mean(hr))
    return np.array(final_bpm)


def heart_rate_metrics(real_bpm, fake_bpm):
    valid = (~np.isnan(fake_bpm)) & (~np.isnan(real_bpm))
    paired = float(np.mean(np.absolute(real_bpm[valid] - fake_bpm[valid])))
    real_group, fake_group = real_bpm[real_bpm > 0], fake_bpm[fake_bpm > 0]
    group = float(abs(np.nanmean(real_group) - np.nanmean(fake_group)))
    return {'MAE_hr_paired': paired, 'MAE_hr_group': group}
