import hera_sim
from hera_sim import Simulator
import numpy as np
import datetime
import pickle
from matplotlib import pyplot as plt
import os


# ---------------------------------------------------------------------------------------------------------------------
def simulate(sim: Simulator):

    sim.refresh()
    if np.random.uniform(0) < 0.5:
        hera_sim.defaults.set('h1c')
    else:
        hera_sim.defaults.set('h2c')

    nonrfi_components = []

    # ########################################  FOREGROUNDS  ################################################

    sim.add(
        'diffuse_foreground',
        component_name='diffuse_foreground',
        seed='redundant',
    )
    nonrfi_components.append('diffuse_foreground')

    sim.add(
        'pntsrc_foreground',
        component_name='pntsrc_foreground',
        seed='redundant',
        nsrcs=1000,
        Smin=0.3,
        Smax=1000,
        beta=-1.5,
        spectral_index_mean=-1,
        spectral_index_std=0.5,
        reference_freq=np.random.uniform(0.35, 0.55)
    )
    nonrfi_components.append('pntsrc_foreground')

    # ########################################  RFI  ################################################

    f_min = sim.freqs.min()
    f_max = sim.freqs.max()
    ch_width = (f_max - f_min) / sim.Nfreqs

    rfi_components = []

    # ------------------------------ RFI DTV ------------------------------

    n_dtv = int(np.random.uniform(0, 1) * 6) + 0
    for i in range(n_dtv):
        component_name = f'rfi_dtv_{i}'
        sim.add(
            "rfi_dtv",
            dtv_band=(f_min, f_max),
            dtv_channel_width=np.random.uniform(5*ch_width, 64*ch_width),
            dtv_chance=np.abs(np.random.normal(0.0001, 0.002)),
            dtv_strength=np.random.uniform(200, 20000),
            dtv_std=np.random.uniform(100, 30000),
            component_name=component_name,
            seed="redundant",
        )
        rfi_components.append(component_name)
    # ------------------------------ RFI IMPULSE ------------------------------

    n_impulse = int(np.random.uniform(0, 1) * 6) + 0
    for i in range(n_impulse):
        component_name = f'rfi_impulse_{i}'
        sim.add(
            "rfi_impulse",
            impulse_chance=np.abs(np.random.normal(0.0001, 0.002)),
            impulse_strength=np.random.uniform(100, 40000),
            component_name=component_name,
            seed="redundant",
        )
        rfi_components.append(component_name)

    # ------------------------------ RFI SCATTER ------------------------------
    # pixel blips
    n_scatter = int(np.random.uniform(0, 1) * 6) + 0
    for i in range(n_scatter):
        component_name = f'rfi_scatter_{i}'
        sim.add(
            "rfi_scatter",
            component_name=component_name,
            seed="redundant",
            scatter_chance=np.abs(np.random.normal(0.0001, 0.002)),
            scatter_strength=np.random.uniform(200, 20000),
            scatter_std=np.random.uniform(100, 30000),
        )
        rfi_components.append(component_name)

    # ------------------------------ RFI STATIONS ------------------------------

    # + 2 * 0.09 / 512
    n_stations = int(np.random.uniform(0, 1) * 6) + 0
    stations = []
    for i in range(n_stations):
        st = hera_sim.rfi.RfiStation(f0=np.random.uniform(f_min, f_max),
                                    duty_cycle=np.random.uniform(0.1, 0.9),
                                    std=np.random.uniform(100, 30000),
                                    strength=np.random.uniform(200, 20000),
                                    timescale=np.random.uniform(30, 300))
        stations.append(st)
    if stations:
        sim.add('stations', stations=stations, seed='redundant', component_name='rfi_stations')
        rfi_components.append('rfi_stations')

    # ########################################  ADDITIONAL EFFECTS  ################################################

    sim.add("thermal_noise",
            seed="initial",
            Trx=60,
            component_name="thermal_noise")
    nonrfi_components.append('thermal_noise')

    sim.add("whitenoisecrosstalk", amplitude=20, seed="redundant", component_name='whitenoisecrosstalk')
    nonrfi_components.append('whitenoisecrosstalk')

    sim.add("bandpass", gain_spread=0.1, dly_rng=(-20, 20))

    return sim, nonrfi_components, rfi_components


# ---------------------------------------------------------------------------------------------------------------------
def get_images_masks(sim: Simulator, rfi_components, antpairpols=None, incl_phase=True, dtype='float32') -> (np.ndarray, np.ndarray):

    if antpairpols is None:
        antpairpols = sim.get_antpairpols()

    if rfi_components is None:
        raise TypeError('names may not be None')
    if isinstance(rfi_components, str):
        rfi_components = [rfi_components]
    if not isinstance(rfi_components, (list, tuple)):
        raise TypeError('names must be in a list or tuple')

    masks = []
    images = []
    shape = sim.get_data(antpairpols[0]).shape  # 2 dims: Nlsts, Nfreqs
    for pair in antpairpols:
        mask = np.zeros(shape, dtype=bool)
        for name in rfi_components:
            im = np.abs(sim.get(name, pair))
            mask = np.logical_or(mask, im > 0)
        masks.append(mask)
        images.append(sim.get_data(pair))

    if incl_phase:
        return np.concatenate(
            [np.expand_dims(np.abs(images), axis=-1),
             np.expand_dims(np.angle(images), axis=-1)],
            axis=3, dtype=dtype
        ), np.expand_dims(np.array(masks), axis=-1)
    else:
        return np.expand_dims(np.array(np.abs(images), dtype=dtype), axis=-1), np.expand_dims(np.array(masks), axis=-1)


# ---------------------------------------------------------------------------------------------------------------------
def simulate_and_save(sim, sample_ratio, n_sims, incl_phase, test_split=0.2):

    all_antpairpols = sim.get_antpairpols()
    # n_baselines = int(len(all_antpairpols) * sample_ratio)
    n_baselines = 1
    if incl_phase:
        images_shape = (n_baselines*n_sims, sim.Ntimes, sim.Nfreqs, 2)
    else:
        images_shape = (n_baselines*n_sims, sim.Ntimes, sim.Nfreqs, 1)
    masks_shape = (n_baselines*n_sims, sim.Ntimes, sim.Nfreqs, 1)
    images = np.empty(images_shape, dtype='float32')
    masks = np.empty(masks_shape, dtype='bool')
    antpairpol_list = []
    for i in range(n_sims):
        lo = i * n_baselines
        hi = (i + 1) * n_baselines

        # Randomly sample baselines
        indexes = np.random.choice(np.arange(len(all_antpairpols)), n_baselines, replace=False)
        antpairpols = [all_antpairpols[i] for i in indexes]

        # Simulate and extract images and masks
        sim, nonrfi_components, rfi_components = simulate(sim)
        print(rfi_components)
        images[lo:hi, ...], masks[lo:hi, ...] = get_images_masks(sim, rfi_components, antpairpols, incl_phase)
        antpairpol_list = antpairpol_list + antpairpols
        print(f'Sim: {i}')

    # Save stats and images
    # dir_path = './hera_images'
    # save_mag_phase_masks_batches(dir_path, images, masks)

    # Split data
    print(images.shape, masks.shape)
    n_test = int(images.shape[0] * test_split)
    np.random.RandomState(seed=42).shuffle(images)
    np.random.RandomState(seed=42).shuffle(masks)
    np.random.RandomState(seed=42).shuffle(antpairpol_list)
    test_data = images[:n_test]
    train_data = images[n_test:]
    test_masks = masks[:n_test]
    train_masks = masks[n_test:]
    test_antpairpol_list = antpairpol_list[:n_test]
    train_antpairpol_list = antpairpol_list[n_test:]
    
    # Save to pickle
    f_name = '../../data/HERA_{}.pkl'.format(
        datetime.datetime.now().strftime("%d-%m-%Y"))
    pickle.dump([train_data, train_masks, test_data, test_masks], open(f_name, 'wb'), protocol=4)
    print('{} saved!'.format(f_name))
    
    f_name_antpairpols = '../../data/HERA_{}_antpairpols.pkl'.format(
    datetime.datetime.now().strftime("%d-%m-%Y"))
    with open(f_name_antpairpols, 'wb') as file:
        pickle.dump([train_antpairpol_list, test_antpairpol_list], file)


# ---------------------------------------------------------------------------------------------------------------------
def save_mag_phase_masks_batches(dir_path, data, masks, batch_size=20, figsize=(20, 60)):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i in range(0, len(data), batch_size):
        strt = i
        fnsh = np.minimum(strt + batch_size, len(data))
        d = data[strt:fnsh, ...]
        m = masks[strt:fnsh, ...]

        fig, ax = plt.subplots(len(d), data.shape[-1] + 1, figsize=figsize)

        ax[0, 0].title.set_text('Magnitude')
        if data.shape[-1] == 2:
            ax[0, 1].title.set_text('Phase')
            ax[0, 2].title.set_text('Mask')
        else:
            ax[0, 1].title.set_text('Mask')

        for j in range(len(d)):
            ax[j, 0].imshow(np.log(d[j, ..., 0]))
            if data.shape[-1] == 2:
                ax[j, 1].imshow(d[j, ..., 1])
                ax[j, 2].imshow(m[j, ..., 0])
            else:
                ax[j, 1].imshow(m[j, ..., 0])

        plt.tight_layout()

        if data.shape[-1] == 2:
            fig.savefig(os.path.join(dir_path, f'mag_phase_masks_{strt}'))
        else:
            fig.savefig(os.path.join(dir_path, f'mag_masks_{strt}'))

        plt.close('all')


# ---------------------------------------------------------------------------------------------------------------------
def load_hera_pickle(data_path):
    (train_data, train_masks,
    test_data, test_masks) = np.load(f'{data_path}/HERA_28-03-2023_all.pkl', allow_pickle=True)

    train_data[train_data==np.inf] = np.finfo(train_data.dtype).max
    test_data[test_data==np.inf] = np.finfo(test_data.dtype).max

    return train_data.astype('float32'), train_masks, test_data.astype('float32'), test_masks


# ---------------------------------------------------------------------------------------------------------------------
def main():
    sim_params = dict(
        Nfreqs=512,
        start_freq=100e6,
        bandwidth=90e6,
        Ntimes=512,
        start_time=2457458.1738949567,
        integration_time=3.512,
        array_layout=hera_sim.antpos.hex_array(2, split_core=False, outriggers=0),
    )

    sim = Simulator(**sim_params)
    simulate_and_save(sim, sample_ratio=0.0358, n_sims=1200, incl_phase=False, test_split=0.2)


if __name__ == '__main__':
    main()
