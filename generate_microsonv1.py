import masp as srs
import numpy as np
import soundfile as sf
from IPython.display import Audio
import scipy.signal as sig
import copy
import pandas as pd
import os
import json
from os.path import join as pjoin
from multiprocessing import Pool
import matplotlib.pyplot as plt
import mat73
import pyrubberband as pyrb
from multiprocessing import Pool
import helpers as hlp
import importlib
importlib.reload(hlp);

#a = df.iloc[i]
def process(a):

    try:
        mic = np.array(hlp.head_2_ku_ears(np.array([a.headC_x, a.headC_y, a.headC_z]),
                                            np.array([a.headOrient_azi,a.headOrient_ele])))
        # load noise:
        noise, _ = sf.read(pjoin(pjoin(pjoin(wham_path, 'wham_noise'), a.wham_split), a.noise_path))

        # time stretch if needed
        if a.stretch != 0.0:
            noise = pyrb.time_stretch(noise, a.fs_noise, a.stretch)

        # extend if needed with hanning window
        noise = np.array([hlp.extend_noise(noise[:,0], a.num_chunks * 4 * a.fs_noise, a.fs_noise),
                hlp.extend_noise(noise[:,1], a.num_chunks * 4 * a.fs_noise, a.fs_noise)]).T
        # crop 4 seconds chunk
        noise = noise[a.chunk * 4 * a.fs_noise:(a.chunk + 1) * 4 * a.fs_noise]

        # invert phase for augmentation
        if a.phase_inv:
            noise *= -1

        # invert channels for augmentation
        if a.lr_inv:
            noise = noise[:, [1,0]]

        noise = noise.T
        
        # load speech and crop at the 4s chunk that has more energy
        speech_folder = pjoin(pjoin(mls_path, a.mls_split), 'audio')
        speech, _ = sf.read(pjoin(pjoin(pjoin(speech_folder, str(a.speaker)), str(a.book)), a.speech_path))
        env = sig.fftconvolve(speech, np.ones(4*a.fs_noise), 'same')
        idx_candidates = np.flip(np.argsort(env**2))
        idx = idx_candidates[idx_candidates < (len(speech)-(4*a.fs_noise))][0]
        speech = speech[idx:idx+4*a.fs_noise]
        
        room = np.array([a.room_x, a.room_y, a.room_z])
        rt60 = np.array([a.rt60])
        rt60 *= 0.5#furniture absorption? 
        #snr 0, more people, more reduction -> 0.3 * rt60
        #snr 5, less people, no rt60 reduction -> 1.0 * rt60
        rt60 *= ((a.snr+0.3)/5.3) # people absoprtion
        src = np.array([[a.src_x, a.src_y, a.src_z]])
        head_orient = np.array([a.headOrient_azi, a.headOrient_ele])

        # Compute absorption coefficients for desired rt60 and room dimensions
        abs_walls,rt60_true = srs.find_abs_coeffs_from_rt(room, rt60)
        # Small correction for sound absorption coefficients:
        if sum(rt60_true-rt60>0.05*rt60_true)>0 :
            abs_walls,rt60_true = srs.find_abs_coeffs_from_rt(room, rt60_true + abs(rt60-rt60_true))

        # Generally, we simulate up to RT60:
        limits = np.minimum(rt60, maxlim)

        abs_echograms = srs.compute_echograms_sh(room, src, mic, abs_walls, limits, ambi_order, rims_d, head_orient)
        ane_echograms = hlp.crop_echogram(copy.deepcopy(abs_echograms))
        mic_rirs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs)
        ane_rirs = srs.render_rirs_sh(ane_echograms, band_centerfreqs, fs)
        #hlp.plot_scene(room,np.array([a.headC_x, a.headC_y, a.headC_z]),head_orient,mic,src,perspective="xy")
        bin_ir = np.array([sig.fftconvolve(np.squeeze(mic_rirs[:,:,0, 0]), decoder[:,:,0], 'full', 0).sum(1),
                            sig.fftconvolve(np.squeeze(mic_rirs[:,:,1, 0]), decoder[:,:,1], 'full', 0).sum(1)])
        bin_aneIR = np.array([sig.fftconvolve(np.squeeze(ane_rirs[:,:,0, 0]), decoder[:,:,0], 'full', 0).sum(1),
                            sig.fftconvolve(np.squeeze(ane_rirs[:,:,1, 0]), decoder[:,:,1], 'full', 0).sum(1)])
        reverberant_src = np.array([sig.fftconvolve(speech, bin_ir[0, :], 'same'), sig.fftconvolve(speech, bin_ir[1, :], 'same')])
        anechoic_src = np.array([sig.fftconvolve(speech, bin_aneIR[0, :], 'same'), sig.fftconvolve(speech, bin_aneIR[1, :], 'same')])
        ini_snr = 10 * np.log10(hlp.power(reverberant_src) / hlp.power(noise) + np.finfo(noise.dtype).resolution)
        noise_gain_db = ini_snr - a.snr

        noise = noise * np.power(10, noise_gain_db/20)
        norm_fact = np.max(np.abs(reverberant_src + noise))

        anechoic_src /= norm_fact
        noise /= norm_fact
        reverberant_src /= norm_fact

        anechoic_src *= 0.99
        noise *= 0.99
        reverberant_src *= 0.99
        
        writepath = pjoin(output_path, a.mls_split)
        sf.write(pjoin(pjoin(writepath, 'anechoic'), os.path.splitext(a.speech_path)[0]+'.wav'), anechoic_src.T, fs, subtype='FLOAT')
        sf.write(pjoin(pjoin(writepath, 'reverberant'), os.path.splitext(a.speech_path)[0]+'.wav'), reverberant_src.T, fs, subtype='FLOAT')
        sf.write(pjoin(pjoin(writepath, 'noise'), os.path.splitext(a.speech_path)[0]+'.wav'), noise.T, fs, subtype='FLOAT')
        sf.write(pjoin(pjoin(writepath, 'ir'), os.path.splitext(a.speech_path)[0]+'.wav'), bin_ir.T, fs, subtype='FLOAT')
        sf.write(pjoin(pjoin(writepath, 'ane_ir'), os.path.splitext(a.speech_path)[0]+'.wav'), bin_aneIR.T, fs, subtype='FLOAT')
        print('Processed ' + str(a.idx))
    except:
        print('ERROR when processing ' + str(a.idx))

if __name__ == '__main__':
    num_workers = 8

 #   decoder_path = 'ku100_inear_test.mat'
    decoder_path = 'ku100_ha_test.mat'

    mls_path = '/home/ubuntu/Data/mls_spanish'
    wham_path = '/home/ubuntu/Data/wham'
    output_path = '/home/ubuntu/Data/microson_v1/'
    df_path = 'meta_microson_v1.csv'
    df = pd.read_csv(df_path)
    fs = 16000
    ambi_order = 10
    rims_d = .0
    maxlim = 2.
    band_centerfreqs=np.array([1000])
    decoder = mat73.loadmat(decoder_path)['hnm']
    # we copy the metadata file to the output dir
    df.to_csv(pjoin(output_path, 'meta_microson_v1.csv'), index=False)
    # also save the configuration:
    config = {'mls_path' : mls_path, 'wham_path' : wham_path, 
              'decoder_path' : decoder_path, 'df_path' : df_path,
              'fs' : fs, 'ambi_order': ambi_order, 'success': False}
    with open(pjoin(output_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    with Pool(num_workers) as p:
        p.map(process, [df.iloc[i] for i in range(len(df))])
    
    config['success'] = True
    with open(pjoin(output_path, 'config.json'), 'w') as f:
        json.dump(config, f)
    print('All files processed. Done.')
