import numpy as np
from IPython.display import Audio
import scipy.signal as sig
import soundfile as sf
import matplotlib.pyplot as plt



# ----------- FUNCTION DEFINITIONS: -----------
def power(signal):
    return np.mean(signal**2)

def crop_echogram(anechoic_echogram):
    nSrc = anechoic_echogram.shape[0]
    nRec = anechoic_echogram.shape[1]
    nBands = anechoic_echogram.shape[2]
    # Returns the "anechoic" version of an echogram
    # Should keep the receiver directivy
    for src in range(nSrc):
        for rec in range(nRec):
            for band in range(nBands):
                anechoic_echogram[src, rec, band].time = anechoic_echogram[src, rec, band].time[:2]
                anechoic_echogram[src, rec, band].coords = anechoic_echogram[src, rec, band].coords[:2, :]
                anechoic_echogram[src, rec, band].value = anechoic_echogram[src, rec, band].value[:2,:]
                anechoic_echogram[src, rec, band].order = anechoic_echogram[src, rec, band].order[:2,:]
    return anechoic_echogram


def place_on_circle(head_pos,r,angle_deg):
# place a source around the reference point (like head)
    angle_rad = (90-angle_deg) * (np.pi / 180)
    x_coord=head_pos[0]+r*np.sin(angle_rad)
    y_coord=head_pos[1]+r*np.cos(angle_rad)
    src_pos=np.array([x_coord, y_coord, head_pos[2]]) 
    return [src_pos]

def head_2_ku_ears(head_pos,head_orient):
# based on head pos and orientation, compute coordinates of ears
  ear_distance_ku100=0.0875
  theta = (90-head_orient[0]) * np.pi / 180
  R_ear = [head_pos[0] - ear_distance_ku100 * np.cos(theta),
              head_pos[1] + ear_distance_ku100 * np.sin(theta), 
              head_pos[2]]
  L_ear = [head_pos[0] + ear_distance_ku100 * np.cos(theta),
              head_pos[1] - ear_distance_ku100 * np.sin(theta), 
              head_pos[2]]
  return [L_ear,R_ear]


def add_signals(a,b):
# add values of two arrays of different lengths
  if len(a) < len(b):
    c = b.copy()
    c[:len(a)] += a
  else:
    c = a.copy()
    c[:len(b)] += b
  return c

def plot_scene(room_dims,head_pos,head_orient,l_mic_pos,l_src_pos,perspective="xy"):
#   function to plot the designed scene
#   room_dims - dimensions of the room [x,y,z]
#   head_pos - head position [x,y,z]
#   head_orient - [az,el]
#   l_src_pos - list of source positions [[x,y,z],...,[x,y,z]]
#   perspective - which two dimensions to show 
  if perspective=="xy":
    dim1=1
    dim2=0
  elif perspective=="yz":
    dim1=2
    dim2=1
  elif perspective=="xz":
    dim1=2
    dim2=0
  plt.figure()
  plt.xlim((0,room_dims[dim1]))
  plt.ylim((0,room_dims[dim2]))
  # plot sources and receivers
  plt.plot(head_pos[dim1],head_pos[dim2], "o", ms=10, mew=2, color="black")
  # plot ears
  plt.plot(l_mic_pos[0][dim1],l_mic_pos[0][dim2], "o", ms=3, mew=2, color="blue")# left ear in blue
  plt.plot(l_mic_pos[1][dim1],l_mic_pos[1][dim2], "o", ms=3, mew=2, color="red")# right ear in red

  for i,src_pos in enumerate(l_src_pos):
    plt.plot(src_pos[dim1],src_pos[dim2], "o", ms=10, mew=2, color="red")
    plt.annotate(str(i), (src_pos[dim1],src_pos[dim2]))
  # plot head orientation if looking from above 
  if perspective=="xy":
    plt.plot(head_pos[dim1],head_pos[dim2], marker=(1, 1, -head_orient[0]), ms=20, mew=2,color="black")

def set_level(sig_in,L_des):
# set FS level of the signal
    sig_zeromean=np.subtract(sig_in,np.mean(sig_in,axis=0))
    sig_norm_en=sig_zeromean/np.std(sig_zeromean.reshape(-1))
    sig_out =sig_norm_en*np.power(10,L_des/20)
    print(20*np.log10(np.sqrt(np.mean(np.power(sig_out,2)))))
    return sig_out

def generate_scenes(sources_sigs,levels,mic_rirs,decoder):
# generate binaural mixture signal based on generated irs, binaural decoder and source signals
    sig_L_mix=np.zeros(100)
    sig_R_mix=np.zeros(100)
    for i, source_sig in enumerate(sources_sigs):
        #mic_rirs[:, :, ear, source]
        filter_L=sig.fftconvolve(np.squeeze(mic_rirs[:,:,0, i]).T, decoder[:,:,0].T, 'full', 1).sum(0)
        filter_R=sig.fftconvolve(np.squeeze(mic_rirs[:,:,1, i]).T, decoder[:,:,1].T, 'full', 1).sum(0)
        # set level for current source BEFORE SPATIALIZING:
        source_sig=set_level(source_sig.T,levels[i])
        # spatialize:
        sig_L=sig.fftconvolve(source_sig, filter_L, 'full')
        sig_R=sig.fftconvolve(source_sig, filter_R, 'full')
        # # set level for current source AFTER SPATIALIZING:
        # sig_LR_leveled=set_level(np.array((sig_L,sig_R)).T,levels[i])
        sig_LR_leveled=np.array((sig_L,sig_R)).T
        # add generated source signal to the mixture using a function that takes variable signal lenghts
        sig_L_mix=add_signals(sig_L_mix,sig_LR_leveled[:,0])# left channel
        sig_R_mix=add_signals(sig_R_mix,sig_LR_leveled[:,1])# right channel
        # put left and right signal into one array
        mix=np.array((sig_L_mix,sig_R_mix))

    return mix

def decode_noise(MOA_noise,level,decoder):
    MOA_noise=set_level(MOA_noise.T,level[0])
    # convolve signal in ambisonics domain with a decoder
    # decoder[:,:,0] - for left ear
    # decoder[:,:,1] - for the right ear
    noise_L=sig.fftconvolve(MOA_noise, decoder[:,:,0].T, 'full', 1).sum(0)
    noise_R=sig.fftconvolve(MOA_noise, decoder[:,:,1].T, 'full', 1).sum(0)
    noise=np.array((noise_L,noise_R))
    return noise


    


