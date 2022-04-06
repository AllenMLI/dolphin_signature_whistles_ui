import os
import sys
import librosa
import streamlit as st

# Internal packages
from dolphin.augment import waveform_augment, mixture_augment
import dolphin.io_utils as io_utils
import dolphin.preprocess.feature_extraction as feature_extraction


def main():

    # This is where visualizations will be saved
    output_dir = 'outputs/ui/augmentations/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {}
    st.header("Visualize the different kinds of augmentation that we used to train our models!")
    uploaded_data = st.file_uploader("Choose a single **short** audio file (1-5 seconds)", accept_multiple_files=False, type=['wav'])

    # Sidebar user input options - they can change the augmentation parameters from the default
    st.sidebar.title('Augmentation Settings')
    sr = int(st.sidebar.selectbox(
        'Sample Rate?', (60000, 90000, 48000, 22500, 16000)
    ))
    pitchup = int(st.sidebar.text_input('How many steps do you want to shift the pitch up?', 1))
    pitchdown = int(st.sidebar.text_input('How many steps do you want to shift the pitch down?', -1))
    speedup = float(st.sidebar.text_input('How much would you like to speed up the wav?', 1.25))
    slowdown = float(st.sidebar.text_input('How much would you like to slow down the wav?', 0.75))
    randnoise = float(st.sidebar.text_input('How much random noise would you like to add?', 0.005))
    ebrs = st.sidebar.multiselect(
        'What event-to-background ratios (ebrs) are you interested in viewing for mixture augmentation?',
        [-12, -6, 0, 6, 12,"all"], "all"
    )
    if "all" in ebrs:
        ebrs = [-12, -6, 0, 6, 12]

    aug_button = st.button("Generate Augmentations")
    st.warning('Do NOT click on the expander buttons, it will take you to another page and all progress will be lost. Use your trackpad to zoom in and out.')
    if aug_button:

        # Load in the wav file
        cfg = {"preprocess": {"nfft": 1024, "spectrogram_max_length": 3, "window": "hamming", "contrast_percentile": 50, "dynamic_range": 80, "sampling_rate": sr},
                "output": {"inches_per_sec": 2, "inches_per_KHz": 0.1, "color_map": "YlGnBu_r"} }
        wav, _ = librosa.load(uploaded_data, sr=sr)
        dur = librosa.get_duration(wav, sr)

        # Generate spectrogram of original wav file
        spec, f, t = feature_extraction.compute_spectrogram(wav, sr=sr, cfg=cfg, random_pad=False)
        io_utils.save_fig(spec, f, t, output_dir=output_dir+'original_spec.png', cfg=cfg)
        st.image(output_dir+'original_spec.png')
        st.caption("Spectrogram of the original wav file")
        
        st.header("Pitch Augmentations")        
        col1, col2 = st.columns(2)

        # Pitch up and down
        pitch_up = waveform_augment.augment_waveform(wav, sr, 'shiftpitchup', pitchup)
        pitch_up, f, t = feature_extraction.compute_spectrogram(pitch_up, sr=sr, cfg=cfg, random_pad=False)
        io_utils.save_fig(pitch_up, f, t, output_dir=output_dir+'pitch_up.png', cfg=cfg)
        col1.image(output_dir+'pitch_up.png')
        col1.caption("Spectrogram of the wav with pitch shifted up")

        pitch_down = waveform_augment.augment_waveform(wav, sr, 'shiftpitchdown', pitchdown)
        pitch_down, f, t = feature_extraction.compute_spectrogram(pitch_down, sr=sr, cfg=cfg, random_pad=False)
        io_utils.save_fig(pitch_down, f, t, output_dir=output_dir+'pitch_down.png', cfg=cfg)
        col2.image(output_dir+'pitch_down.png')
        col2.caption("Spectrogram of the wav with pitch shifted down")

        st.header("Speed Augmentations")
        col1, col2 = st.columns(2)

        # Speed slow and fast
        slow_down = waveform_augment.augment_waveform(wav, sr, 'slowdown', slowdown)
        slow_down, f, t = feature_extraction.compute_spectrogram(slow_down, sr=sr, cfg=cfg, random_pad=False)
        io_utils.save_fig(slow_down, f, t, output_dir=output_dir+'slow_down.png', cfg=cfg)
        col1.image(output_dir+'slow_down.png')
        col1.caption("Spectrogram of the wav speed slowed down")

        speed_up = waveform_augment.augment_waveform(wav, sr, 'speedup', speedup)
        speed_up, f, t = feature_extraction.compute_spectrogram(speed_up, sr=sr, cfg=cfg, random_pad=False)
        io_utils.save_fig(speed_up, f, t, output_dir=output_dir+'speed_up.png', cfg=cfg)
        col2.image(output_dir+'speed_up.png')
        col2.caption("Spectrogram of the wav speed sped up")

        # Add random, normal noise
        st.header("Random Noise Augmentation")
        rand_noise = waveform_augment.augment_waveform(wav, sr, 'addrandomnoise', randnoise)
        rand_noise, f, t = feature_extraction.compute_spectrogram(rand_noise, sr=sr, cfg=cfg, random_pad=False)
        io_utils.save_fig(rand_noise, f, t, output_dir=output_dir+'rand_noise.png', cfg=cfg)
        st.image(output_dir+'rand_noise.png')
        st.caption("Spectrogram of the wav with random normal noise added")

        # Mixing in background noise, a lot or a little
        st.header("Mixture Augmentations")

        max_dur = cfg['preprocess']['spectrogram_max_length']
        pad_wav = mixture_augment.to_shape(wav, (max_dur * sr) + 1, True)  # pad or crop the wav to be max_dur seconds (to match the bg_audio)

        bg_files = mixture_augment.find_bg_files('data/app/bg_audio_examples/')
        bg_audio = mixture_augment.get_bg_clip(bg_files, pad_wav, max_dur, sr)
        
        # Display what the background spectrogram looks like alone
        bg_spec, f, t = feature_extraction.compute_spectrogram(bg_audio, sr=sr, cfg=cfg, random_pad=False)
        io_utils.save_fig(bg_spec, f, t, output_dir=output_dir+'bg_spec.png', cfg=cfg)
        st.image(output_dir+'bg_spec.png')
        st.caption("Spectrogram of the background audio, alone")

        col1, col2 = st.columns(2)

        # For each EBR the user was interested in, mix the whistle with the background...
        for i,ebr in enumerate(ebrs):
            ebr_wav = mixture_augment.mix(bg_audio, wav, pad_wav, int(ebr), sr)
            ebr_spec, f, t = feature_extraction.compute_spectrogram(ebr_wav, sr=sr, cfg=cfg, random_pad=False)
            io_utils.save_fig(ebr_spec, f, t, output_dir=output_dir+'ebr_'+str(ebr)+'.png', cfg=cfg)
            if i % 2 == 0:
                col1.image(output_dir+'ebr_'+str(ebr)+'.png')
                col1.caption("Spectrogram of the wav with background noise overlaid, event-to-background-ratio="+str(ebr))
            else:
                col2.image(output_dir+'ebr_'+str(ebr)+'.png')          
                col2.caption("Spectrogram of the wav with background noise overlaid, event-to-background-ratio="+str(ebr))            
