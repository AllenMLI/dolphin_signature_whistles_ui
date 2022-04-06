import os
import sys
import csv
import streamlit as st

# Internal packages
sys.path.append('src/')
import dolphin.app.app_detect as app_detect


def write_to_csv(savedir: str, user_labels: list, fps: list, start_times: list, sr: int):
    """
    Takes the model predictions, user boolean labels, and filepaths and writes to csv file.

    Args:
        savedir (str): where to save the csvfile
        user_labels (list): list of lists, holding the boolean values for whether user thinks there's NOT a whistle
        fps (list): list of lists, holding the filepaths to original wav files
        start_times (list): list of lists, holding the start times of whistles, relative to original wav file
    """

    csvdict = {}
    columns = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
                'Low Freq (Hz)', 'High Freq (Hz)', 'Filepath', 'Found']

    for i,f in enumerate(fps):
        fp = f[0][:-4]

        with open(savedir + fp + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            # Iterate over the list of user labels and write that and all other info to file
            for j,label in enumerate(user_labels[i]):
                start_time, end_time = int(start_times[i][j]), int(start_times[i][j]) + 3

                # If label==True then the user said this does NOT contain a whistle
                if label:
                    continue
                # If label==False then the user agrees with the model that there IS a whistle
                else:
                    writer.writerow({'Selection': j, 'View': 'Spectrogram 1', 'Channel': '1', 'Begin Time (s)': start_time, 'End Time (s)': end_time,
                                    'Low Freq (Hz)': 0.0, 'High Freq (Hz)': sr / 2, 'Filepath': fp, 'Found': 'whistle'})



def main():

    ui_dir = 'outputs/ui/detection/spectrograms/'
    if not os.path.exists(ui_dir):
        os.makedirs(ui_dir)
    annots_dir = 'outputs/ui/detection/annotations/'
    if not os.path.exists(annots_dir):
        os.makedirs(annots_dir)

    config = {}
    st.header("Detect Whistles")
    uploaded_data = st.file_uploader("Choose audio files or a directory of audio files (each with minute/hour long durations).", accept_multiple_files=True)

    st.sidebar.title('Experiment Settings')
    st.sidebar.write("Model being used: mobilenetv2")
    model = 'mobilenetv2'

    confidence_threshold = st.sidebar.slider('How confident do you want to model to be?', min_value=0.0, max_value=1.0, value=0.5)

    uploaded_weights = st.sidebar.text_input("Enter path to model weights (otherwise default weights are used)", "default.h5")
    if uploaded_weights == "default.h5":
        weights = 'weights/detector_weights.h5'
    else:
        weights = uploaded_weights 

    global all_predictions
    global all_images
    global all_visuals
    global all_start_times
    global all_wav_fps
    upload_button = st.button("Detect Whistles")
    if upload_button:
        all_predictions = []
        all_images = []
        all_visuals = []
        all_start_times = []
        all_wav_fps = []

        # Run 1 file through the model at a time
        # Save the outputs from ALL files at once
        for data in uploaded_data:
            predictions, confidences, images, visuals = app_detect.run(data, model, confidence_threshold, weights) 
            
            # We ONLY want to visualize spectrogram windows where the model predicted 1 (whistle)
            # So we filter out the lists of images, filepaths, and predictions based on that 
            pos_predictions, pos_images, pos_visuals = [], [], []
            pos_start_times, pos_wav_fps = [], []
            for i,p in enumerate(predictions):
                if p == 1:
                    pos_predictions.append(p)
                    pos_images.append(images[i])
                    pos_visuals.append(visuals[i])
                    pos_start_times.append(i * 3)  # start times are in 3 second intervals
                    pos_wav_fps.append(data.name)

            # Only append info for this file if there is at least 1 chunk that the model thought contained a whistle
            if len(pos_predictions) > 0:
                all_predictions.append(pos_predictions)
                all_images.append(pos_images)
                all_visuals.append(pos_visuals)
                all_start_times.append(pos_start_times)
                all_wav_fps.append(pos_wav_fps)
            else:
                st.write("**There were no whistle instances that the model was sufficiently confident about in ", data.name, "**")

        st.success("Predictions are complete! Go to the Whistle Labeling section to label.")


    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    st.header("Detection Verification")
    st.warning('Do NOT click on the expander buttons, it will take you to another page and all progress will be lost. Use your trackpad to zoom in and out.')
    st.caption("Select the checkbox beneath any spectrogram window that ***DO NOT*** contain whistles.")

    detect_label_button = st.button('Start Verifying')
    if detect_label_button or ("labels" in st.session_state):    

        if "labels" not in st.session_state:
            st.session_state.files = all_images  # list of lists of file chunks
            st.session_state.predictions = all_predictions  # list of lists of predictions
            st.session_state.start_times = all_start_times  # list of lists of start times, relative to original audio file
    
            st.session_state.visuals = all_visuals  # list of lists of the filepaths to the spectrograms
            st.session_state.wav_fps = all_wav_fps  # list of lists of filepaths to the original wav

            st.session_state.len = len(all_images)  # this is the number of files
            if len(all_predictions) == 0:
                st.session_state.count = st.session_state.len
            else:
                st.session_state.current_predictions = all_predictions[0]  # list of predictions for the chunks of the 0th image file
                st.session_state.current_images = all_images[0]  # list of image chunks in the 0th image file
                st.session_state.current_start_times = all_start_times[0]  # list of start times in the 0th file

                st.session_state.count = 0  

            st.session_state.labels = [[] for i in range(st.session_state.len)]  # I'm thinking this will be a list of lists of labels for each chunk in each file

        def form_callback():
            # Iterate over the boolean user inputs from the previous form and save that info
            for i in range(len(st.session_state.current_images)):
                st.session_state.labels[st.session_state.count].append(st.session_state[i])

            st.session_state.count = st.session_state.count + 1
            if st.session_state.count < st.session_state.len:
                st.session_state.current_images = st.session_state.files[st.session_state.count]
                st.session_state.current_predictions = st.session_state.predictions[st.session_state.count]
                st.session_state.current_start_times = st.session_state.start_times[st.session_state.count]


        if st.session_state.count < st.session_state.len:
            st.write(
                "Annotated:",
                st.session_state.count,
                "– Remaining:",
                st.session_state.len - st.session_state.count
            )
        else:
            st.success(
                f"🎈 Done! All {st.session_state.len} images annotated."
            )        

            st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

            # The human-in-the-loop step is complete, write the output to csv files
            st.header("Annotations")
            st.write("To access these annotations, click on your dolphin_whistles folder.")
            st.write("These are being written to... **dolphin_whistles/" + annots_dir + "**")
            write_to_csv(annots_dir, st.session_state.labels, st.session_state.wav_fps, st.session_state.start_times, 60000)


        if st.session_state.count < st.session_state.len:

            form = st.form("checkboxes", clear_on_submit=True)
            with form:
                # Display images in rows of 4
                col1, col2, col3, col4 = st.columns(4)

                for i,img in enumerate(st.session_state.current_images):
                    if i % 4 == 0:
                        col1.image(img)
                        col1.checkbox("", key=i)
                    elif i % 4 == 1:
                        col2.image(img)
                        col2.checkbox("", key=i)
                    elif i % 4 == 2:
                        col3.image(img)
                        col3.checkbox("", key=i)
                    elif i % 4 == 3:
                        col4.image(img)
                        col4.checkbox("", key=i)

                # When the user presses this submit button, all checkbox info is submitted and the script is rerun
                submit_button = st.form_submit_button(label='Submit', on_click=form_callback)
    

if __name__ == '__main__':
    main()
