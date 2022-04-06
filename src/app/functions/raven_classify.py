import os
import sys
import csv
import random
import streamlit as st

# Internal packages
sys.path.append('src/')
import dolphin.app.app_raven_classify as app_raven_classify


def write_to_raven(orig_dfs: list, model_info: dict, savedir: str, fns: str):
    """
    Takes the model predictions, user boolean labels, and filepaths and writes to csv file.

    Args:
        orig_df (list): list of dictionaries of the original csv files
        model_info (dict): the saved model predictions and user label for the file
        savedir (str): directory where the new raven files will be saved
        fns (list): list of filenames of the original audio files
    """
    columns = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
                'Low Freq (Hz)', 'High Freq (Hz)', 'Filepath', 'Label',
                '1st Prediction, Confidence', '2nd Prediction, Confidence',
                '3rd Prediction, Confidence']

    count = 0
    for j,orig_df in enumerate(orig_dfs):
        fn = fns[count]

        with open(savedir + fn + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter='\t')
            writer.writeheader()

            for i,row in orig_df.iterrows():
                count += 1 # increment how many spectrogram chunks we've looked at

                row = row.to_dict()
                writer.writerow({'Selection': row['Selection'], 'View': row['View'], 'Channel': row['Channel'], 
                                'Begin Time (s)': row['Begin Time (s)'], 'End Time (s)': row['End Time (s)'],
                                'Low Freq (Hz)': row['Low Freq (Hz)'], 'High Freq (Hz)': row['High Freq (Hz)'],
                                'Filepath': fn, 'Label': model_info[i]['User Label'],
                                '1st Prediction, Confidence': model_info[i]['1st Prediction, Confidence'], 
                                '2nd Prediction, Confidence': model_info[i]['2nd Prediction, Confidence'],
                                '3rd Prediction, Confidence': model_info[i]['3rd Prediction, Confidence']
                                })
        


def main():

    ui_dir = 'outputs/ui/raven_classification/spectrograms/'
    if not os.path.exists(ui_dir):
        os.makedirs(ui_dir)
    annots_dir = 'outputs/ui/raven_classification/annotations/'
    if not os.path.exists(annots_dir):
        os.makedirs(annots_dir)

    config = {}

    st.header("Classify Whistles in Long Recordings")
    uploaded_data = st.file_uploader("Choose audio files and their respective csv file. \nEach audio file + csv file pair should share the same basename (ex. audio_file.wav + audio_file.csv). \nRequired columns of the csv file: Selection, View, Channel, Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz)", accept_multiple_files=True)

    upload_button = st.button("Classify")
    st.sidebar.title('Experiment Settings')

    st.sidebar.text("Model being used: mobilenetv2")
    model = 'mobilenetv2'

    uploaded_weights = st.sidebar.text_input("Enter path to model weights (otherwise default weights are used)", "default.h5")
    if uploaded_weights == "default.h5":
        weights = 'weights/classifier_weights.h5'
    else:
        weights = uploaded_weights 

    global basenames
    global indices
    global predictions
    global confidences
    global model_info
    global dfs
    if upload_button:
        predictions, confidences, images, basenames, indices, dfs = app_raven_classify.run(uploaded_data, model, weights=weights)
        
        st.success("""Predictions are complete! Go to Spectrogram Labeling section to annotate. """)

        # Format the model predicted labels and confidence scores for ultimately writing to csv
        model_info = []
        for i,img in enumerate(images):
            entry = {
                'Filename': basenames[i],
                '1st Prediction, Confidence': predictions[i][0] + ", " + str(confidences[i][0]),
                '2nd Prediction, Confidence': predictions[i][1] + ", " + str(confidences[i][1]),
                '3rd Prediction, Confidence': predictions[i][2] + ", " + str(confidences[i][2])
            }
            model_info.append(entry)
    
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)    

    st.header("Spectrogram Labeling")
    st.warning('Do NOT click on the expander buttons, it will take you to another page and all progress will be lost. Use your trackpad to zoom in and out.')
    st.caption("Please select which signature whistle matches best.")

    vis_button = st.button("Start Annotating")
    if vis_button or ("annotations" in st.session_state):

        if "annotations" not in st.session_state:
            st.session_state.annotations = {}
            st.session_state.files = basenames 
            st.session_state.indices = indices

            st.session_state.predictions = predictions
            st.session_state.confidences = confidences
            st.session_state.current_prediction = predictions[0]
            st.session_state.current_confidence = confidences[0]

            st.session_state.current_image = basenames[0] + str(st.session_state.indices[0]) + '.png'
            st.session_state.count = 0
            st.session_state.len = len(basenames)

        def annotate(label):
            st.session_state.annotations[st.session_state.current_image] = label
            model_info[st.session_state.count]['User Label'] = label

            st.session_state.count = st.session_state.count + 1
            if st.session_state.count < st.session_state.len:
                st.session_state.current_image = st.session_state.files[st.session_state.count] + str(st.session_state.indices[st.session_state.count]) + '.png'
                st.session_state.current_prediction = st.session_state.predictions[st.session_state.count]
                st.session_state.current_confidence = st.session_state.confidences[st.session_state.count]

        # Path to the current image we want classified and annotated
        image_path = ui_dir + st.session_state.current_image 
        
        # A countdown of how many images are left to be annotated
        if st.session_state.count < st.session_state.len:
            st.write(
                "Annotated:",
                len(st.session_state.annotations),
                "– Remaining:",
                st.session_state.len - st.session_state.count,
            )
        else:
            st.success(
                f"🎈 Done! All {len(st.session_state.annotations)} images annotated."
            )

        # Human-in-the-loop classification 
        if st.session_state.count < st.session_state.len:
            st.subheader("Current spectrogram...")
            col1, col2, col3 = st.columns([1,6,1])
            col2.image(image_path)

            st.subheader("The model's predicted matches...", st.session_state.current_prediction)
            col2, col3, col4 = st.columns(3)
            with col2:
                img0 = 'data/app/individual_examples/' + st.session_state.current_prediction[0] + '.png'
                st.image(img0)
                st.caption(st.session_state.current_prediction[0] + ", " + st.session_state.current_confidence[0])
                st.button("This is " + st.session_state.current_prediction[0], on_click=annotate, 
                    args=(st.session_state.current_prediction[0],))
                st.button("None of the Above", on_click=annotate, args=("NULL",))
            with col3:
                img1 = 'data/app/individual_examples/' + st.session_state.current_prediction[1] + '.png'
                st.image(img1)
                st.caption(st.session_state.current_prediction[1] + ", " + st.session_state.current_confidence[1])
                st.button("This is " + st.session_state.current_prediction[1], on_click=annotate, 
                    args=(st.session_state.current_prediction[1],))
            with col4:
                img2 = 'data/app/individual_examples/' + st.session_state.current_prediction[2] + '.png'
                st.image(img2)
                st.caption(st.session_state.current_prediction[2] + ", " + st.session_state.current_confidence[2])
                st.button("This is " + st.session_state.current_prediction[2], on_click=annotate, 
                    args=(st.session_state.current_prediction[2],))

        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        # Writing annotations to file
        st.header("Annotations")
        if st.session_state.count >= st.session_state.len:
            st.write(st.session_state.annotations)
            write_to_raven(dfs, model_info, annots_dir, basenames)
            st.write("To access these annotations, click on your dolphin_whistles folder.")
            st.write("These are being written to... **dolphin_whistles/" + annots_dir + "**")


if __name__ == "__main__":
    main()
