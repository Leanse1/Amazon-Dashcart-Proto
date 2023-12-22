from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import config

def _display_detected_frames(conf, model, st_count, st_frame, image):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    inText = 'Product Added'
    outText = 'Product Returned'
    total_cost_in = 0
    total_cost_out = 0

    # Calculate total cost for OBJECT_COUNTER
    if config.OBJECT_COUNTER != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER.items()):
            outText += f' - {key}: {value}'

            if config.OBJECT_PRICES != None and key in config.OBJECT_PRICES:
                price = config.OBJECT_PRICES[key]
                total_cost_out += value * price

    # Calculate total cost for OBJECT_COUNTER1
    if config.OBJECT_COUNTER1 != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER1.items()):
            inText += f' - {key}: {value}'

            if config.OBJECT_PRICES != None and key in config.OBJECT_PRICES:
                price = config.OBJECT_PRICES[key]
                total_cost_in += value * price
                
    st_count.write(
    f'{inText}\n\n{outText}\n\nTotal Cost In: {total_cost_in}\n\nTotal Cost Out: {total_cost_out}\n\nFinal Amount to be paid: {total_cost_in - total_cost_out}')            

    # st_count.markdown(
    # f'<div style="font-family: Arial, sans-serif; font-size: 16px; color: #333;">{inText}<br>{outText}<br><br>Total Cost Added: ₹ {total_cost_in}<br>Total Cost Returned: ₹ {total_cost_out}<br><br>Final amount to be paid: ₹ {total_cost_in - total_cost_out}</div>',
    # unsafe_allow_html=True)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


def infer_uploaded_video(conf, model):
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )
    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    config.OBJECT_COUNTER1 = None
                    config.OBJECT_COUNTER = None
                    config.OBJECT_PRICES = None
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_count = st.empty()
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_count,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_count,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
