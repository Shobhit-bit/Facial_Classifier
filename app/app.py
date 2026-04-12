import streamlit as st
from insightface.app import FaceAnalysis
import cv2 
import os
import core 
import shutil


@st.cache_resource
def load_models():
    model = FaceAnalysis(name="buffalo_l",providers = ["CUDAExecutionProvider","CPUExecutionProvider"])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model
model = load_models()

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

st.title("face sorting system")

st.header("Settings")
input_path = st.text_input("input_here",value = "")
output_path = st.text_input("Output_here",value = "")
st.subheader("Quatlity Filters")
blur_val = st.slider("Blur thres",0,200,85)
conf_val= st.slider("Face_detection_confidence",0.0,1.0,0.85)
ratio_val = st.slider("Min_Face_to_photo_ratio",0.0,0.2,0.06)

if st.button("STEP 1: SCAN DIRECTORY"):
    if not os.path.exists(input_path):
        st.error("Input path does not exist. Check your mount point.")
    else:
        with st.spinner("Analyzing images... this takes time."):
            data = core.extract_embeddings(model, input_path, blur_val, conf_val, ratio_val)
            st.session_state.processed_data = data
            st.success(f"Scan complete. Found {len(data)} valid faces.")

if st.session_state.processed_data:
    st.divider()
    num_cluster = st.number_input("Number of folders to create",min_value=1,value=7)
    if st.button("CLUSTER NOOOOWWWWW!!!!"):
        with st.spinner("CLussttteerrrr"):
            results = core.run_clustering(st.session_state.processed_data,num_cluster)
            count = 0
            for face in results:
                target_dir = os.path.join(output_path,f"Hito_wa_{face["id"]}")
                os.makedirs(target_dir,exist_ok = True)
                shutil.copy(face["path"],os.path.join(target_dir,face["name"]))
                count +=1
            st.success(f"Sorted {count} images into {num_cluster} folders.")