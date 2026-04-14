import streamlit as st
from insightface.app import FaceAnalysis
import os
import core 
import shutil
import zipfile
import tempfile


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
uploaded_zip = st.file_uploader("upload zip here",type = ["zip"])
# input_path = st.text_input("input_here",value = "")
# tmp_dir = "/mnt/113117fa-e8af-4d8b-9b69-072b1b91c742/Code/facialRegonitionModel/tmp"
output_path = st.text_input("Output_here",value = "")
st.subheader("Quatlity Filters")
blur_val = st.slider("Blur thres",0,200,85)
conf_val= st.slider("Face_detection_confidence",0.0,1.0,0.85)
ratio_val = st.slider("Min_Face_to_photo_ratio",0.0,0.2,0.06)

if st.button("STEP 1: Scan Zip"):
    # if not os.path.exists(input_path):
    #     st.error("Input path does not exist. Check your mount point.")
    if not uploaded_zip:
        st.error("Upload a zip first")
    elif not output_path:
        st.error("plz specify out_directory")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with st.spinner("Unzipping"):
                with zipfile.ZipFile(uploaded_zip,"r") as zip_ref:
                    zip_ref.extractall(tmp_dir)
            data = core.extract_embeddings(model, tmp_dir, blur_val, conf_val, ratio_val)
            embed_dir = os.path.join(output_path,"embeds")
            os.makedirs(embed_dir,exist_ok=True)
            for item in data:
                new_path = os.path.join(embed_dir,item["name"])
                shutil.copy(item["path"],new_path)
                item["path"] = new_path
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
                target_dir = os.path.join(output_path,f"Hito_wa_{face['id']}")
                os.makedirs(target_dir,exist_ok = True)
                shutil.copy(face["path"],os.path.join(target_dir,face["name"]))
                count +=1
            st.success(f"Sorted {count} images into {num_cluster} folders.")
            os.rmdir(embed_dir)

#python3 -m streamlit run app.py --server.port 6969