import tempfile
from pathlib import Path

import streamlit as st

from app.inference import MADNetPredictor

st.set_page_config(
    page_title="M-ADNet | Research Prototype",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    return MADNetPredictor()


st.title("M-ADNet")
st.subheader("Multimodal Alzheimer's Disease Classification Research Prototype")

st.warning(
    """
    **Research use only — not a medical diagnostic tool.**

    This application is an experimental research prototype. Its predictions
    must not be used to diagnose, treat, or make clinical decisions about
    any individual. The model has not been established as clinically valid.
    """
)

st.markdown(
    """
    Upload an original 3D structural MRI in NIfTI format and enter the
    corresponding clinical information. The application will preprocess
    the MRI and generate model predictions for three classes:
    **CN, MCI, and Dementia**.
    """
)

st.info(
    "Upload an original MRI, not a volume that has already undergone "
    "the M-ADNet preprocessing pipeline. Do not upload identifiable "
    "patient information to a public deployment."
)

with st.sidebar:
    st.header("About M-ADNet")
    st.write("Multimodal MRI and clinical-feature classification.")
    st.write("Model classes:")
    st.write("- CN")
    st.write("- MCI")
    st.write("- Dementia")

    st.divider()
    st.caption("Experimental research software")

st.header("1. MRI upload")

uploaded_file = st.file_uploader(
    "Choose an original 3D structural MRI",
    type=["nii", "gz"],
    help="Supported formats: .nii and .nii.gz",
)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        st.error("Please upload a .nii or .nii.gz file.")
        st.stop()

    st.success(f"Uploaded: {uploaded_file.name}")

st.header("2. Clinical information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=120,
        value=72,
        step=1,
    )

with col2:
    education = st.number_input(
        "Education (years)",
        min_value=0,
        max_value=40,
        value=16,
        step=1,
    )

with col3:
    sex = st.selectbox(
        "Sex",
        options=["Male", "Female"],
    )

st.caption(
    "Clinical features are standardized using the training-set "
    "statistics stored in the model checkpoint."
)

st.divider()

st.header("3. Generate prediction")

if st.button(
    "Run M-ADNet prediction",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
):
    try:
        predictor = load_predictor()

        with st.spinner(
            "Preprocessing MRI and generating prediction. "
            "This may take several minutes on CPU."
        ):
            suffix = (
                ".nii.gz"
                if uploaded_file.name.lower().endswith(".nii.gz")
                else ".nii"
            )

            with tempfile.TemporaryDirectory(
                prefix="madnet_upload_"
            ) as temp_dir:

                input_path = Path(temp_dir) / f"uploaded_mri{suffix}"

                input_path.write_bytes(uploaded_file.getvalue())

                result = predictor.predict(
                    nifti_path=input_path,
                    age=age,
                    education=education,
                    sex=sex,
                )

        st.success("Prediction completed.")

        st.header("4. Prediction results")

        prediction = result["prediction"]
        probabilities = result["probabilities"]

        st.subheader("Predicted class")

        st.markdown(f"## {prediction}")

        st.subheader("Class probabilities")

        for class_name, probability in probabilities.items():
            st.write(
                f"**{class_name}:** {probability * 100:.2f}%"
            )

            st.progress(probability)

        st.divider()

        st.subheader("Model information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.metric(
                "Checkpoint epoch",
                str(result["checkpoint_epoch"]),
            )

        with info_col2:
            score = result["validation_macro_f1"]

            st.metric(
                "Validation macro-F1",
                f"{score:.4f}" if score is not None else "Unavailable",
            )

        st.warning(
            "These probabilities are model outputs, not calibrated "
            "clinical risk estimates. Prediction performance is limited, "
            "particularly for the MCI and Dementia classes. "
            "Do not interpret this result as a diagnosis."
        )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.exception(error)


st.divider()

st.caption(
    "M-ADNet research prototype | Experimental machine-learning software. "
    "Not intended for clinical diagnosis or treatment."
)
