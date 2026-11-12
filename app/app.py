import tempfile
from pathlib import Path

import streamlit as st

from inference import MADNetPredictor
from explainability import (
    load_shap_background,
    run_gradcam,
    run_clinical_shap,
    shap_figure,
)

st.set_page_config(
    page_title="M-ADNet | Research Prototype",
    page_icon="🧠",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAP_BACKGROUND_PATH = PROJECT_ROOT / "app" / "assets" / "shap_background.npy"


@st.cache_resource
def load_predictor():
    return MADNetPredictor()


st.title("M-ADNet")
st.subheader("Multimodal Alzheimer's Disease Classification Research Prototype")

st.warning(
    "**Research use only — not a medical diagnostic tool.** This experimental "
    "prototype has not been established as clinically valid. Do not use its "
    "outputs to diagnose, treat, or make clinical decisions."
)

st.markdown(
    "Upload an original 3D structural MRI in NIfTI format and enter the "
    "corresponding clinical information. The model returns CN, MCI, or Dementia."
)

st.info(
    "Upload an original MRI, not one already processed by the M-ADNet pipeline. "
    "Do not upload identifiable patient information to a public deployment."
)

with st.sidebar:
    st.header("About M-ADNet")
    st.write("Multimodal MRI and clinical-feature classification.")
    st.write("Classes: CN, MCI, Dementia")
    st.divider()
    st.caption("Experimental research software")

st.header("1. MRI upload")
uploaded_file = st.file_uploader(
    "Choose an original 3D structural MRI",
    type=["nii", "gz"],
    help="Supported formats: .nii and .nii.gz",
)

if uploaded_file is not None:
    lower_name = uploaded_file.name.lower()
    if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
        st.error("Please upload a .nii or .nii.gz file.")
        st.stop()
    st.success(f"Uploaded: {uploaded_file.name}")

st.header("2. Clinical information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=72)
with col2:
    education = st.number_input(
        "Education (years)", min_value=0, max_value=40, value=16
    )
with col3:
    sex = st.selectbox("Sex", options=["Male", "Female"])

st.caption(
    "Age and education are standardized using training-set statistics stored "
    "in the checkpoint. Sex encoding follows the research pipeline."
)

st.divider()
st.header("3. Generate prediction and explanations")

if st.button(
    "Run M-ADNet",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
):
    st.session_state.pop("madnet_result", None)
    st.session_state.pop("madnet_gradcam_fig", None)
    st.session_state.pop("madnet_shap_result", None)
    st.session_state.pop("madnet_shap_fig", None)
    st.session_state.pop("madnet_explanation_errors", None)

    try:
        predictor = load_predictor()
        suffix = (
            ".nii.gz"
            if uploaded_file.name.lower().endswith(".nii.gz")
            else ".nii"
        )

        with tempfile.TemporaryDirectory(prefix="madnet_upload_") as temp_dir:
            input_path = Path(temp_dir) / f"uploaded_mri{suffix}"
            input_path.write_bytes(uploaded_file.getvalue())

            with st.spinner(
                "Preprocessing MRI and preparing model inputs. "
                "This may take several minutes on CPU."
            ):
                mri_tensor, clinical_tensor = predictor.prepare_inputs(
                    input_path, age, education, sex
                )

        with st.spinner("Generating model prediction..."):
            result = predictor.predict_prepared(mri_tensor, clinical_tensor)

        st.session_state["madnet_result"] = result
        errors = {}

        with st.spinner("Generating 3D Grad-CAM explanation..."):
            try:
                fig, _ = run_gradcam(
                    predictor.model,
                    mri_tensor,
                    clinical_tensor,
                    result["predicted_index"],
                )
                st.session_state["madnet_gradcam_fig"] = fig
            except Exception as exc:
                errors["Grad-CAM"] = str(exc)

        with st.spinner("Generating clinical-branch SHAP explanation..."):
            try:
                background = load_shap_background(SHAP_BACKGROUND_PATH)
                shap_result = run_clinical_shap(
                    predictor.model, clinical_tensor, background
                )
                st.session_state["madnet_shap_result"] = shap_result
                st.session_state["madnet_shap_fig"] = shap_figure(shap_result)
            except Exception as exc:
                errors["SHAP"] = str(exc)

        st.session_state["madnet_explanation_errors"] = errors
        st.success("Prediction completed.")

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.exception(error)

result = st.session_state.get("madnet_result")

if result is not None:
    st.header("4. Prediction results")
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Predicted class")
        st.markdown(f"## {result['prediction']}")
    with right:
        st.subheader("Class probabilities")
        for class_name, probability in result["probabilities"].items():
            st.write(f"**{class_name}:** {probability * 100:.2f}%")
            st.progress(probability)

    info1, info2 = st.columns(2)
    with info1:
        st.metric("Checkpoint epoch", str(result["checkpoint_epoch"]))
    with info2:
        score = result["validation_macro_f1"]
        st.metric(
            "Validation macro-F1",
            f"{score:.4f}" if score is not None else "Unavailable",
        )

    st.warning(
        "These are model output probabilities, not calibrated clinical risk "
        "estimates. The current evaluation showed limited minority-class "
        "performance, particularly for MCI and Dementia. This is not a diagnosis."
    )

    st.header("5. Explainability")
    gradcam_tab, shap_tab = st.tabs(["MRI explanation — Grad-CAM", "Clinical explanation — SHAP"])

    with gradcam_tab:
        st.subheader("3D Grad-CAM overlays")
        fig = st.session_state.get("madnet_gradcam_fig")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info(
                "Grad-CAM was not generated. See the explanation error below."
            )
        st.caption(
            "The highlighted areas indicate regions influencing the selected "
            "class score in the MRI branch. Slices are central axial, coronal, "
            "and sagittal views in resized model-input voxel coordinates; they "
            "are not registered overlays in native scanner space. This is not "
            "evidence of disease-specific anatomy or causation."
        )

    with shap_tab:
        st.subheader("Clinical-branch SHAP contributions")
        fig = st.session_state.get("madnet_shap_fig")
        shap_result = st.session_state.get("madnet_shap_result")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.dataframe(
                {
                    "Feature": shap_result["feature_names"],
                    "Standardized input": shap_result["input_values"],
                    "CN SHAP": shap_result["values"][:, 0],
                    "MCI SHAP": shap_result["values"][:, 1],
                    "Dementia SHAP": shap_result["values"][:, 2],
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "SHAP was not generated. Ensure the training background file "
                "has been exported; see the explanation error below."
            )
        st.caption(
            "These SHAP values explain the clinical correction logits only. "
            "They do not explain the complete MRI-plus-clinical fused logits, "
            "and they do not represent causal effects. Positive and negative "
            "values indicate direction relative to the SHAP background."
        )

    errors = st.session_state.get("madnet_explanation_errors", {})
    if errors:
        st.warning("One or more explanations could not be generated.")
        for method, message in errors.items():
            with st.expander(f"{method} error"):
                st.code(message)

st.divider()
st.caption(
    "M-ADNet research prototype | Experimental machine-learning software. "
    "Not intended for clinical diagnosis or treatment."
)
