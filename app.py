import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Resolve model path relative to this file
MODEL_NAME = "student_pass_model3.pkl"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_NAME

# Use cache_resource when available, otherwise fall back to st.cache
_cache = getattr(st, "cache_resource", None)
if _cache is None:
    _cache = st.cache(allow_output_mutation=True)

@_cache
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        # Re-raise so caller can show a friendly message
        raise RuntimeError(f"Failed to load model from {path}: {e}")

# Try load model and show friendly error if not available
if not MODEL_PATH.exists():
    st.error(f"Model file '{MODEL_NAME}' not found at: {MODEL_PATH}\nPlease ensure it is in the same directory as app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(str(e))
    st.stop()

st.title("Student Pass/Fail Prediction System")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=168.0, value=10.0, step=0.5)
    attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=75.0, step=0.5)

with col2:
    marks = st.number_input("Current Marks", min_value=0.0, max_value=100.0, value=50.0, step=0.5)

# Predict button (avoid using type parameter for compatibility)
if st.button("Predict Result"):
    # Ensure features are in the exact order the model was trained on
    input_data = np.array([[study_hours, attendance, marks]], dtype=float)
    try:
        if not hasattr(model, "predict"):
            st.error("Loaded model does not implement a 'predict' method.")
        else:
            raw_pred = model.predict(input_data)
            if len(raw_pred) == 0:
                st.error("Model returned an empty prediction.")
            else:
                pred0 = raw_pred[0]

                # Normalize different possible prediction types:
                # - numeric labels (0/1)
                # - probabilities (predict_proba might be used by some models)
                # - strings like "pass"/"fail"
                label = None

                # If model returned probabilities (shape Nx2 or Nx1), try to handle that
                try:
                    arr = np.array(pred0)
                except Exception:
                    arr = None

                if isinstance(pred0, (int, np.integer)):
                    label = int(pred0)
                elif isinstance(pred0, (float, np.floating)):
                    # If a float label, round / threshold at 0.5
                    label = int(round(pred0)) if pred0 in (0.0, 1.0) else (1 if pred0 >= 0.5 else 0)
                elif arr is not None and getattr(arr, 'ndim', 0) > 0 and arr.size > 1:
                    # e.g., predict_proba output for a single sample like [0.3, 0.7]
                    # take the class with highest probability
                    try:
                        probs = np.asarray(arr, dtype=float).ravel()
                        label = int(np.argmax(probs))
                    except Exception:
                        label = None
                else:
                    # Fallback for strings/bytes or unknown types
                    s = str(pred0).strip().lower()
                    if s in ("1", "true", "pass", "passed", "yes", "y"):
                        label = 1
                    elif s in ("0", "false", "fail", "failed", "no", "n"):
                        label = 0
                    else:
                        # try converting to float then to int
                        try:
                            f = float(s)
                            label = int(round(f)) if f in (0.0, 1.0) else (1 if f >= 0.5 else 0)
                        except Exception:
                            label = None

                if label is None:
                    st.error(f"Unable to interpret model output: {pred0}")
                else:
                    st.markdown("---")
                    if label == 1:
                        st.success("### Prediction: **PASS**")
                        st.balloons()
                    else:
                        st.error("### Prediction: **FAIL**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
