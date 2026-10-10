import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Cancer Classifier", page_icon="🏥", layout="wide")

# Initialize session state for slider values
if 'slider_values' not in st.session_state:
    st.session_state.slider_values = {
        'radius_mean': 15.0,
        'texture_mean': 20.0,
        'perimeter_mean': 100.0,
        'area_mean': 500.0,
        'smoothness_mean': 0.1,
        'compactness_mean': 0.1,
        'concavity_mean': 0.1,
        'concave_points_mean': 0.05,
        'symmetry_mean': 0.2,
        'fractal_dimension_mean': 0.06,
        'radius_se': 0.5,
        'texture_se': 1.0,
        'perimeter_se': 2.0,
        'area_se': 25.0,
        'smoothness_se': 0.01,
        'compactness_se': 0.02,
        'concavity_se': 0.02,
        'concave_points_se': 0.01,
        'symmetry_se': 0.02,
        'fractal_dimension_se': 0.003,
        'radius_worst': 20.0,
        'texture_worst': 25.0,
        'perimeter_worst': 130.0,
        'area_worst': 1000.0,
        'smoothness_worst': 0.13,
        'compactness_worst': 0.25,
        'concavity_worst': 0.3,
        'concave_points_worst': 0.1,
        'symmetry_worst': 0.3,
        'fractal_dimension_worst': 0.1
    }

# Test data values
test_data = [
    14.54, 27.54, 96.73, 658.8, 0.1139, 0.1595, 0.1639, 0.07364, 0.2303, 0.07077,
    0.37, 1.033, 2.879, 32.55, 0.005607, 0.0424, 0.04741, 0.0109, 0.01857, 0.005466,
    17.46, 37.13, 124.1, 943.2, 0.1678, 0.6577, 0.7026, 0.1712, 0.4218, 0.1341
]

# Simple number input function
def simple_number_input(label, min_val, max_val, default_val, key, step=1.0):
    current_value = st.session_state.slider_values[key]
    
    value = st.number_input(
        label, 
        min_value=min_val, 
        max_value=max_val, 
        value=current_value, 
        step=step, 
        key=f"{key}_input",
        format="%.4f" if step < 0.01 else "%.1f"
    )
    st.session_state.slider_values[key] = value
    
    return value

# Load test data function
def load_test_data():
    test_keys = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    for i, key in enumerate(test_keys):
        st.session_state.slider_values[key] = test_data[i]

# Parse comma-separated data function
def parse_comma_data(data_string):
    try:
        # Split by comma and convert to float
        values = [float(x.strip()) for x in data_string.split(',')]
        
        if len(values) != 30:
            return False, f"Expected 30 values, got {len(values)}"
        
        # Load the parsed values into session state
        test_keys = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        for i, key in enumerate(test_keys):
            st.session_state.slider_values[key] = values[i]
        
        return True, "Data loaded successfully!"
    except ValueError as e:
        return False, f"Error parsing data: {str(e)}"

# Title
st.title("🏥 Breast Cancer Classifier")
st.markdown("Enter tumor measurements to classify as Benign or Malignant")

# Load model
@st.cache_resource
def load_model():
    model_path = 'breast_cancer_model.pkl'
    with open('breast_cancer_model.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_model()
model = model_data[0]  # Get the model from the tuple
scaler = model_data[1]  # Get the scaler from the tuple

# Sidebar info
with st.sidebar:
    st.header("📋 Model Info")
    st.info("""
    **Algorithm:** Logistic Regression
    - Training Accuracy: 95.7%
    - Test Accuracy: 91.4%
    """)
    
    st.markdown("---")
    
    # Quick data input section
    st.markdown("### 📝 Quick Data Input")
    st.markdown("Paste 30 comma-separated values:")
    
    # Text area for comma-separated data
    comma_data = st.text_area(
        "Data (30 values separated by commas)",
        placeholder="14.54,27.54,96.73,658.8,0.1139,0.1595,...",
        height=100,
        key="comma_input"
    )
    
    # Buttons in the quick data input section
    col1, = st.columns(1)
    
    with col1:
        if st.button("📥 Load Data", use_container_width=True) and comma_data:
            success, message = parse_comma_data(comma_data)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    

# Single Prediction Section
st.subheader("Enter Tumor Measurements")

# Create 3 columns for input
col1, col2, col3 = st.columns(3)

input_values = []

# Mean values
with col1:
    st.write("**Mean Values**")
    input_values.append(simple_number_input("Radius Mean", 5.0, 35.0, 15.0, 'radius_mean', 0.1))
    input_values.append(simple_number_input("Texture Mean", 5.0, 40.0, 20.0, 'texture_mean', 0.1))
    input_values.append(simple_number_input("Perimeter Mean", 20.0, 200.0, 100.0, 'perimeter_mean', 1.0))
    input_values.append(simple_number_input("Area Mean", 100.0, 2500.0, 500.0, 'area_mean', 10.0))
    input_values.append(simple_number_input("Smoothness Mean", 0.05, 0.15, 0.1, 'smoothness_mean', 0.001))
    input_values.append(simple_number_input("Compactness Mean", 0.01, 0.3, 0.1, 'compactness_mean', 0.001))
    input_values.append(simple_number_input("Concavity Mean", 0.0, 0.5, 0.1, 'concavity_mean', 0.001))
    input_values.append(simple_number_input("Concave Points Mean", 0.0, 0.3, 0.05, 'concave_points_mean', 0.001))
    input_values.append(simple_number_input("Symmetry Mean", 0.1, 0.4, 0.2, 'symmetry_mean', 0.001))
    input_values.append(simple_number_input("Fractal Dimension Mean", 0.04, 0.1, 0.06, 'fractal_dimension_mean', 0.001))

# SE values
with col2:
    st.write("**Standard Error (SE)**")
    input_values.append(simple_number_input("Radius SE", 0.1, 3.0, 0.5, 'radius_se', 0.01))
    input_values.append(simple_number_input("Texture SE", 0.1, 5.0, 1.0, 'texture_se', 0.01))
    input_values.append(simple_number_input("Perimeter SE", 0.1, 25.0, 2.0, 'perimeter_se', 0.1))
    input_values.append(simple_number_input("Area SE", 0.5, 500.0, 25.0, 'area_se', 0.5))
    input_values.append(simple_number_input("Smoothness SE", 0.001, 0.05, 0.01, 'smoothness_se', 0.0001))
    input_values.append(simple_number_input("Compactness SE", 0.001, 0.1, 0.02, 'compactness_se', 0.0001))
    input_values.append(simple_number_input("Concavity SE", 0.0, 0.3, 0.02, 'concavity_se', 0.0001))
    input_values.append(simple_number_input("Concave Points SE", 0.0, 0.1, 0.01, 'concave_points_se', 0.0001))
    input_values.append(simple_number_input("Symmetry SE", 0.01, 0.1, 0.02, 'symmetry_se', 0.0001))
    input_values.append(simple_number_input("Fractal Dimension SE", 0.001, 0.03, 0.003, 'fractal_dimension_se', 0.0001))

# Worst values
with col3:
    st.write("**Worst Values**")
    input_values.append(simple_number_input("Radius Worst", 7.0, 40.0, 20.0, 'radius_worst', 0.1))
    input_values.append(simple_number_input("Texture Worst", 7.0, 50.0, 25.0, 'texture_worst', 0.1))
    input_values.append(simple_number_input("Perimeter Worst", 30.0, 250.0, 130.0, 'perimeter_worst', 1.0))
    input_values.append(simple_number_input("Area Worst", 200.0, 4500.0, 1000.0, 'area_worst', 10.0))
    input_values.append(simple_number_input("Smoothness Worst", 0.08, 0.22, 0.13, 'smoothness_worst', 0.001))
    input_values.append(simple_number_input("Compactness Worst", 0.04, 1.1, 0.25, 'compactness_worst', 0.001))
    input_values.append(simple_number_input("Concavity Worst", 0.0, 1.3, 0.3, 'concavity_worst', 0.001))
    input_values.append(simple_number_input("Concave Points Worst", 0.0, 0.7, 0.1, 'concave_points_worst', 0.001))
    input_values.append(simple_number_input("Symmetry Worst", 0.1, 0.7, 0.3, 'symmetry_worst', 0.001))
    input_values.append(simple_number_input("Fractal Dimension Worst", 0.05, 0.25, 0.1, 'fractal_dimension_worst', 0.001))

# Predict button
if st.button("🔍 Predict", use_container_width=True):
    # Convert to array and scale the input data
    X = np.array([input_values])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    st.markdown("---")
    
    if prediction == 0:
        st.error(f"🚨 **MALIGNANT (Cancerous)**")
    else:
        st.success(f"✅ **BENIGN (Non-cancerous)**")

st.markdown("---")
st.markdown("⚠️ **Disclaimer:** For educational purposes only. Consult healthcare professionals for medical decisions.")

