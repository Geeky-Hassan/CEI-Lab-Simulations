import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import datetime
import traceback

# Page configuration
st.set_page_config(
    page_title="Ohm's Law Interactive Laboratory - UMT Physics Dept",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.theory-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1e3c72;
    margin: 1rem 0;
}
.safety-warning {
    background-color: #fff3cd;
    border: 1px solid #ffecb5;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
.experiment-section {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"⚠️ Error in {func.__name__}: {str(e)}")
            if st.checkbox("Show detailed error (for debugging)"):
                st.code(traceback.format_exc())
            return None
    return wrapper

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Ohm's Law Interactive Laboratory Simulator</h1>
    <h3>Proof-of-Concept for UMT Physics Department Simulations</h3>
    <p>Developed by: Noor Ul Hassan (Team Lead CEI) | University of Management and Technology</p>
    <p>A comprehensive virtual laboratory replacing physical experiments with interactive simulations</p>
</div>
""", unsafe_allow_html=True)

# Utility Functions
@handle_errors
def calculate_current(voltage, resistance):
    """Calculate current using Ohm's Law: I = V/R"""
    if resistance <= 0:
        raise ValueError("Resistance must be positive")
    return voltage / resistance

@handle_errors
def calculate_power(voltage, current):
    """Calculate power: P = V * I"""
    return voltage * current

@handle_errors
def calculate_resistance_temp(R0, alpha, T, T0=293):
    """Calculate temperature-dependent resistance: R(T) = R0[1 + α(T - T0)]"""
    return R0 * (1 + alpha * (T - T0))

@handle_errors
def equivalent_resistance(resistors, connection_type):
    """Calculate equivalent resistance for series or parallel connection"""
    resistors = np.array(resistors)
    if connection_type == 'Series':
        return np.sum(resistors)
    elif connection_type == 'Parallel':
        return 1 / np.sum(1 / resistors)
    else:
        raise ValueError("Connection type must be 'Series' or 'Parallel'")

@handle_errors
def safety_check(voltage, current, power):
    """Check safety conditions and return warnings"""
    warnings = []
    if voltage > 50:
        warnings.append(f"⚠️ HIGH VOLTAGE WARNING: {voltage:.1f}V > 50V")
    if current > 10:
        warnings.append(f"⚠️ HIGH CURRENT WARNING: {current:.2f}A > 10A")
    if power > 50:
        warnings.append(f"⚠️ HIGH POWER WARNING: {power:.1f}W > 50W")
    return warnings

@handle_errors
def generate_experimental_error(value, error_percent):
    """Add realistic experimental error to measurements"""
    error = np.random.normal(0, error_percent / 100)
    return value * (1 + error)

# Initialize session state
if 'experiment_data' not in st.session_state:
    st.session_state.experiment_data = []
if 'completed_sections' not in st.session_state:
    st.session_state.completed_sections = set()
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'dark_mode': False,
        'auto_save': True,
        'precision': 4
    }

# Sidebar Navigation
st.sidebar.markdown("## 📋 Laboratory Navigation")

# User preferences
with st.sidebar.expander("⚙️ User Preferences"):
    st.session_state.user_preferences['precision'] = st.slider("Display Precision", 2, 6, 4)
    st.session_state.user_preferences['auto_save'] = st.checkbox("Auto-save data", True)
    if st.button("🔄 Reset All Data"):
        st.session_state.experiment_data = []
        st.session_state.completed_sections = set()
        st.success("Data reset successfully!")

lab_sections = [
    "📚 Pre-Laboratory Preparation",
    "🎯 Learning Objectives & Theory", 
    "🔬 Equipment & Setup",
    "📊 Part A: Ohmic Verification",
    "💡 Part B: Non-Ohmic Analysis", 
    "📈 Part C: Statistical Analysis",
    "🔍 Part D: Advanced Testing",
    "📊 Part E: Graph Analysis",
    "🔌 Part F: Series & Parallel Circuits",
    "🌡️ Part G: Temperature Effects",
    "⚠️ Safety Protocols",
    "📝 Report Generation & Export"
]

current_section = st.sidebar.radio("Select Laboratory Section:", lab_sections)

# Progress tracking
progress = len(st.session_state.completed_sections) / len(lab_sections)
st.sidebar.progress(progress)
st.sidebar.markdown(f"**Progress: {progress*100:.0f}% Complete**")
st.sidebar.markdown(f"**Completed: {len(st.session_state.completed_sections)}/{len(lab_sections)} sections**")

# Emergency Stop Button
if st.sidebar.button("🛑 EMERGENCY STOP", type="primary"):
    st.sidebar.error("⚠️ EMERGENCY STOP ACTIVATED")
    st.sidebar.markdown("All experiments halted for safety.")
    st.stop()

# Quick statistics
if st.session_state.experiment_data:
    st.sidebar.markdown("## 📊 Quick Stats")
    total_measurements = sum(len(exp['data']) for exp in st.session_state.experiment_data)
    st.sidebar.metric("Total Measurements", total_measurements)
    st.sidebar.metric("Experiments Done", len(st.session_state.experiment_data))

# === PRE-LABORATORY PREPARATION ===
if current_section == "📚 Pre-Laboratory Preparation":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("📚 Pre-Laboratory Preparation")
    
    st.markdown("""
    <div class="theory-box">
    <h4>🎯 Before Starting the Experiment</h4>
    <p>Complete all pre-laboratory requirements to ensure successful learning outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pre-lab checklist with interactive elements
    st.subheader("✅ Pre-Laboratory Checklist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📖 Reading Requirements:**")
        read_theory = st.checkbox("Read Ohm's Law theoretical background")
        read_safety = st.checkbox("Review safety protocols and procedures")
        read_equipment = st.checkbox("Study virtual equipment specifications")
        read_procedures = st.checkbox("Understand experimental procedures")
        
        if read_theory:
            st.success("✅ Theory reading completed")
    
    with col2:
        st.markdown("**🧮 Pre-Lab Calculations:**")
        calc_basic = st.checkbox("Complete basic Ohm's Law calculations")
        calc_power = st.checkbox("Practice power calculation formulas")
        calc_errors = st.checkbox("Understand uncertainty analysis")
        calc_temp = st.checkbox("Study temperature coefficient effects")
        
        if calc_basic:
            st.success("✅ Basic calculations completed")
    
    # Pre-lab Questions
    st.subheader("❓ Pre-Laboratory Questions")
    
    questions_answered = 0
    
    with st.expander("Question 1: State Ohm's Law and explain its limitations"):
        q1_answer = st.text_area("Your Answer:", key="q1", height=100)
        if len(q1_answer) > 50:
            questions_answered += 1
            st.success("✅ Question 1 answered")
        
    with st.expander("Question 2: Why do tungsten bulbs show non-ohmic behavior?"):
        q2_answer = st.text_area("Your Answer:", key="q2", height=100)
        if len(q2_answer) > 50:
            questions_answered += 1
            st.success("✅ Question 2 answered")
    
    # Verification
    all_reading = all([read_theory, read_safety, read_equipment, read_procedures])
    all_calc = all([calc_basic, calc_power, calc_errors, calc_temp])
    all_questions = questions_answered >= 2
    
    if all_reading and all_calc and all_questions:
        st.balloons()
        st.success("🎉 Pre-laboratory preparation complete!")
        st.session_state.completed_sections.add(current_section)
    else:
        st.info("⏳ Please complete all pre-laboratory requirements.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === LEARNING OBJECTIVES & THEORY ===
elif current_section == "🎯 Learning Objectives & Theory":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("🎯 Learning Objectives & Theoretical Background")
    
    # Learning objectives
    st.subheader("📋 Learning Objectives")
    
    col1, col2 = st.columns(2)
    with col1:
        obj1 = st.checkbox("✓ Verify Ohm's Law experimentally")
        obj2 = st.checkbox("✓ Analyze linear V-I relationships")
        obj3 = st.checkbox("✓ Investigate non-ohmic behavior")
        obj4 = st.checkbox("✓ Apply statistical analysis")
    
    with col2:
        obj5 = st.checkbox("✓ Interpret V-I characteristic curves")
        obj6 = st.checkbox("✓ Understand Ohm's Law limitations")
        obj7 = st.checkbox("✓ Develop data analysis skills")
        obj8 = st.checkbox("✓ Practice scientific reporting")
    
    objectives_completed = sum([obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8])
    st.progress(objectives_completed / 8)
    
    # Theory tabs
    tab1, tab2, tab3 = st.tabs(["🔋 Basic Theory", "📐 Mathematical Relations", "🧪 Material Properties"])
    
    with tab1:
        st.markdown("""
        <div class="theory-box">
        <h4>⚡ Ohm's Law Fundamentals</h4>
        <p>Georg Simon Ohm discovered in 1827 that the current through a conductor is directly 
        proportional to the voltage across it: <strong>V = IR</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive calculator
        col1, col2 = st.columns(2)
        with col1:
            calc_mode = st.selectbox("Find:", ["Current (I)", "Voltage (V)", "Resistance (R)"])
            
        with col2:
            if calc_mode == "Current (I)":
                v_val = st.number_input("V (volts):", value=12.0)
                r_val = st.number_input("R (ohms):", value=100.0)
                if r_val > 0:
                    i_val = v_val / r_val
                    st.metric("Current", f"{i_val:.4f} A")
    
    with tab2:
        st.markdown("**Power Relationships:**")
        
        # Interactive power graph
        voltage_range = st.slider("Voltage Range (V):", 0, 20, (0, 12))
        resistance_demo = st.number_input("Demo Resistance (Ω):", value=100.0)
        
        voltages = np.linspace(voltage_range[0], voltage_range[1], 50)
        currents = voltages / resistance_demo
        powers = voltages * currents
        
        fig_power = px.line(x=voltages, y=powers, title="Power vs Voltage",
                           labels={'x': 'Voltage (V)', 'y': 'Power (W)'})
        st.plotly_chart(fig_power, use_container_width=True)
    
    with tab3:
        st.markdown("**Material Properties Comparison:**")
        
        materials = {
            'Copper': {'resistivity': 1.68e-8, 'alpha': 0.0039},
            'Tungsten': {'resistivity': 5.60e-8, 'alpha': 0.0045},
            'Silicon': {'resistivity': 2300, 'alpha': -0.075}
        }
        
        material_df = pd.DataFrame(materials).T
        st.dataframe(material_df, use_container_width=True)
    
    if objectives_completed >= 4:
        if st.button("✅ Complete Theory Section"):
            st.session_state.completed_sections.add(current_section)
            st.success("Theory section completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === EQUIPMENT & SETUP ===
elif current_section == "🔬 Equipment & Setup":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("🔬 Virtual Laboratory Equipment & Setup")
    
    st.subheader("🎛️ Equipment Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Power Supply:**")
        ps_voltage = st.number_input("Voltage (V):", 0.0, 50.0, 12.0)
        ps_enabled = st.checkbox("Power ON")
        if ps_enabled:
            st.success(f"✅ Output: {ps_voltage}V")
    
    with col2:
        st.markdown("**Voltmeter:**")
        if ps_enabled:
            vm_reading = ps_voltage + np.random.normal(0, 0.01)
            st.metric("Reading", f"{vm_reading:.3f} V")
        else:
            st.metric("Reading", "0.000 V")
    
    with col3:
        st.markdown("**Ammeter:**")
        if ps_enabled:
            am_reading = ps_voltage / 100 + np.random.normal(0, 0.001)
            st.metric("Reading", f"{am_reading:.4f} A")
        else:
            st.metric("Reading", "0.0000 A")
    
    # Component selection
    st.subheader("🧪 Test Components")
    
    component_type = st.selectbox("Component:", ["Resistors", "Tungsten Bulbs"])
    
    if component_type == "Resistors":
        resistor_values = [10, 47, 100, 220, 470, 1000]
        selected_resistor = st.selectbox("Value (Ω):", resistor_values)
        st.info(f"Selected: {selected_resistor}Ω resistor")
    
    if st.button("✅ Complete Equipment Setup"):
        st.session_state.completed_sections.add(current_section)
        st.success("Equipment setup completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PART A: OHMIC VERIFICATION ===
elif current_section == "📊 Part A: Ohmic Verification":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("📊 Part A: Ohmic Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        resistance_a = st.number_input("Resistance (Ω)", value=100.0, key="res_a")
        voltage_range = st.slider("Voltage Range (V)", 0.0, 20.0, (0.0, 12.0), key="v_range_a")
        voltage_step = st.number_input("Voltage Step (V)", value=1.0, key="v_step_a")
    
    with col2:
        include_errors = st.checkbox("Include measurement errors", value=True)
        voltage_error = st.number_input("Voltage Error (%)", value=0.5) if include_errors else 0
        current_error = st.number_input("Current Error (%)", value=0.5) if include_errors else 0
    
    if st.button("🔬 Run Ohmic Verification", key="run_a", type="primary"):
        try:
            voltages = np.arange(voltage_range[0], voltage_range[1] + voltage_step, voltage_step)
            
            data = []
            for voltage in voltages:
                current_ideal = voltage / resistance_a if resistance_a > 0 else 0
                
                if include_errors:
                    voltage_measured = generate_experimental_error(voltage, voltage_error) if voltage > 0 else 0
                    current_measured = generate_experimental_error(current_ideal, current_error) if current_ideal > 0 else 0
                else:
                    voltage_measured = voltage
                    current_measured = current_ideal
                
                resistance_calculated = voltage_measured / current_measured if current_measured > 0 else 0
                power = voltage_measured * current_measured
                
                data.append({
                    'Voltage (V)': voltage_measured,
                    'Current (A)': current_measured,
                    'Resistance (Ω)': resistance_calculated,
                    'Power (W)': power
                })
            
            df_ohmic = pd.DataFrame(data)
            
            # Calculate R²
            if len(df_ohmic) > 1:
                X = df_ohmic['Voltage (V)'].values.reshape(-1, 1)
                y = df_ohmic['Current (A)'].values
                if np.var(X) > 0:
                    model = LinearRegression().fit(X, y)
                    r_squared = r2_score(y, model.predict(X))
                else:
                    r_squared = 0
            else:
                r_squared = 0
            
            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                mean_resistance = df_ohmic['Resistance (Ω)'].mean()
                st.metric("Mean Resistance", f"{mean_resistance:.3f} Ω")
            with col2:
                max_power = df_ohmic['Power (W)'].max()
                st.metric("Max Power", f"{max_power:.3f} W")
            with col3:
                st.metric("Linearity (R²)", f"{r_squared:.4f}")
            
            # V-I plot
            fig1 = px.scatter(df_ohmic, x='Voltage (V)', y='Current (A)', 
                             title='V-I Characteristic Curve',
                             trendline="ols")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Data table
            st.subheader("📊 Measurement Data")
            st.dataframe(df_ohmic, use_container_width=True)
            
            # Export
            csv_data = df_ohmic.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv_data,
                file_name=f"ohmic_verification_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Store data
            st.session_state.experiment_data.append({
                'section': 'Part A',
                'data': df_ohmic,
                'parameters': {'resistance': resistance_a, 'r_squared': r_squared}
            })
            
            st.session_state.completed_sections.add(current_section)
            st.success("🎉 Part A completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Experiment failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PART B: NON-OHMIC ANALYSIS ===
elif current_section == "💡 Part B: Non-Ohmic Analysis":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("💡 Part B: Non-Ohmic Behavior - Tungsten Bulb")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bulb_specs = {
            "12V 10W Standard": {"voltage": 12, "power": 10, "R0": 1.44},
            "12V 55W Automotive": {"voltage": 12, "power": 55, "R0": 0.26},
            "120V 60W Household": {"voltage": 120, "power": 60, "R0": 240}
        }
        
        selected_bulb = st.selectbox("Bulb Type:", list(bulb_specs.keys()))
        bulb_info = bulb_specs[selected_bulb]
        
        st.info(f"Rated: {bulb_info['voltage']}V, {bulb_info['power']}W")
        st.info(f"Cold R₀: {bulb_info['R0']:.2f}Ω")
    
    with col2:
        R0_bulb = st.number_input("Cold Resistance R₀ (Ω)", value=bulb_info['R0'])
        alpha_bulb = st.number_input("Temperature Coefficient α", value=0.0045, format="%.6f")
        voltage_range_bulb = st.slider("Voltage Range (V)", 0.0, 20.0, (0.0, 12.0))
    
    if st.button("🔬 Run Non-Ohmic Analysis", key="run_bulb", type="primary"):
        try:
            voltages_bulb = np.linspace(voltage_range_bulb[0], voltage_range_bulb[1], 50)
            
            bulb_data = []
            for voltage in voltages_bulb:
                if voltage == 0:
                    current = 0
                    resistance_hot = R0_bulb
                    temperature = 293
                else:
                    # Iterative solution for temperature-dependent resistance
                    resistance_est = R0_bulb
                    for _ in range(10):
                        current_est = voltage / resistance_est
                        power_est = voltage * current_est
                        temp_rise = 0.1 * voltage**2  # Simplified model
                        temperature_new = 293 + temp_rise
                        resistance_new = R0_bulb * (1 + alpha_bulb * (temperature_new - 293))
                        
                        if abs(resistance_new - resistance_est) < 0.001:
                            break
                        resistance_est = resistance_new
                    
                    current = current_est
                    resistance_hot = resistance_new
                    temperature = temperature_new
                
                power = voltage * current
                
                bulb_data.append({
                    'Voltage (V)': voltage,
                    'Current (A)': current,
                    'Resistance_Hot (Ω)': resistance_hot,
                    'Temperature (K)': temperature,
                    'Power (W)': power,
                    'Resistance_Ratio': resistance_hot / R0_bulb
                })
            
            df_bulb = pd.DataFrame(bulb_data)
            
            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                max_current = df_bulb['Current (A)'].max()
                st.metric("Max Current", f"{max_current:.3f} A")
            with col2:
                max_temp = df_bulb['Temperature (K)'].max()
                st.metric("Max Temperature", f"{max_temp:.0f} K")
            with col3:
                max_ratio = df_bulb['Resistance_Ratio'].max()
                st.metric("R_hot/R_cold", f"{max_ratio:.1f}x")
            
            # Visualizations
            tab1, tab2, tab3 = st.tabs(["V-I Curve", "Temperature", "Data"])
            
            with tab1:
                fig1 = px.scatter(df_bulb, x='Voltage (V)', y='Current (A)',
                                 title='Non-Ohmic V-I Characteristic')
                # Add ohmic line for comparison
                ohmic_current = voltages_bulb / R0_bulb
                fig1.add_scatter(x=voltages_bulb, y=ohmic_current, 
                               mode='lines', name='Ohmic (Cold R)', 
                               line=dict(dash='dash'))
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = px.line(df_bulb, x='Voltage (V)', y='Temperature (K)',
                              title='Temperature vs Voltage')
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                st.dataframe(df_bulb, use_container_width=True)
            
            # Store data
            st.session_state.experiment_data.append({
                'section': 'Part B',
                'data': df_bulb,
                'parameters': {'bulb_type': selected_bulb, 'R0': R0_bulb}
            })
            
            st.session_state.completed_sections.add(current_section)
            st.success("🎉 Part B completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PART C: STATISTICAL ANALYSIS ===
elif current_section == "📈 Part C: Statistical Analysis":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("📈 Part C: Statistical Analysis & Repeatability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_trials = st.number_input("Number of Trials", value=10, min_value=3, max_value=50)
        base_resistance = st.number_input("Base Resistance (Ω)", value=100.0)
        test_voltage = st.number_input("Test Voltage (V)", value=12.0)
    
    with col2:
        systematic_error = st.number_input("Systematic Error (%)", value=0.1)
        random_error = st.number_input("Random Error (%)", value=0.5)
        confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
    
    if st.button("🔬 Run Statistical Analysis", key="run_stat", type="primary"):
        try:
            trial_data = []
            
            for trial in range(num_trials):
                current_ideal = test_voltage / base_resistance
                
                # Add errors
                voltage_error = systematic_error/100 + random_error/100 * np.random.normal()
                current_error = systematic_error/100 + random_error/100 * np.random.normal()
                
                voltage_measured = test_voltage * (1 + voltage_error)
                current_measured = current_ideal * (1 + current_error)
                resistance_measured = voltage_measured / current_measured
                
                trial_data.append({
                    'Trial': trial + 1,
                    'Voltage (V)': voltage_measured,
                    'Current (A)': current_measured,
                    'Resistance (Ω)': resistance_measured
                })
            
            df_stat = pd.DataFrame(trial_data)
            
            # Statistics
            mean_r = df_stat['Resistance (Ω)'].mean()
            std_r = df_stat['Resistance (Ω)'].std()
            cv_r = (std_r / mean_r) * 100
            
            # Confidence interval
            confidence_z = {90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
            margin_error = confidence_z * std_r / np.sqrt(num_trials)
            
            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                accuracy = (1 - abs(mean_r - base_resistance) / base_resistance) * 100
                st.metric("Accuracy", f"{accuracy:.2f}%")
            with col2:
                st.metric("Precision (CV)", f"{cv_r:.3f}%")
            with col3:
                st.metric(f"{confidence_level}% CI", f"±{margin_error:.3f}Ω")
            
            # Visualizations
            tab1, tab2 = st.tabs(["Control Chart", "Distribution"])
            
            with tab1:
                fig1 = px.line(df_stat, x='Trial', y='Resistance (Ω)',
                              title='Resistance Control Chart',
                              markers=True)
                fig1.add_hline(y=mean_r, line_dash="solid", 
                              annotation_text=f"Mean: {mean_r:.3f}Ω")
                fig1.add_hline(y=mean_r + 3*std_r, line_dash="dash",
                              annotation_text="UCL")
                fig1.add_hline(y=mean_r - 3*std_r, line_dash="dash",
                              annotation_text="LCL")
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = px.histogram(df_stat, x='Resistance (Ω)',
                                   title='Resistance Distribution',
                                   nbins=10)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Data table
            st.subheader("📊 Statistical Data")
            st.dataframe(df_stat, use_container_width=True)
            
            # Store data
            st.session_state.experiment_data.append({
                'section': 'Part C',
                'data': df_stat,
                'parameters': {'trials': num_trials, 'mean_r': mean_r, 'std_r': std_r}
            })
            
            st.session_state.completed_sections.add(current_section)
            st.success("🎉 Part C completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Statistical analysis failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PART D: ADVANCED TESTING ===
elif current_section == "🔍 Part D: Advanced Testing":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("🔍 Part D: Advanced Testing & Edge Cases")
    
    testing_scenario = st.selectbox("Select Testing Scenario:", [
        "🌡️ Extreme Temperature Effects",
        "⚡ High Voltage Breakdown",
        "🌊 AC vs DC Comparison",
        "🔬 Quantum Effects"
    ])
    
    if testing_scenario == "🌡️ Extreme Temperature Effects":
        st.subheader("🌡️ Extreme Temperature Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temp_min = st.number_input("Min Temperature (K)", value=77.0)
            temp_max = st.number_input("Max Temperature (K)", value=500.0)
            material_alpha = st.number_input("Temperature Coefficient", value=0.004)
        
        with col2:
            base_resistance = st.number_input("Base Resistance at 293K (Ω)", value=100.0)
            
        if st.button("🔬 Run Temperature Test"):
            temperatures = np.linspace(temp_min, temp_max, 50)
            resistances = base_resistance * (1 + material_alpha * (temperatures - 293))
            
            temp_data = pd.DataFrame({
                'Temperature (K)': temperatures,
                'Temperature (°C)': temperatures - 273.15,
                'Resistance (Ω)': resistances,
                'Resistance_Ratio': resistances / base_resistance
            })
            
            fig_temp = px.line(temp_data, x='Temperature (K)', y='Resistance (Ω)',
                              title='Resistance vs Temperature')
            st.plotly_chart(fig_temp, use_container_width=True)
            
            st.dataframe(temp_data, use_container_width=True)
            st.success("✅ Temperature testing completed!")
    
    elif testing_scenario == "⚡ High Voltage Breakdown":
        st.subheader("⚡ High Voltage Breakdown Analysis")
        
        max_voltage = st.number_input("Maximum Test Voltage (kV)", value=10.0)
        breakdown_threshold = st.number_input("Expected Breakdown (kV)", value=5.0)
        
        if st.button("⚡ Run Breakdown Test"):
            voltages = np.linspace(0, max_voltage*1000, 100)
            
            # Simple breakdown model
            breakdown_prob = 1 - np.exp(-((voltages / (breakdown_threshold*1000))**3))
            current = np.where(voltages < breakdown_threshold*1000, 
                             1e-12 * voltages,  # Leakage current
                             1e-3 * voltages)   # Post-breakdown current
            
            breakdown_data = pd.DataFrame({
                'Voltage (V)': voltages,
                'Current (A)': current,
                'Breakdown_Probability': breakdown_prob
            })
            
            fig_breakdown = px.line(breakdown_data, x='Voltage (V)', y='Current (A)',
                                   title='High Voltage Breakdown Characteristic',
                                   log_y=True)
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
            st.success("⚡ Breakdown testing completed!")
    
    if st.button("✅ Complete Advanced Testing"):
        st.session_state.completed_sections.add(current_section)
        st.success("🎉 Part D completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PART E: GRAPH ANALYSIS ===
elif current_section == "📊 Part E: Graph Analysis":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("📊 Part E: Interactive Graph Analysis")
    
    if st.session_state.experiment_data:
        st.subheader("📈 Previous Experiment Data Analysis")
        
        # Select experiment data
        experiment_names = [f"{exp['section']}" for exp in st.session_state.experiment_data]
        selected_exp = st.selectbox("Select Experiment:", experiment_names)
        
        if selected_exp:
            exp_index = experiment_names.index(selected_exp)
            exp_data = st.session_state.experiment_data[exp_index]['data']
            
            # Column selection for plotting
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis:", exp_data.columns)
            with col2:
                y_column = st.selectbox("Y-axis:", exp_data.columns)
            
            # Plot options
            plot_type = st.selectbox("Plot Type:", ["Scatter", "Line", "Both"])
            
            if st.button("📊 Generate Analysis Graph"):
                fig = go.Figure()
                
                if plot_type in ["Scatter", "Both"]:
                    fig.add_trace(go.Scatter(
                        x=exp_data[x_column], 
                        y=exp_data[y_column],
                        mode='markers',
                        name='Data Points',
                        marker=dict(size=8)
                    ))
                
                if plot_type in ["Line", "Both"]:
                    fig.add_trace(go.Scatter(
                        x=exp_data[x_column], 
                        y=exp_data[y_column],
                        mode='lines',
                        name='Trend Line'
                    ))
                
                # Add regression line
                if len(exp_data) > 1:
                    X = exp_data[x_column].values.reshape(-1, 1)
                    y = exp_data[y_column].values
                    
                    if np.var(X) > 0:
                        model = LinearRegression().fit(X, y)
                        y_pred = model.predict(X)
                        r2 = r2_score(y, y_pred)
                        
                        fig.add_trace(go.Scatter(
                            x=exp_data[x_column], 
                            y=y_pred,
                            mode='lines',
                            name=f'Regression (R²={r2:.3f})',
                            line=dict(dash='dash', color='red')
                        ))
                
                fig.update_layout(
                    title=f'Analysis: {selected_exp}',
                    xaxis_title=x_column,
                    yaxis_title=y_column,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical analysis
                correlation = exp_data[x_column].corr(exp_data[y_column])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correlation", f"{correlation:.4f}")
                with col2:
                    st.metric("Data Points", len(exp_data))
                with col3:
                    if 'r2' in locals():
                        st.metric("R² Value", f"{r2:.4f}")
    else:
        st.warning("No experiment data available. Complete some experiments first!")
    
    # Custom data analysis
    st.subheader("📈 Custom Data Analysis")
    
    if st.button("🎲 Generate Sample Data for Analysis"):
        # Generate sample V-I data
        voltages = np.linspace(0, 12, 20)
        currents = voltages / 100 + np.random.normal(0, 0.001, len(voltages))
        
        sample_data = pd.DataFrame({
            'Voltage (V)': voltages,
            'Current (A)': currents,
            'Power (W)': voltages * currents,
            'Resistance (Ω)': voltages / currents
        })
        
        # Interactive plot
        fig_sample = px.scatter(sample_data, x='Voltage (V)', y='Current (A)',
                               title='Sample V-I Characteristic',
                               trendline="ols")
        st.plotly_chart(fig_sample, use_container_width=True)
        
        st.dataframe(sample_data, use_container_width=True)
    
    if st.button("✅ Complete Graph Analysis"):
        st.session_state.completed_sections.add(current_section)
        st.success("🎉 Part E completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PART F: SERIES & PARALLEL CIRCUITS ===
elif current_section == "🔌 Part F: Series & Parallel Circuits":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("🔌 Part F: Series & Parallel Circuit Analysis")
    
    st.markdown("""
    <div class="theory-box">
    <h4>🎯 Objective</h4>
    <p>Investigate equivalent resistance in series and parallel resistor combinations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Circuit configuration
    st.subheader("⚙️ Circuit Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Resistor Values:**")
        num_resistors = st.number_input("Number of Resistors", value=3, min_value=2, max_value=5)
        
        resistor_values = []
        for i in range(num_resistors):
            value = st.number_input(f"R{i+1} (Ω)", value=100.0 * (i+1), key=f"r{i}")
            resistor_values.append(value)
    
    with col2:
        st.markdown("**Circuit Configuration:**")
        connection_type = st.radio("Connection Type:", ["Series", "Parallel"])
        supply_voltage = st.number_input("Supply Voltage (V)", value=12.0)
        
        # Calculate equivalent resistance
        R_eq = equivalent_resistance(resistor_values, connection_type)
        st.metric("Equivalent Resistance", f"{R_eq:.2f} Ω")
        
        # Total current
        I_total = supply_voltage / R_eq if R_eq > 0 else 0
        st.metric("Total Current", f"{I_total:.4f} A")
    
    if st.button("🔬 Analyze Circuit", key="run_circuit", type="primary"):
        try:
            # Calculate individual currents and voltages
            circuit_data = []
            
            if connection_type == "Series":
                # Series: same current, voltages add up
                current = I_total
                for i, R in enumerate(resistor_values):
                    voltage = current * R
                    power = voltage * current
                    
                    circuit_data.append({
                        'Component': f'R{i+1}',
                        'Resistance (Ω)': R,
                        'Voltage (V)': voltage,
                        'Current (A)': current,
                        'Power (W)': power
                    })
            
            else:  # Parallel
                # Parallel: same voltage, currents add up
                voltage = supply_voltage
                for i, R in enumerate(resistor_values):
                    current = voltage / R
                    power = voltage * current
                    
                    circuit_data.append({
                        'Component': f'R{i+1}',
                        'Resistance (Ω)': R,
                        'Voltage (V)': voltage,
                        'Current (A)': current,
                        'Power (W)': power
                    })
            
            df_circuit = pd.DataFrame(circuit_data)
            
            # Results summary
            st.subheader("📊 Circuit Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_power = df_circuit['Power (W)'].sum()
                st.metric("Total Power", f"{total_power:.3f} W")
            
            with col2:
                max_power_component = df_circuit.loc[df_circuit['Power (W)'].idxmax(), 'Component']
                max_power = df_circuit['Power (W)'].max()
                st.metric("Highest Power", f"{max_power:.3f} W", f"in {max_power_component}")
            
            with col3:
                verification_power = supply_voltage * I_total
                power_error = abs(total_power - verification_power) / verification_power * 100 if verification_power > 0 else 0
                st.metric("Power Check", f"{power_error:.2f}% error", "P = VI verification")
            
            # Data table
            st.subheader("📋 Component Analysis")
            st.dataframe(df_circuit, use_container_width=True)
            
            # Visualizations
            tab1, tab2, tab3 = st.tabs(["Power Distribution", "Current/Voltage", "Circuit Diagram"])
            
            with tab1:
                fig_power = px.bar(df_circuit, x='Component', y='Power (W)',
                                  title=f'Power Distribution - {connection_type} Circuit')
                st.plotly_chart(fig_power, use_container_width=True)
            
            with tab2:
                fig_vi = make_subplots(rows=1, cols=2,
                                      subplot_titles=['Voltage Distribution', 'Current Distribution'])
                
                fig_vi.add_trace(go.Bar(x=df_circuit['Component'], y=df_circuit['Voltage (V)'],
                                       name='Voltage'), row=1, col=1)
                fig_vi.add_trace(go.Bar(x=df_circuit['Component'], y=df_circuit['Current (A)'],
                                       name='Current'), row=1, col=2)
                
                fig_vi.update_layout(title_text=f"Voltage and Current - {connection_type} Circuit")
                st.plotly_chart(fig_vi, use_container_width=True)
            
            with tab3:
                # Simple text-based circuit diagram
                st.markdown("**Circuit Diagram:**")
                
                if connection_type == "Series":
                    diagram = "---[R1]---[R2]---[R3]---"
                    for i, R in enumerate(resistor_values):
                        diagram = diagram.replace(f"[R{i+1}]", f"[{R:.0f}Ω]")
                else:
                    diagram = """
                    ┌─[R1]─┐
                    ├─[R2]─┤
                    └─[R3]─┘
                    """
                    for i, R in enumerate(resistor_values):
                        diagram = diagram.replace(f"[R{i+1}]", f"[{R:.0f}Ω]")
                
                st.code(diagram)
                
                # Equivalent circuit
                st.markdown(f"**Equivalent Circuit:** ---[{R_eq:.2f}Ω]---")
            
            # Theoretical verification
            st.subheader("🔬 Theoretical Verification")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Kirchhoff's Laws Check:**")
                
                if connection_type == "Series":
                    voltage_sum = df_circuit['Voltage (V)'].sum()
                    voltage_error = abs(voltage_sum - supply_voltage) / supply_voltage * 100
                    st.write(f"• Voltage sum: {voltage_sum:.3f}V (KVL)")
                    st.write(f"• Supply voltage: {supply_voltage:.3f}V")
                    st.write(f"• Error: {voltage_error:.2f}%")
                    
                    if voltage_error < 1:
                        st.success("✅ KVL verified")
                else:
                    current_sum = df_circuit['Current (A)'].sum()
                    current_error = abs(current_sum - I_total) / I_total * 100 if I_total > 0 else 0
                    st.write(f"• Current sum: {current_sum:.6f}A (KCL)")
                    st.write(f"• Supply current: {I_total:.6f}A")
                    st.write(f"• Error: {current_error:.2f}%")
                    
                    if current_error < 1:
                        st.success("✅ KCL verified")
            
            with col2:
                st.markdown("**Equivalent Resistance Check:**")
                
                if connection_type == "Series":
                    R_eq_theory = sum(resistor_values)
                else:
                    R_eq_theory = 1 / sum(1/R for R in resistor_values)
                
                req_error = abs(R_eq - R_eq_theory) / R_eq_theory * 100
                st.write(f"• Calculated: {R_eq:.3f}Ω")
                st.write(f"• Theoretical: {R_eq_theory:.3f}Ω")
                st.write(f"• Error: {req_error:.2f}%")
                
                if req_error < 0.1:
                    st.success("✅ Equivalent resistance verified")
            
            # Export data
            csv_data = df_circuit.to_csv(index=False)
            st.download_button(
                label="📥 Download Circuit Data (CSV)",
                data=csv_data,
                file_name=f"circuit_analysis_{connection_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Store data
            st.session_state.experiment_data.append({
                'section': 'Part F',
                'data': df_circuit,
                'parameters': {
                    'connection_type': connection_type,
                    'equivalent_resistance': R_eq,
                    'supply_voltage': supply_voltage,
                    'resistor_values': resistor_values
                }
            })
            
            st.session_state.completed_sections.add(current_section)
            st.success("🎉 Part F: Series & Parallel Circuit Analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Circuit analysis failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PART G: TEMPERATURE EFFECTS ===
elif current_section == "🌡️ Part G: Temperature Effects":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("🌡️ Part G: Temperature Effects on Resistance")
    
    st.markdown("""
    <div class="theory-box">
    <h4>🎯 Objective</h4>
    <p>Study how temperature affects resistance in different materials and understand temperature coefficients.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Material selection and parameters
    st.subheader("🧪 Material Selection & Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Material Database:**")
        materials = {
            'Copper': {'R0': 100.0, 'alpha': 0.00393, 'type': 'Metal', 'color': 'orange'},
            'Aluminum': {'R0': 100.0, 'alpha': 0.00429, 'type': 'Metal', 'color': 'silver'},
            'Tungsten': {'R0': 100.0, 'alpha': 0.0045, 'type': 'Metal', 'color': 'gray'},
            'Carbon': {'R0': 100.0, 'alpha': -0.0005, 'type': 'Non-metal', 'color': 'black'},
            'Silicon': {'R0': 1000.0, 'alpha': -0.075, 'type': 'Semiconductor', 'color': 'blue'},
            'Germanium': {'R0': 500.0, 'alpha': -0.048, 'type': 'Semiconductor', 'color': 'purple'}
        }
        
        selected_materials = st.multiselect(
            "Select Materials to Compare:",
            list(materials.keys()),
            default=['Copper', 'Tungsten', 'Silicon']
        )
    
    with col2:
        st.markdown("**Temperature Range:**")
        temp_range = st.slider("Temperature Range (°C)", -200, 1000, (-50, 200))
        temp_steps = st.number_input("Temperature Steps", value=50, min_value=10, max_value=200)
        
        reference_temp = st.number_input("Reference Temperature (°C)", value=20.0)
        
        # Display selected material properties
        if selected_materials:
            st.markdown("**Selected Materials:**")
            for material in selected_materials:
                props = materials[material]
                st.write(f"• {material}: α = {props['alpha']:.6f} /°C ({props['type']})")
    
    if st.button("🔬 Run Temperature Analysis", key="run_temp_analysis", type="primary"):
        try:
            if not selected_materials:
                st.error("Please select at least one material!")
                st.stop()
            
            # Generate temperature data
            temperatures_c = np.linspace(temp_range[0], temp_range[1], temp_steps)
            temperatures_k = temperatures_c + 273.15
            ref_temp_k = reference_temp + 273.15
            
            # Calculate resistance for each material
            temp_data = []
            
            for material in selected_materials:
                props = materials[material]
                R0 = props['R0']
                alpha = props['alpha']
                
                # Temperature-dependent resistance: R(T) = R0[1 + α(T - T0)]
                temp_diff = temperatures_c - reference_temp
                resistances = R0 * (1 + alpha * temp_diff)
                
                # Handle special cases for semiconductors at extreme temperatures
                if props['type'] == 'Semiconductor' and np.any(temperatures_c > 100):
                    # Add intrinsic conductivity effects at high temperatures
                    high_temp_mask = temperatures_c > 100
                    if np.any(high_temp_mask):
                        # Simplified intrinsic behavior
                        intrinsic_factor = np.exp(-(temperatures_c[high_temp_mask] - 100) / 100)
                        resistances[high_temp_mask] *= intrinsic_factor
                
                for i, (temp_c, temp_k, resistance) in enumerate(zip(temperatures_c, temperatures_k, resistances)):
                    temp_data.append({
                        'Material': material,
                        'Temperature_C': temp_c,
                        'Temperature_K': temp_k,
                        'Resistance': resistance,
                        'Resistance_Ratio': resistance / R0,
                        'TCR': alpha,  # Temperature Coefficient of Resistance
                        'Material_Type': props['type']
                    })
            
            df_temp = pd.DataFrame(temp_data)
            
            # Results summary
            st.subheader("📊 Temperature Analysis Results")
            
            # Key metrics for each material
            metrics_cols = st.columns(len(selected_materials))
            
            for i, material in enumerate(selected_materials):
                material_data = df_temp[df_temp['Material'] == material]
                min_resistance = material_data['Resistance'].min()
                max_resistance = material_data['Resistance'].max()
                resistance_range = max_resistance / min_resistance if min_resistance > 0 else 0
                
                with metrics_cols[i]:
                    st.metric(
                        f"{material}",
                        f"{resistance_range:.1f}x",
                        f"Range: {min_resistance:.1f}-{max_resistance:.1f}Ω"
                    )
            
            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "🌡️ R vs Temperature", 
                "📈 Normalized Resistance", 
                "📊 TCR Comparison", 
                "📋 Data Table"
            ])
            
            with tab1:
                # Resistance vs Temperature for all materials
                fig_temp = go.Figure()
                
                for material in selected_materials:
                    material_data = df_temp[df_temp['Material'] == material]
                    props = materials[material]
                    
                    fig_temp.add_trace(go.Scatter(
                        x=material_data['Temperature_C'],
                        y=material_data['Resistance'],
                        mode='lines',
                        name=material,
                        line=dict(width=3, color=props['color'])
                    ))
                
                # Add reference temperature line
                fig_temp.add_vline(
                    x=reference_temp,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Reference: {reference_temp}°C"
                )
                
                fig_temp.update_layout(
                    title='Resistance vs Temperature Comparison',
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Resistance (Ω)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Physical interpretation
                st.markdown("**Physical Interpretation:**")
                st.write("• **Metals (positive TCR):** Resistance increases with temperature due to increased lattice vibrations")
                st.write("• **Semiconductors (negative TCR):** Resistance decreases with temperature due to increased charge carriers")
                st.write("• **Carbon:** Shows negative TCR due to its unique electronic structure")
            
            with tab2:
                # Normalized resistance (R/R0) vs Temperature
                fig_norm = go.Figure()
                
                for material in selected_materials:
                    material_data = df_temp[df_temp['Material'] == material]
                    props = materials[material]
                    
                    fig_norm.add_trace(go.Scatter(
                        x=material_data['Temperature_C'],
                        y=material_data['Resistance_Ratio'],
                        mode='lines',
                        name=material,
                        line=dict(width=3, color=props['color'])
                    ))
                
                # Add reference line at R/R0 = 1
                fig_norm.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="R/R₀ = 1"
                )
                
                fig_norm.update_layout(
                    title='Normalized Resistance vs Temperature',
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Resistance Ratio (R/R₀)',
                    height=500
                )
                
                st.plotly_chart(fig_norm, use_container_width=True)
            
            with tab3:
                # TCR comparison bar chart
                tcr_data = []
                for material in selected_materials:
                    props = materials[material]
                    tcr_data.append({
                        'Material': material,
                        'TCR': props['alpha'] * 1000,  # Convert to /K × 10³
                        'Type': props['type']
                    })
                
                tcr_df = pd.DataFrame(tcr_data)
                
                fig_tcr = px.bar(
                    tcr_df, 
                    x='Material', 
                    y='TCR',
                    color='Type',
                    title='Temperature Coefficient of Resistance Comparison',
                    labels={'TCR': 'TCR (×10⁻³ /°C)'}
                )
                
                # Add zero line
                fig_tcr.add_hline(y=0, line_dash="dash", line_color="black")
                
                st.plotly_chart(fig_tcr, use_container_width=True)
                
                # TCR explanation
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Positive TCR Materials:**")
                    positive_tcr = [m for m in selected_materials if materials[m]['alpha'] > 0]
                    for material in positive_tcr:
                        alpha = materials[material]['alpha']
                        st.write(f"• {material}: {alpha*1000:.2f} ×10⁻³ /°C")
                
                with col2:
                    st.markdown("**Negative TCR Materials:**")
                    negative_tcr = [m for m in selected_materials if materials[m]['alpha'] < 0]
                    for material in negative_tcr:
                        alpha = materials[material]['alpha']
                        st.write(f"• {material}: {alpha*1000:.2f} ×10⁻³ /°C")
            
            with tab4:
                # Complete data table with filtering
                st.markdown("**Complete Temperature Data:**")
                
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    material_filter = st.multiselect(
                        "Filter by Material:",
                        selected_materials,
                        default=selected_materials
                    )
                
                with col2:
                    temp_filter = st.slider(
                        "Temperature Range Filter (°C):",
                        float(df_temp['Temperature_C'].min()),
                        float(df_temp['Temperature_C'].max()),
                        (float(df_temp['Temperature_C'].min()), float(df_temp['Temperature_C'].max()))
                    )
                
                # Apply filters
                filtered_df = df_temp[
                    (df_temp['Material'].isin(material_filter)) &
                    (df_temp['Temperature_C'] >= temp_filter[0]) &
                    (df_temp['Temperature_C'] <= temp_filter[1])
                ]
                
                st.dataframe(filtered_df, use_container_width=True)
                
                st.markdown(f"**Showing {len(filtered_df)} of {len(df_temp)} total data points**")
            
            # Temperature coefficient analysis
            st.subheader("🔬 Temperature Coefficient Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Applications:**")
                st.write("• **Temperature Sensors:** Use materials with predictable TCR")
                st.write("• **Precision Resistors:** Choose materials with low TCR")
                st.write("• **Heating Elements:** Use materials with positive TCR for self-regulation")
                st.write("• **Thermistors:** Use semiconductors for sensitive temperature detection")
            
            with col2:
                st.markdown("**Material Selection Guide:**")
                st.write("• **High Stability:** Choose materials with |TCR| < 100 ppm/°C")
                st.write("• **Temperature Sensing:** Choose materials with high |TCR|")
                st.write("• **Power Applications:** Consider thermal management")
                st.write("• **Cryogenic Use:** Consider quantum effects at low temperatures")
            
            # Export options
            st.subheader("💾 Export Temperature Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Raw data export
                csv_data = df_temp.to_csv(index=False)
                st.download_button(
                    label="📥 Download Temperature Data (CSV)",
                    data=csv_data,
                    file_name=f"temperature_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Summary report
                report_content = f"""
TEMPERATURE EFFECTS ANALYSIS REPORT
==================================
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXPERIMENTAL PARAMETERS:
- Temperature Range: {temp_range[0]} to {temp_range[1]} °C
- Reference Temperature: {reference_temp} °C
- Materials Tested: {', '.join(selected_materials)}

MATERIALS ANALYSIS:
"""
                
                for material in selected_materials:
                    material_data = df_temp[df_temp['Material'] == material]
                    props = materials[material]
                    min_r = material_data['Resistance'].min()
                    max_r = material_data['Resistance'].max()
                    
                    report_content += f"""
{material} ({props['type']}):
- Temperature Coefficient: {props['alpha']:.6f} /°C
- Resistance Range: {min_r:.2f} - {max_r:.2f} Ω
- Resistance Ratio: {max_r/min_r:.2f}x
"""
                
                report_content += f"""

CONCLUSIONS:
✓ Temperature effects on resistance successfully characterized
✓ Material behavior consistent with theoretical predictions
✓ Data suitable for temperature coefficient verification
✓ Results applicable to sensor and precision resistor design
"""
                
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report_content,
                    file_name=f"temperature_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            # Store data
            st.session_state.experiment_data.append({
                'section': 'Part G',
                'data': df_temp,
                'parameters': {
                    'materials': selected_materials,
                    'temperature_range': temp_range,
                    'reference_temp': reference_temp,
                    'material_properties': {m: materials[m] for m in selected_materials}
                }
            })
            
            st.session_state.completed_sections.add(current_section)
            st.balloons()
            st.success("🎉 Part G: Temperature Effects Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Temperature analysis failed: {str(e)}")
            if st.checkbox("Show detailed error information", key="error_detail_g"):
                st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

# === SAFETY PROTOCOLS ===
elif current_section == "⚠️ Safety Protocols":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("⚠️ Laboratory Safety Protocols")
    
    st.markdown("""
    <div class="safety-warning">
    <h4>🛡️ Safety First</h4>
    <p>Even in virtual laboratories, understanding safety protocols is crucial for real-world applications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Safety categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Electrical Safety", 
        "🔥 Thermal Safety", 
        "🧪 Equipment Safety", 
        "📋 Emergency Procedures"
    ])
    
    with tab1:
        st.subheader("⚡ Electrical Safety Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Voltage Safety Levels:**")
            
            voltage_levels = {
                "Extra Low Voltage (ELV)": {"range": "< 50V AC, < 120V DC", "risk": "Low", "color": "green"},
                "Low Voltage (LV)": {"range": "50-1000V AC, 120-1500V DC", "risk": "Medium", "color": "orange"},
                "High Voltage (HV)": {"range": "> 1000V AC, > 1500V DC", "risk": "High", "color": "red"}
            }
            
            for level, info in voltage_levels.items():
                if info["color"] == "green":
                    st.success(f"✅ **{level}:** {info['range']} - {info['risk']} Risk")
                elif info["color"] == "orange":
                    st.warning(f"⚠️ **{level}:** {info['range']} - {info['risk']} Risk")
                else:
                    st.error(f"🚨 **{level}:** {info['range']} - {info['risk']} Risk")
        
        with col2:
            st.markdown("**Current Safety Thresholds:**")
            
            current_effects = {
                "< 1 mA": "Not perceptible",
                "1-5 mA": "Barely perceptible",
                "5-10 mA": "Painful shock",
                "10-20 mA": "Muscular control lost",
                "50-100 mA": "Ventricular fibrillation",
                "> 200 mA": "Severe burns, cardiac arrest"
            }
            
            for current, effect in current_effects.items():
                if "Not perceptible" in effect or "Barely" in effect:
                    st.success(f"✅ **{current}:** {effect}")
                elif "Painful" in effect:
                    st.warning(f"⚠️ **{current}:** {effect}")
                else:
                    st.error(f"🚨 **{current}:** {effect}")
        
        # Interactive safety calculator
        st.markdown("**⚡ Safety Assessment Tool:**")
        
        test_voltage = st.number_input("Test Voltage (V):", value=12.0, min_value=0.0)
        test_current = st.number_input("Expected Current (mA):", value=120.0, min_value=0.0)
        test_power = st.number_input("Power Dissipation (W):", value=1.44, min_value=0.0)
        
        # Assess safety level
        safety_warnings = []
        
        if test_voltage > 50:
            safety_warnings.append("🚨 HIGH VOLTAGE - Requires special precautions")
        if test_current > 10:
            safety_warnings.append("🚨 HIGH CURRENT - Risk of painful shock")
        if test_power > 25:
            safety_warnings.append("🔥 HIGH POWER - Risk of burns")
        
        if safety_warnings:
            st.error("⚠️ **Safety Warnings:**")
            for warning in safety_warnings:
                st.error(warning)
        else:
            st.success("✅ **Safe operating conditions**")
        
        # Safety equipment checklist
        st.markdown("**🛡️ Required Safety Equipment:**")
        
        safety_equipment = [
            ("Safety glasses", test_voltage > 30),
            ("Insulated gloves", test_voltage > 50),
            ("Non-conductive footwear", test_voltage > 100),
            ("Emergency shut-off", test_voltage > 50 or test_current > 100),
            ("Fire extinguisher (Class C)", test_power > 50),
            ("First aid kit", True),
            ("Emergency contact numbers", True)
        ]
        
        for equipment, required in safety_equipment:
            if required:
                st.warning(f"⚠️ **Required:** {equipment}")
            else:
                st.info(f"ℹ️ **Recommended:** {equipment}")
    
    with tab2:
        st.subheader("🔥 Thermal Safety Guidelines")
        
        st.markdown("**Temperature Hazard Assessment:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            component_temp = st.number_input("Component Temperature (°C):", value=85.0)
            ambient_temp = st.number_input("Ambient Temperature (°C):", value=25.0)
            
            temp_rise = component_temp - ambient_temp
            
            if component_temp > 100:
                st.error(f"🔥 **BURN HAZARD:** {component_temp}°C > 100°C")
            elif component_temp > 60:
                st.warning(f"⚠️ **HOT SURFACE:** {component_temp}°C > 60°C")
            else:
                st.success(f"✅ **SAFE TEMPERATURE:** {component_temp}°C")
        
        with col2:
            st.markdown("**Thermal Management:**")
            st.write("• **Heat Sinks:** For components > 50°C")
            st.write("• **Ventilation:** Ensure adequate airflow")
            st.write("• **Thermal Monitoring:** Use temperature sensors")
            st.write("• **Power Limiting:** Prevent thermal runaway")
            st.write("• **Cool-down Time:** Allow components to cool")
        
        # Thermal protection
        st.markdown("**🌡️ Thermal Protection Measures:**")
        
        protection_measures = {
            "Thermal Fuses": "Automatic disconnection at preset temperature",
            "Temperature Sensors": "Continuous monitoring and alarm",
            "Heat Sinks": "Enhanced heat dissipation",
            "Forced Cooling": "Fans or liquid cooling systems",
            "Thermal Interface Materials": "Improved heat transfer",
            "Thermal Barriers": "Protect nearby components"
        }
        
        for measure, description in protection_measures.items():
            st.info(f"🛡️ **{measure}:** {description}")
    
    with tab3:
        st.subheader("🧪 Equipment Safety Guidelines")
        
        # Equipment inspection checklist
        st.markdown("**🔍 Pre-Experiment Equipment Inspection:**")
        
        inspection_items = [
            "Power supply output voltage/current settings",
            "Meter calibration and range settings", 
            "Cable insulation integrity",
            "Connection tightness and cleanliness",
            "Grounding connections",
            "Ventilation and cooling systems",
            "Emergency stop button functionality",
            "Safety equipment availability"
        ]
        
        st.markdown("**Inspection Checklist:**")
        for item in inspection_items:
            checked = st.checkbox(f"✓ {item}", key=f"inspect_{item}")
        
        # Equipment limits
        st.markdown("**📊 Equipment Operating Limits:**")
        
        equipment_limits = {
            "Power Supply": {"Voltage": "0-50V", "Current": "0-10A", "Power": "500W"},
            "Digital Multimeter": {"Voltage": "0-1000V", "Current": "0-20A", "Resistance": "0-200MΩ"},
            "Oscilloscope": {"Voltage": "±400V", "Frequency": "DC-100MHz", "Input Z": "1MΩ"},
            "Function Generator": {"Voltage": "0-20Vpp", "Frequency": "1mHz-25MHz", "Output Z": "50Ω"}
        }
        
        for equipment, limits in equipment_limits.items():
            st.info(f"🔧 **{equipment}:**")
            for parameter, limit in limits.items():
                st.write(f"   • {parameter}: {limit}")
    
    with tab4:
        st.subheader("📋 Emergency Procedures")
        
        # Emergency scenarios
        emergency_scenarios = {
            "⚡ Electrical Shock": {
                "immediate": [
                    "Do NOT touch the victim if still in contact with electricity",
                    "Turn off power source immediately",
                    "Use non-conductive object to separate victim from source",
                    "Call emergency services",
                    "Check breathing and pulse",
                    "Begin CPR if necessary"
                ],
                "prevention": [
                    "Use proper lockout/tagout procedures",
                    "Wear appropriate PPE",
                    "Test circuits before touching",
                    "Keep one hand behind back when probing"
                ]
            },
            "🔥 Electrical Fire": {
                "immediate": [
                    "Turn off electrical power if safely possible",
                    "Use Class C fire extinguisher",
                    "Never use water on electrical fires",
                    "Evacuate area if fire cannot be controlled",
                    "Call fire department",
                    "Ventilate area after fire is extinguished"
                ],
                "prevention": [
                    "Regular inspection of wiring",
                    "Avoid overloading circuits",
                    "Keep combustibles away from electrical equipment",
                    "Maintain proper ventilation"
                ]
            },
            "🔥 Component Overheating": {
                "immediate": [
                    "Turn off power immediately",
                    "Do not touch hot components",
                    "Allow cooling time before investigation",
                    "Check for damage before restarting",
                    "Investigate cause of overheating"
                ],
                "prevention": [
                    "Monitor component temperatures",
                    "Ensure adequate ventilation",
                    "Use proper heat sinks",
                    "Stay within component ratings"
                ]
            }
        }
        
        selected_scenario = st.selectbox("Select Emergency Scenario:", list(emergency_scenarios.keys()))
        
        if selected_scenario:
            scenario_info = emergency_scenarios[selected_scenario]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**🚨 Immediate Actions for {selected_scenario}:**")
                for i, action in enumerate(scenario_info["immediate"], 1):
                    st.error(f"{i}. {action}")
            
            with col2:
                st.markdown(f"**🛡️ Prevention Measures:**")
                for i, measure in enumerate(scenario_info["prevention"], 1):
                    st.success(f"{i}. {measure}")
        
        # Emergency contacts
        st.markdown("**📞 Emergency Contact Information:**")
        
        emergency_contacts = {
            "Campus Security": "Ext. 2911",
            "Campus Health Center": "Ext. 2922", 
            "Fire Department": "16 (Pakistan)",
            "Police": "15 (Pakistan)",
            "Ambulance": "115 (Pakistan)",
            "Lab Supervisor": "Ext. 3001",
            "Department Head": "Ext. 3010",
            "Facilities Management": "Ext. 2950"
        }
        
        contact_cols = st.columns(2)
        
        for i, (service, number) in enumerate(emergency_contacts.items()):
            col = contact_cols[i % 2]
            with col:
                st.info(f"📞 **{service}:** {number}")
        
        # Emergency kit contents
        st.markdown("**🎒 Emergency Kit Contents:**")
        
        kit_contents = [
            "First aid supplies (bandages, antiseptic, burn gel)",
            "Class C fire extinguisher",
            "Emergency flashlight with batteries",
            "Emergency shut-off procedure card",
            "Safety data sheets (SDS) for all chemicals",
            "Emergency contact list",
            "Spill cleanup materials",
            "Insulated rescue hook"
        ]
        
        for item in kit_contents:
            st.write(f"• {item}")
    
    # Safety quiz
    st.subheader("🧠 Safety Knowledge Check")
    
    with st.expander("Take Safety Quiz"):
        quiz_score = 0
        total_questions = 3
        
        # Question 1
        q1 = st.radio(
            "1. What is the maximum safe voltage for direct human contact?",
            ["12V", "24V", "50V", "120V"]
        )
        if q1 == "50V":
            st.success("✅ Correct!")
            quiz_score += 1
        elif q1:
            st.error("❌ Incorrect. The safe limit is 50V AC.")
        
        # Question 2
        q2 = st.radio(
            "2. What type of fire extinguisher should be used for electrical fires?",
            ["Class A (Water)", "Class B (Foam)", "Class C (CO₂)", "Class D (Dry Powder)"]
        )
        if q2 == "Class C (CO₂)":
            st.success("✅ Correct!")
            quiz_score += 1
        elif q2:
            st.error("❌ Incorrect. Use Class C extinguishers for electrical fires.")
        
        # Question 3
        q3 = st.radio(
            "3. When should you use lockout/tagout procedures?",
            ["Only during maintenance", "Before any electrical work", "Only for high voltage", "Never in labs"]
        )
        if q3 == "Before any electrical work":
            st.success("✅ Correct!")
            quiz_score += 1
        elif q3:
            st.error("❌ Incorrect. Always use lockout/tagout before electrical work.")
        
        if quiz_score == total_questions:
            st.balloons()
            st.success(f"🎉 Perfect score! {quiz_score}/{total_questions}")
        elif quiz_score > 0:
            st.info(f"Good job! Score: {quiz_score}/{total_questions}")
        
    if st.button("✅ Complete Safety Training"):
        st.session_state.completed_sections.add(current_section)
        st.success("🎉 Safety protocols training completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === REPORT GENERATION & EXPORT ===
elif current_section == "📝 Report Generation & Export":
    st.markdown('<div class="experiment-section">', unsafe_allow_html=True)
    st.header("📝 Laboratory Report Generation & Export")
    
    if not st.session_state.experiment_data:
        st.warning("⚠️ No experiment data available. Complete some experiments first!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
        <h4>📊 Comprehensive Laboratory Report</h4>
        <p>Generate a complete laboratory report including all experiments, analysis, and conclusions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Report configuration
        st.subheader("⚙️ Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Report details
            student_name = st.text_input("Student Name:", value="Student Name")
            student_id = st.text_input("Student ID:", value="2021-CS-001")
            course_code = st.text_input("Course Code:", value="PHY-101")
            instructor_name = st.text_input("Instructor:", value="Dr. Physics Professor")
            
        with col2:
            # Report options
            include_raw_data = st.checkbox("Include raw data tables", value=True)
            include_graphs = st.checkbox("Include graphs and visualizations", value=True)
            include_analysis = st.checkbox("Include statistical analysis", value=True)
            include_conclusions = st.checkbox("Include conclusions and discussion", value=True)
            
            report_format = st.selectbox("Report Format:", ["Comprehensive", "Summary", "Data Only"])
        
        # Experiment selection
        st.subheader("📋 Experiment Selection")
        
        available_experiments = [exp['section'] for exp in st.session_state.experiment_data]
        selected_experiments = st.multiselect(
            "Select experiments to include:",
            available_experiments,
            default=available_experiments
        )
        
        # Progress summary
        st.subheader("📊 Laboratory Progress Summary")
        
        progress_data = []
        for section in lab_sections:
            completed = section in st.session_state.completed_sections
            has_data = any(exp['section'] == section for exp in st.session_state.experiment_data)
            
            progress_data.append({
                'Section': section,
                'Completed': '✅' if completed else '❌',
                'Has Data': '📊' if has_data else '❌',
                'Status': 'Complete' if completed else 'Incomplete'
            })
        
        progress_df = pd.DataFrame(progress_data)
        st.dataframe(progress_df, use_container_width=True)
        
        # Generate report
        if st.button("📄 Generate Laboratory Report", type="primary"):
                # Generate comprehensive report content
                report_content = f"""
OHMS LAW INTERACTIVE LABORATORY REPORT
=====================================

STUDENT INFORMATION:
Student Name: {student_name}
Student ID: {student_id}
Course: {course_code}
Instructor: {instructor_name}
Date: {datetime.datetime.now().strftime('%Y-%m-%d')}
Laboratory: Ohm's Law Interactive Virtual Laboratory

EXECUTIVE SUMMARY:
This report presents the results of a comprehensive virtual laboratory investigation 
of Ohm's Law and related electrical phenomena. The experiments were conducted using 
an interactive simulation system developed for the University of Management and 
Technology Physics Department.

OBJECTIVES:
1. Verify Ohm's Law experimentally for ohmic resistors
2. Investigate non-ohmic behavior in tungsten filament bulbs  
3. Perform statistical analysis of measurement repeatability
4. Explore advanced electrical phenomena and edge cases
5. Analyze temperature effects on electrical resistance
6. Study series and parallel circuit configurations

EXPERIMENTAL METHODOLOGY:
All experiments were conducted using a virtual laboratory environment that simulates
real electrical measurement equipment including:
- Variable voltage power supplies (0-50V)
- Digital multimeters for voltage and current measurement
- Various resistive components (resistors, tungsten bulbs)
- Temperature control systems
- Statistical analysis tools

"""
                
                # Add experiment-specific results
                for exp in st.session_state.experiment_data:
                    if exp['section'] in selected_experiments:
                        section = exp['section']
                        data = exp['data']
                        params = exp.get('parameters', {})
                        
                        report_content += f"""

{section.upper()} RESULTS:
{'='*50}
"""
                        
                        if section == "Part A":
                            resistance = params.get('resistance', 'N/A')
                            r_squared = params.get('r_squared', 'N/A')
                            mean_resistance = data['Resistance (Ω)'].mean() if 'Resistance (Ω)' in data.columns else 'N/A'
                            
                            report_content += f"""
Experimental Parameters:
- Test Resistance: {resistance} Ω
- Voltage Range: Variable
- Number of Data Points: {len(data)}

Key Results:
- Mean Measured Resistance: {mean_resistance:.4f} Ω
- Linearity (R²): {r_squared:.4f}
- Ohmic Behavior: {'Confirmed' if r_squared > 0.98 else 'Deviations observed'}

Analysis:
The V-I characteristic demonstrates {'excellent' if r_squared > 0.98 else 'good' if r_squared > 0.95 else 'moderate'} 
linearity, confirming Ohm's Law within experimental uncertainties.
"""
                        
                        elif section == "Part B":
                            bulb_type = params.get('bulb_type', 'N/A')
                            R0 = params.get('R0', 'N/A')
                            max_temp = data['Temperature (K)'].max() if 'Temperature (K)' in data.columns else 'N/A'
                            max_ratio = data['Resistance_Ratio'].max() if 'Resistance_Ratio' in data.columns else 'N/A'
                            
                            report_content += f"""
Experimental Parameters:
- Bulb Type: {bulb_type}
- Cold Resistance (R₀): {R0} Ω
- Maximum Temperature: {max_temp:.0f} K ({max_temp-273:.0f}°C)

Key Results:
- Maximum Resistance Ratio: {max_ratio:.1f}x
- Non-ohmic Behavior: Confirmed
- Temperature Dependence: Demonstrated

Analysis:
The tungsten filament exhibits strong non-ohmic behavior due to temperature-dependent
resistance, with resistance increasing by factor of {max_ratio:.1f} due to heating.
"""
                        
                        elif section == "Part C":
                            trials = params.get('trials', 'N/A')
                            mean_r = params.get('mean_r', 'N/A')
                            std_r = params.get('std_r', 'N/A')
                            
                            report_content += f"""
Experimental Parameters:
- Number of Trials: {trials}
- Statistical Analysis: Repeatability Study

Key Results:
- Mean Resistance: {mean_r:.4f} ± {std_r:.4f} Ω
- Coefficient of Variation: {(std_r/mean_r*100):.3f}%
- Measurement Quality: {'Excellent' if (std_r/mean_r*100) < 1 else 'Good' if (std_r/mean_r*100) < 3 else 'Acceptable'}

Analysis:
Statistical analysis confirms {'excellent' if (std_r/mean_r*100) < 1 else 'good'} 
measurement repeatability and precision.
"""
                        
                        # Add raw data if requested
                        if include_raw_data and report_format == "Comprehensive":
                            report_content += f"""

Raw Data Summary:
Number of measurements: {len(data)}
Data columns: {', '.join(data.columns)}

Sample Data (first 5 rows):
{data.head().to_string()}
"""
                
                # Add overall conclusions
                if include_conclusions:
                    completed_count = len(st.session_state.completed_sections)
                    total_count = len(lab_sections)
                    
                    report_content += f"""

OVERALL CONCLUSIONS:
===================

Laboratory Completion:
- Sections Completed: {completed_count}/{total_count} ({completed_count/total_count*100:.0f}%)
- Experiments with Data: {len(st.session_state.experiment_data)}

Key Findings:
1. Ohm's Law Verification: Successfully demonstrated linear V-I relationship for ohmic resistors
2. Non-ohmic Behavior: Confirmed temperature-dependent resistance in tungsten filaments
3. Statistical Analysis: Demonstrated measurement repeatability and uncertainty quantification
4. Circuit Analysis: Verified Kirchhoff's laws in series and parallel configurations
5. Temperature Effects: Characterized material-dependent temperature coefficients

Scientific Learning Outcomes:
✓ Experimental verification of fundamental electrical laws
✓ Understanding of measurement uncertainty and error analysis  
✓ Distinction between ohmic and non-ohmic materials
✓ Application of statistical methods to experimental data
✓ Practical understanding of electrical safety protocols

Recommendations for Future Work:
- Investigate frequency-dependent impedance effects
- Study quantum effects in low-resistance materials
- Explore breakdown phenomena in detail
- Extend temperature range studies

LABORATORY ASSESSMENT:
This virtual laboratory successfully demonstrates all key concepts of electrical
resistance and Ohm's Law while providing hands-on experience with data analysis,
error assessment, and scientific reporting. The interactive simulation environment
effectively replaces traditional hardware while enabling exploration of extreme
conditions not feasible in physical laboratories.

Report generated by: UMT Virtual Laboratory System
Institution: University of Management and Technology
Department: Physics
Laboratory: Ohm's Law Interactive Simulation
"""
                
                # Display report preview
