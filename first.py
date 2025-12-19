import streamlit as st
from google import genai
from datetime import datetime
import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# Configure page
st.set_page_config(
    page_title="Smart Health Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini Client
@st.cache_resource
def init_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not api_key:
        return None, api_key
    
    try:
        # Set environment variable for client
        os.environ["GEMINI_API_KEY"] = api_key
        client = genai.Client(api_key=api_key)
        return client, api_key
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None, api_key

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_assessment' not in st.session_state:
    st.session_state.current_assessment = None
if 'api_key_input' not in st.session_state:
    st.session_state.api_key_input = ""

# Main header
st.markdown('<h1 class="main-header">🏥 Smart Health Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Google Gemini 2.5 Flash 🚀")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    client, stored_key = init_gemini_client()
    
    if not stored_key:
        st.warning("⚠️ Please enter your Gemini API key")
        api_key = st.text_input("Enter Gemini API Key:", type="password", key="api_input")
        
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            try:
                client = genai.Client(api_key=api_key)
                st.success("✅ API Key configured!")
                st.rerun()
            except Exception as e:
                st.error(f"Invalid API key: {str(e)}")
    else:
        st.success("✅ Gemini API Connected")
        st.info(f"Model: gemini-2.5-flash-lite")
    
    st.markdown("---")
    st.header("📊 Quick Stats")
    st.metric("Total Assessments", len(st.session_state.history))
    if st.session_state.history:
        high_risk = sum(1 for h in st.session_state.history if h.get('risk_level') == 'High')
        st.metric("High Risk Cases", high_risk)
        
        recent = st.session_state.history[-1]
        st.metric("Last Assessment", recent['timestamp'].strftime("%d %b"))
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_assessment = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
    1. **Symptom Analysis**: Get instant AI health assessment
    2. **Report Analysis**: Upload medical documents
    3. **Dashboard**: View your health trends
    4. **Risk Predictor**: Long-term health predictions
    5. **History**: Access past assessments
    """)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Symptom Analysis", 
    "📄 Medical Report Analysis", 
    "📊 Health Dashboard",
    "🔍 Risk Predictor",
    "📜 History"
])

# Tab 1: Symptom Analysis
with tab1:
    st.header("AI-Powered Symptom Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Information")
        name = st.text_input("👤 Full Name", placeholder="John Doe")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with c3:
            weight = st.number_input("Weight (kg)", min_value=1, max_value=300, value=70)
        
        st.subheader("Current Symptoms")
        symptoms = st.text_area(
            "Describe your symptoms in detail:",
            placeholder="e.g., Persistent cough for 3 days, fever above 38°C, fatigue, body aches...",
            height=100
        )
        
        st.subheader("Quick Symptom Selector")
        common_symptoms = [
            "Fever", "Cough", "Headache", "Fatigue", "Chest Pain",
            "Shortness of Breath", "Nausea", "Dizziness", "Abdominal Pain",
            "Joint Pain", "Muscle Aches", "Sore Throat", "Runny Nose",
            "Loss of Appetite", "Weakness", "Sweating"
        ]
        
        selected_symptoms = st.multiselect(
            "Select common symptoms:",
            common_symptoms
        )
        
        duration = st.selectbox(
            "How long have you had these symptoms?",
            ["Less than 24 hours", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"]
        )
        
        medical_history = st.text_area(
            "Medical History (Optional):",
            placeholder="Previous conditions, chronic diseases, allergies, current medications...",
            height=80
        )
    
    with col2:
        st.subheader("Severity Indicators")
        pain_level = st.slider("Pain Level (0-10)", 0, 10, 0)
        fever = st.checkbox("Fever present")
        fever_temp = None
        if fever:
            fever_temp = st.number_input("Temperature (°C)", 36.0, 42.0, 37.5, step=0.1)
        
        chronic_conditions = st.multiselect(
            "Chronic Conditions:",
            ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Kidney Disease", "None"]
        )
        
        st.info("💡 Be as detailed as possible for accurate AI assessment")
        
        st.markdown("### 🎯 Assessment Score")
        if st.session_state.current_assessment:
            score = st.session_state.current_assessment['result']['risk_score']
            st.progress(score / 100)
            st.metric("Current Risk", f"{score}/100")
    
    if st.button("🔍 Analyze Health Risk with AI", type="primary", use_container_width=True):
        if not client:
            st.error("⚠️ Please configure Gemini API key in the sidebar")
        elif not symptoms and not selected_symptoms:
            st.warning("⚠️ Please describe your symptoms or select from common symptoms")
        else:
            with st.spinner("🧠 AI is analyzing your health data with Gemini 2.5 Flash..."):
                # Combine symptoms
                all_symptoms = symptoms
                if selected_symptoms:
                    all_symptoms += "\n" + ", ".join(selected_symptoms)
                
                # Create comprehensive prompt
                prompt = f"""You are an expert medical AI assistant powered by advanced deep learning. Analyze the following patient information and provide a detailed, accurate health risk assessment.

Patient Information:
- Name: {name}
- Age: {age} years
- Gender: {gender}
- Weight: {weight} kg
- Symptoms: {all_symptoms}
- Symptom Duration: {duration}
- Pain Level: {pain_level}/10
- Fever: {'Yes, ' + str(fever_temp) + '°C' if fever else 'No'}
- Medical History: {medical_history if medical_history else 'None provided'}
- Chronic Conditions: {', '.join(chronic_conditions) if chronic_conditions else 'None'}

Provide a comprehensive analysis in the following JSON format (ensure valid JSON):
{{
    "risk_level": "Low/Medium/High",
    "risk_score": <number 0-100>,
    "primary_concern": "<main health concern>",
    "possible_conditions": [
        {{"name": "<condition>", "probability": <0-100>, "severity": "Low/Medium/High", "description": "<brief description>"}},
        {{"name": "<condition>", "probability": <0-100>, "severity": "Low/Medium/High", "description": "<brief description>"}},
        {{"name": "<condition>", "probability": <0-100>, "severity": "Low/Medium/High", "description": "<brief description>"}}
    ],
    "immediate_actions": ["action1", "action2", "action3"],
    "recommendations": ["rec1", "rec2", "rec3", "rec4"],
    "warning_signs": ["sign1", "sign2", "sign3"],
    "when_to_seek_help": "<clear guidance on when to visit doctor/ER>",
    "lifestyle_advice": ["advice1", "advice2", "advice3"],
    "follow_up": "<follow up guidance with timeline>",
    "detailed_analysis": "<comprehensive 2-3 paragraph analysis of the condition>",
    "prevention_tips": ["tip1", "tip2", "tip3"]
}}

Be thorough, accurate, and provide actionable medical guidance. Consider all symptoms, risk factors, and patient history. Prioritize patient safety."""

                try:
                    # Use new Gemini API
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=prompt
                    )
                    
                    result_text = response.text
                    
                    # Extract JSON
                    if "```json" in result_text:
                        result_text = result_text.split("```json")[1].split("```")[0]
                    elif "```" in result_text:
                        result_text = result_text.split("```")[1].split("```")[0]
                    
                    result = json.loads(result_text.strip())
                    
                    # Store in session
                    assessment = {
                        'timestamp': datetime.now(),
                        'patient_name': name,
                        'age': age,
                        'gender': gender,
                        'symptoms': all_symptoms,
                        'result': result,
                        'risk_level': result['risk_level']
                    }
                    st.session_state.current_assessment = assessment
                    st.session_state.history.append(assessment)
                    
                    # Display results
                    st.success("✅ AI Analysis Complete!")
                    
                    # Risk Level Card
                    risk_class = f"risk-{result['risk_level'].lower()}"
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <h2>🎯 Risk Level: {result['risk_level']}</h2>
                        <h3>📊 Risk Score: {result['risk_score']}/100</h3>
                        <p><strong>🔍 Primary Concern:</strong> {result['primary_concern']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['risk_score'],
                        title={'text': "Health Risk Score", 'font': {'size': 24}},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 33], 'color': '#e8f5e9'},
                                {'range': [33, 66], 'color': '#fff3e0'},
                                {'range': [66, 100], 'color': '#ffebee'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': result['risk_score']
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed Analysis
                    st.subheader("📋 Detailed Medical Analysis")
                    st.markdown(result['detailed_analysis'])
                    
                    # Possible Conditions
                    st.subheader("🔬 Possible Conditions Analysis")
                    for i, condition in enumerate(result['possible_conditions'], 1):
                        with st.expander(f"{i}. {condition['name']} - {condition['probability']}% probability", expanded=(i==1)):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**Description:** {condition.get('description', 'N/A')}")
                                st.progress(condition['probability'] / 100)
                            with col2:
                                severity_color = "🔴" if condition['severity'] == 'High' else "🟡" if condition['severity'] == 'Medium' else "🟢"
                                st.metric("Severity", f"{severity_color} {condition['severity']}")
                                st.metric("Probability", f"{condition['probability']}%")
                    
                    # Create visualization
                    df = pd.DataFrame(result['possible_conditions'])
                    fig2 = px.bar(df, x='name', y='probability', 
                                 color='severity',
                                 color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'},
                                 title='Condition Probability Analysis')
                    fig2.update_layout(xaxis_title="Condition", yaxis_title="Probability (%)")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Immediate Actions
                    st.subheader("⚡ Immediate Actions Required")
                    for action in result['immediate_actions']:
                        st.warning(f"🚨 {action}")
                    
                    # Two column layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("💊 Medical Recommendations")
                        for rec in result['recommendations']:
                            st.info(f"✓ {rec}")
                        
                        st.subheader("🌟 Lifestyle & Prevention")
                        for advice in result['lifestyle_advice']:
                            st.success(f"• {advice}")
                    
                    with col2:
                        st.subheader("⚠️ Warning Signs to Watch")
                        for sign in result['warning_signs']:
                            st.error(f"⚠️ {sign}")
                        
                        st.subheader("🎯 Prevention Tips")
                        for tip in result.get('prevention_tips', []):
                            st.info(f"💡 {tip}")
                    
                    # When to Seek Help
                    st.subheader("🚨 When to Seek Medical Help")
                    st.error(f"**Important:** {result['when_to_seek_help']}")
                    
                    # Follow-up
                    st.info(f"📅 **Follow-up Recommendation:** {result['follow_up']}")
                    
                    # Download Report
                    st.markdown("---")
                    report_data = json.dumps({
                        'patient_info': {
                            'name': name,
                            'age': age,
                            'gender': gender,
                            'weight': weight
                        },
                        'assessment': result,
                        'timestamp': assessment['timestamp'].isoformat()
                    }, indent=2)
                    
                    st.download_button(
                        label="📥 Download Full Report (JSON)",
                        data=report_data,
                        file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing AI response: {str(e)}")
                    st.write("Raw AI response:", response.text if 'response' in locals() else "No response")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    if 'response' in locals():
                        st.write("Response received:", response.text[:500])

# Tab 2: Medical Report Analysis
with tab2:
    st.header("📄 AI Medical Report & Document Analysis")
    
    st.info("📤 Upload medical reports, lab results, prescriptions, or health documents for instant AI analysis using Gemini 2.5")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Medical Document",
            type=['txt', 'pdf', 'jpg', 'jpeg', 'png'],
            help="Supported: PDF, Images (JPG, PNG), Text files"
        )
        
        report_context = st.text_area(
            "Additional Context (Optional):",
            placeholder="Provide context: test date, symptoms during test, medications, etc...",
            height=100
        )
        
        patient_info_report = st.text_input("Patient Name (Optional)", placeholder="For record keeping")
    
    with col2:
        st.subheader("Analysis Options")
        analysis_type = st.radio(
            "Analysis Focus:",
            ["Comprehensive Analysis", "Lab Results", "Prescription Review", "Imaging Report", "Blood Test Analysis"]
        )
        
        include_trends = st.checkbox("Include trend analysis", value=True)
        detailed_explanation = st.checkbox("Detailed explanations", value=True)
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        # File preview
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)
        
        if st.button("📊 Analyze Document with AI", type="primary", use_container_width=True):
            if not client:
                st.error("⚠️ Please configure Gemini API key")
            else:
                with st.spinner("🔍 AI is analyzing your document with Gemini 2.5..."):
                    try:
                        # Read file content
                        if uploaded_file.type == "text/plain":
                            content = uploaded_file.read().decode()
                        else:
                            content = f"Medical document uploaded: {uploaded_file.name} (Type: {uploaded_file.type})"
                        
                        prompt = f"""Analyze this medical document as an expert medical AI system. Provide detailed, accurate insights.

Document Type: {analysis_type}
File Name: {uploaded_file.name}
Content/Description: {content}
Additional Context: {report_context if report_context else 'None'}
Patient: {patient_info_report if patient_info_report else 'Not specified'}

Provide comprehensive analysis in JSON format:
{{
    "document_type": "{analysis_type}",
    "document_summary": "<2-3 sentence overview>",
    "key_findings": ["finding1", "finding2", "finding3"],
    "abnormal_values": [
        {{"parameter": "<name>", "value": "<value>", "normal_range": "<range>", "severity": "Low/Medium/High", "explanation": "<why abnormal>"}}
    ],
    "normal_values": [
        {{"parameter": "<name>", "value": "<value>", "status": "Within normal range"}}
    ],
    "diagnosis_suggestions": ["suggestion1", "suggestion2"],
    "risk_indicators": ["indicator1", "indicator2"],
    "recommended_tests": ["test1", "test2"],
    "specialist_referral": "<recommendation if needed>",
    "lifestyle_modifications": ["mod1", "mod2"],
    "medication_review": "<review of medications if applicable>",
    "follow_up_timeline": "<specific timeline>",
    "clinical_significance": "<what do these results mean clinically>",
    "patient_action_items": ["action1", "action2"],
    "detailed_interpretation": "<comprehensive 2-3 paragraph interpretation>"
}}

Be thorough and provide medically sound interpretations."""

                        response = client.models.generate_content(
                            model='gemini-2.5-flash-lite',
                            contents=prompt
                        )
                        
                        result_text = response.text
                        
                        if "```json" in result_text:
                            result_text = result_text.split("```json")[1].split("```")[0]
                        elif "```" in result_text:
                            result_text = result_text.split("```")[1].split("```")[0]
                        
                        result = json.loads(result_text.strip())
                        
                        # Display Results
                        st.success("✅ Document Analysis Complete!")
                        
                        # Summary
                        st.markdown(f"### 📋 Document Summary")
                        st.info(result['document_summary'])
                        
                        st.markdown(f"### 🔍 Clinical Significance")
                        st.write(result['clinical_significance'])
                        
                        # Key Findings
                        st.subheader("🎯 Key Findings")
                        for i, finding in enumerate(result['key_findings'], 1):
                            st.success(f"{i}. {finding}")
                        
                        # Abnormal vs Normal Values
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if result.get('abnormal_values'):
                                st.subheader("⚠️ Abnormal Values")
                                for val in result['abnormal_values']:
                                    severity_emoji = "🔴" if val['severity'] == 'High' else "🟡" if val['severity'] == 'Medium' else "🟢"
                                    with st.expander(f"{severity_emoji} {val['parameter']} - {val['severity']} Severity"):
                                        st.write(f"**Value:** {val['value']}")
                                        st.write(f"**Normal Range:** {val['normal_range']}")
                                        st.write(f"**Explanation:** {val['explanation']}")
                        
                        with col2:
                            if result.get('normal_values'):
                                st.subheader("✅ Normal Values")
                                for val in result['normal_values']:
                                    st.info(f"✓ **{val['parameter']}**: {val['value']} - {val['status']}")
                        
                        # Diagnosis and Recommendations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("💡 Diagnosis Suggestions")
                            for diag in result['diagnosis_suggestions']:
                                st.warning(f"• {diag}")
                            
                            st.subheader("🧪 Recommended Tests")
                            for test in result['recommended_tests']:
                                st.info(f"• {test}")
                        
                        with col2:
                            st.subheader("🚨 Risk Indicators")
                            for risk in result['risk_indicators']:
                                st.error(f"⚠️ {risk}")
                            
                            st.subheader("📋 Patient Action Items")
                            for action in result['patient_action_items']:
                                st.success(f"✓ {action}")
                        
                        # Detailed sections
                        with st.expander("📖 Detailed Medical Interpretation", expanded=False):
                            st.write(result['detailed_interpretation'])
                        
                        with st.expander("👨‍⚕️ Specialist Referral"):
                            st.write(result['specialist_referral'])
                        
                        with st.expander("💊 Medication Review"):
                            st.write(result['medication_review'])
                        
                        with st.expander("🌟 Lifestyle Modifications"):
                            for mod in result['lifestyle_modifications']:
                                st.write(f"• {mod}")
                        
                        st.info(f"📅 **Follow-up Timeline:** {result['follow_up_timeline']}")
                        
                        # Download
                        report_json = json.dumps(result, indent=2)
                        st.download_button(
                            label="📥 Download Analysis Report",
                            data=report_json,
                            file_name=f"report_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing response: {str(e)}")
                        with st.expander("Show raw response"):
                            st.write(response.text if 'response' in locals() else "No response")
                    except Exception as e:
                        st.error(f"Error analyzing document: {str(e)}")

# Tab 3: Health Dashboard
with tab3:
    st.header("📊 Personal Health Dashboard")
    
    if st.session_state.current_assessment:
        assessment = st.session_state.current_assessment
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_emoji = "🔴" if assessment['risk_level'] == 'High' else "🟡" if assessment['risk_level'] == 'Medium' else "🟢"
            col1.metric(
                "Current Risk Level", 
                f"{risk_emoji} {assessment['risk_level']}", 
                delta="Monitor" if assessment['risk_level'] != "Low" else "Good"
            )
        
        with col2:
            col2.metric(
                "Risk Score", 
                f"{assessment['result']['risk_score']}/100",
                delta=f"{assessment['result']['risk_score'] - 50}" if len(st.session_state.history) > 1 else None
            )
        
        with col3:
            col3.metric(
                "Last Assessment", 
                assessment['timestamp'].strftime("%d %b %Y")
            )
        
        with col4:
            col4.metric(
                "Total Assessments", 
                len(st.session_state.history)
            )
        
        st.markdown("---")
        
        # Risk Trend Analysis
        if len(st.session_state.history) > 1:
            st.subheader("📈 Health Risk Trend Analysis")
            
            # Prepare data
            dates = [h['timestamp'] for h in st.session_state.history]
            scores = [h['result']['risk_score'] for h in st.session_state.history]
            risk_levels = [h['risk_level'] for h in st.session_state.history]
            
            # Create DataFrame
            df_trend = pd.DataFrame({
                'Date': dates,
                'Risk Score': scores,
                'Risk Level': risk_levels
            })
            
            # Line chart
            fig = px.line(df_trend, x='Date', y='Risk Score', 
                         title='Health Risk Score Over Time',
                         markers=True)
            
            fig.add_hline(y=66, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Threshold (66)")
            fig.add_hline(y=33, line_dash="dash", line_color="orange",
                         annotation_text="Medium Risk Threshold (33)")
            
            fig.update_layout(
                yaxis_range=[0, 100],
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level distribution
            col1, col2 = st.columns(2)
            
            with col1:
                risk_counts = pd.Series(risk_levels).value_counts()
                fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                             title='Risk Level Distribution',
                             color=risk_counts.index,
                             color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Statistics
                st.subheader("📊 Statistics")
                st.metric("Average Risk Score", f"{sum(scores)/len(scores):.1f}")
                st.metric("Highest Score", f"{max(scores)}")
                st.metric("Lowest Score", f"{min(scores)}")
                st.metric("Score Trend", "Improving" if scores[-1] < scores[0] else "Monitor")
        
        st.markdown("---")
        
        # Current Health Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Current Primary Concern")
            st.info(assessment['result']['primary_concern'])
            
            st.subheader("💊 Active Recommendations")
            for rec in assessment['result']['recommendations'][:3]:
                st.success(f"✓ {rec}")
        
        with col2:
            st.subheader("⚠️ Active Warning Signs")
            for sign in assessment['result']['warning_signs'][:3]:
                st.error(f"⚠️ {sign}")
            
            st.subheader("📅 Follow-up Status")
            st.write(assessment['result']['follow_up'])
        
        # Condition Analysis
        st.markdown("---")
        st.subheader("🔬 Top Conditions Under Monitoring")
        
        conditions_df = pd.DataFrame(assessment['result']['possible_conditions'])
        conditions_df = conditions_df.sort_values('probability', ascending=False)
        
        fig3 = px.bar(conditions_df, 
                     x='probability', 
                     y='name',
                     orientation='h',
                     color='severity',
                     color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'},
                     title='Current Health Conditions Probability',
                     labels={'probability': 'Probability (%)', 'name': 'Condition'})
        
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.info("📊 Complete a symptom analysis in Tab 1 to view your personalized health dashboard")
        st.image("https://via.placeholder.com/800x400/e3f2fd/1976d2?text=Complete+Assessment+to+View+Dashboard", 
                use_container_width=True)

# Tab 4: Risk Predictor
with tab4:
    st.header("🔍 Long-Term Health Risk Predictor")
    st.markdown("### Predict future health risks based on lifestyle and family history")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Personal Information")
        pred_name = st.text_input("Name", placeholder="John Doe", key="pred_name")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            pred_age = st.number_input("Age", 1, 120, 30, key="pred_age")
        with c2:
            pred_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="pred_gender")
        with c3:
            pred_height = st.number_input("Height (cm)", 100, 250, 170, key="pred_height")
        
        st.subheader("Lifestyle Factors")
        
        smoking = st.selectbox("Smoking Status", 
                              ["Never", "Former (quit >5 years)", "Former (quit <5 years)", 
                               "Current (<10/day)", "Current (10-20/day)", "Current (>20/day)"])
        
        alcohol = st.selectbox("Alcohol Consumption",
                              ["None", "Occasional (1-2/week)", "Moderate (3-7/week)", 
                               "Heavy (8-14/week)", "Very Heavy (>14/week)"])
        
        exercise = st.selectbox("Exercise Frequency",
                               ["Sedentary", "Light (1-2 days/week)", "Moderate (3-4 days/week)",
                                "Active (5-6 days/week)", "Very Active (daily)"])
        
        diet = st.selectbox("Diet Quality",
                           ["Poor (mostly processed)", "Fair (some healthy foods)", 
                            "Good (balanced)", "Excellent (mostly whole foods)"])
        
        sleep = st.slider("Average Sleep Hours/Night", 3, 12, 7)
        stress = st.slider("Stress Level (0-10)", 0, 10, 5)
        
        st.subheader("Family History")
        family_conditions = st.multiselect(
            "Family History of:",
            ["Heart Disease", "Diabetes", "Cancer", "Stroke", "Hypertension",
             "Alzheimer's", "Kidney Disease", "Liver Disease", "Asthma", "None"]
        )
        
        st.subheader("Current Health Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            bmi = st.number_input("BMI", 10.0, 60.0, 22.0, step=0.1)
        with c2:
            blood_pressure = st.text_input("Blood Pressure", placeholder="120/80")
        with c3:
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 180)
    
    with col2:
        st.subheader("Risk Timeline")
        prediction_years = st.slider("Predict for next X years", 5, 30, 10)
        
        st.info("💡 More accurate results with complete information")
        
        st.markdown("### 📊 Risk Categories")
        st.markdown("""
        - **Cardiovascular**
        - **Diabetes**
        - **Cancer**
        - **Respiratory**
        - **Mental Health**
        """)
    
    if st.button("🔮 Predict Long-Term Health Risks", type="primary", use_container_width=True):
        if not client:
            st.error("⚠️ Please configure Gemini API key")
        else:
            with st.spinner(f"🧠 AI is analyzing your long-term health risks for next {prediction_years} years..."):
                prompt = f"""You are an advanced predictive health AI. Analyze the following comprehensive patient data and predict long-term health risks.

Patient Profile:
- Name: {pred_name}
- Age: {pred_age}
- Gender: {pred_gender}
- Height: {pred_height} cm
- BMI: {bmi}
- Blood Pressure: {blood_pressure}
- Cholesterol: {cholesterol} mg/dL

Lifestyle Factors:
- Smoking: {smoking}
- Alcohol: {alcohol}
- Exercise: {exercise}
- Diet: {diet}
- Sleep: {sleep} hours/night
- Stress Level: {stress}/10

Family History: {', '.join(family_conditions) if family_conditions else 'None'}

Prediction Timeline: {prediction_years} years

Provide detailed risk predictions in JSON format:
{{
    "overall_health_score": <0-100>,
    "life_expectancy_impact": "<positive/negative impact description>",
    "risk_predictions": [
        {{
            "category": "<risk category>",
            "current_risk": <0-100>,
            "risk_5_years": <0-100>,
            "risk_10_years": <0-100>,
            "risk_long_term": <0-100>,
            "key_factors": ["factor1", "factor2"],
            "prevention_strategy": "<strategy>"
        }}
    ],
    "high_priority_risks": [
        {{"risk": "<risk>", "timeframe": "<when>", "severity": "Low/Medium/High", "preventable": true/false}}
    ],
    "lifestyle_improvements": [
        {{"area": "<area>", "current_impact": "<impact>", "recommendation": "<specific action>", "benefit": "<expected benefit>"}}
    ],
    "protective_factors": ["factor1", "factor2"],
    "risk_factors": ["factor1", "factor2"],
    "medical_screenings": [
        {{"test": "<test name>", "frequency": "<how often>", "reason": "<why needed>", "age_start": <age>}}
    ],
    "personalized_action_plan": [
        {{"priority": "High/Medium/Low", "action": "<specific action>", "timeline": "<when to start>", "expected_outcome": "<benefit>"}}
    ],
    "yearly_risk_trajectory": [
        {{"year": <year>, "overall_risk": <0-100>, "key_concern": "<main concern>"}}
    ],
    "detailed_analysis": "<comprehensive 3-4 paragraph analysis>"
}}

Provide evidence-based predictions considering all factors."""

                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=prompt
                    )
                    
                    result_text = response.text
                    
                    if "```json" in result_text:
                        result_text = result_text.split("```json")[1].split("```")[0]
                    elif "```" in result_text:
                        result_text = result_text.split("```")[1].split("```")[0]
                    
                    result = json.loads(result_text.strip())
                    
                    # Display Results
                    st.success("✅ Long-Term Risk Prediction Complete!")
                    
                    # Overall Score
                    st.metric("Overall Health Score", f"{result['overall_health_score']}/100",
                             help="Higher is better")
                    
                    st.info(f"**Life Expectancy Impact:** {result['life_expectancy_impact']}")
                    
                    # Detailed Analysis
                    with st.expander("📖 Detailed Health Analysis", expanded=True):
                        st.write(result['detailed_analysis'])
                    
                    # Risk Trajectory Visualization
                    st.subheader("📈 Risk Trajectory Over Time")
                    
                    trajectory_df = pd.DataFrame(result['yearly_risk_trajectory'])
                    fig = px.line(trajectory_df, x='year', y='overall_risk',
                                 title=f'Predicted Health Risk Over Next {prediction_years} Years',
                                 markers=True)
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Overall Risk Score",
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk Categories
                    st.subheader("🎯 Risk Categories Analysis")
                    
                    for risk in result['risk_predictions']:
                        with st.expander(f"📊 {risk['category']}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Risk", f"{risk['current_risk']}%")
                            with col2:
                                st.metric("5-Year Risk", f"{risk['risk_5_years']}%")
                            with col3:
                                st.metric("Long-Term Risk", f"{risk['risk_long_term']}%")
                            
                            st.write("**Key Risk Factors:**")
                            for factor in risk['key_factors']:
                                st.write(f"• {factor}")
                            
                            st.info(f"**Prevention Strategy:** {risk['prevention_strategy']}")
                    
                    # Risk comparison visualization
                    risk_categories = [r['category'] for r in result['risk_predictions']]
                    current_risks = [r['current_risk'] for r in result['risk_predictions']]
                    long_term_risks = [r['risk_long_term'] for r in result['risk_predictions']]
                    
                    fig2 = go.Figure(data=[
                        go.Bar(name='Current Risk', x=risk_categories, y=current_risks),
                        go.Bar(name='Long-Term Risk', x=risk_categories, y=long_term_risks)
                    ])
                    fig2.update_layout(barmode='group', title='Current vs Long-Term Risk Comparison')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # High Priority Risks
                    st.subheader("🚨 High Priority Health Risks")
                    for priority in result['high_priority_risks']:
                        severity_color = "error" if priority['severity'] == 'High' else "warning" if priority['severity'] == 'Medium' else "info"
                        preventable_text = "✅ Preventable" if priority['preventable'] else "⚠️ Manage Risk"
                        
                        if severity_color == "error":
                            st.error(f"**{priority['risk']}** - {priority['timeframe']} | {preventable_text}")
                        elif severity_color == "warning":
                            st.warning(f"**{priority['risk']}** - {priority['timeframe']} | {preventable_text}")
                        else:
                            st.info(f"**{priority['risk']}** - {priority['timeframe']} | {preventable_text}")
                    
                    # Lifestyle Improvements
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("💪 Lifestyle Improvements")
                        for improvement in result['lifestyle_improvements']:
                            with st.expander(f"🎯 {improvement['area']}"):
                                st.write(f"**Current Impact:** {improvement['current_impact']}")
                                st.write(f"**Recommendation:** {improvement['recommendation']}")
                                st.success(f"**Expected Benefit:** {improvement['benefit']}")
                    
                    with col2:
                        st.subheader("🛡️ Protective Factors")
                        for factor in result['protective_factors']:
                            st.success(f"✓ {factor}")
                        
                        st.subheader("⚠️ Risk Factors")
                        for factor in result['risk_factors']:
                            st.error(f"✗ {factor}")
                    
                    # Medical Screenings
                    st.subheader("🔬 Recommended Medical Screenings")
                    screening_df = pd.DataFrame(result['medical_screenings'])
                    st.dataframe(screening_df, use_container_width=True, hide_index=True)
                    
                    # Personalized Action Plan
                    st.subheader("📋 Personalized Action Plan")
                    
                    high_priority = [a for a in result['personalized_action_plan'] if a['priority'] == 'High']
                    medium_priority = [a for a in result['personalized_action_plan'] if a['priority'] == 'Medium']
                    low_priority = [a for a in result['personalized_action_plan'] if a['priority'] == 'Low']
                    
                    if high_priority:
                        st.markdown("**🔴 High Priority Actions:**")
                        for action in high_priority:
                            st.error(f"**{action['action']}**\n- Timeline: {action['timeline']}\n- Expected Outcome: {action['expected_outcome']}")
                    
                    if medium_priority:
                        st.markdown("**🟡 Medium Priority Actions:**")
                        for action in medium_priority:
                            st.warning(f"**{action['action']}**\n- Timeline: {action['timeline']}\n- Expected Outcome: {action['expected_outcome']}")
                    
                    if low_priority:
                        st.markdown("**🟢 Low Priority Actions:**")
                        for action in low_priority:
                            st.info(f"**{action['action']}**\n- Timeline: {action['timeline']}\n- Expected Outcome: {action['expected_outcome']}")
                    
                    # Download Report
                    st.markdown("---")
                    prediction_report = json.dumps({
                        'patient_info': {
                            'name': pred_name,
                            'age': pred_age,
                            'gender': pred_gender
                        },
                        'prediction': result,
                        'timestamp': datetime.now().isoformat()
                    }, indent=2)
                    
                    st.download_button(
                        label="📥 Download Prediction Report",
                        data=prediction_report,
                        file_name=f"risk_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing AI response: {str(e)}")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

# Tab 5: History
with tab5:
    st.header("📜 Assessment History")
    
    if not st.session_state.history:
        st.info("📊 No assessments yet. Complete a symptom analysis in Tab 1 to build your history.")
    else:
        st.success(f"Total Assessments: {len(st.session_state.history)}")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_risk = st.multiselect(
                "Filter by Risk Level",
                ["Low", "Medium", "High"],
                default=["Low", "Medium", "High"]
            )
        
        with col2:
            sort_order = st.selectbox(
                "Sort by",
                ["Newest First", "Oldest First", "Highest Risk", "Lowest Risk"]
            )
        
        with col3:
            search_term = st.text_input("🔍 Search symptoms", placeholder="e.g., fever, cough")
        
        # Sort and filter
        filtered_history = [h for h in st.session_state.history if h['risk_level'] in filter_risk]
        
        if search_term:
            filtered_history = [h for h in filtered_history 
                              if search_term.lower() in h['symptoms'].lower()]
        
        if sort_order == "Newest First":
            filtered_history = sorted(filtered_history, key=lambda x: x['timestamp'], reverse=True)
        elif sort_order == "Oldest First":
            filtered_history = sorted(filtered_history, key=lambda x: x['timestamp'])
        elif sort_order == "Highest Risk":
            filtered_history = sorted(filtered_history, key=lambda x: x['result']['risk_score'], reverse=True)
        else:
            filtered_history = sorted(filtered_history, key=lambda x: x['result']['risk_score'])
        
        st.markdown("---")
        
        # Display history
        for i, assessment in enumerate(filtered_history, 1):
            with st.expander(
                f"#{i} - {assessment['patient_name']} | {assessment['timestamp'].strftime('%d %b %Y %H:%M')} | "
                f"Risk: {assessment['risk_level']} ({assessment['result']['risk_score']}/100)",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Patient:** {assessment['patient_name']}")
                    st.write(f"**Age:** {assessment['age']} | **Gender:** {assessment['gender']}")
                    st.write(f"**Symptoms:** {assessment['symptoms'][:200]}...")
                    st.write(f"**Primary Concern:** {assessment['result']['primary_concern']}")
                
                with col2:
                    risk_class = f"risk-{assessment['risk_level'].lower()}"
                    st.markdown(f"""
                    <div class="{risk_class}" style="padding: 10px; border-radius: 5px;">
                        <h4>Risk: {assessment['risk_level']}</h4>
                        <h3>Score: {assessment['result']['risk_score']}/100</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick view of conditions
                st.write("**Top Conditions:**")
                for condition in assessment['result']['possible_conditions'][:3]:
                    st.write(f"• {condition['name']} - {condition['probability']}%")
                
                # Actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📥 Download", key=f"download_{i}"):
                        report_json = json.dumps(assessment, default=str, indent=2)
                        st.download_button(
                            label="Download Report",
                            data=report_json,
                            file_name=f"assessment_{assessment['timestamp'].strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key=f"download_btn_{i}"
                        )
                
                with col2:
                    if st.button("🔄 Reassess", key=f"reassess_{i}"):
                        st.info("Go to Tab 1 to create a new assessment")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        st.session_state.history.remove(assessment)
                        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Smart Health Risk Prediction System</strong> | Powered by Google Gemini 2.5 Flash</p>
    <p>⚠️ <em>Disclaimer: This AI system provides health information and risk assessments for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare 
    providers for medical decisions.</em></p>
    <p>🔒 Your health data is processed securely and not stored permanently.</p>
</div>
""", unsafe_allow_html=True)