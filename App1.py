
import pandas as pd
import streamlit as st


# Ste the page configuration
st.set_page_config(
        page_title="ML Model Recommender",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def determine_technique_model(problem, data_type):
    # Iterate through the problem categories
    for category, problem_list in problems.items(): 
        for item in problem_list:
            if problem == item["problem"]:
                # Check if the data type is compatible with the problem
                if data_type not in item["data_types"]:
                    return f"For the problem '{problem}', there are no recommended models for the data type '{data_type}'."
                
                recommended_models = []
                technique = "Default Learning"
                
                # Check all learning techniques
                if data_type in ml_models.get(category, {}).get("supervised", {}):
                    recommended_models = ml_models[category]["supervised"].get(data_type, ml_models[category]["supervised"].get("default", []))
                    technique = "Supervised Learning"
                elif data_type in ml_models.get(category, {}).get("unsupervised", {}):
                    recommended_models = ml_models[category]["unsupervised"].get(data_type, ml_models[category]["unsupervised"].get("default", []))
                    technique = "Unsupervised Learning"
                elif data_type in ml_models.get(category, {}).get("reinforcement", {}):
                    recommended_models = ml_models[category]["reinforcement"].get(data_type, ml_models[category]["reinforcement"].get("default", []))
                    technique = "Reinforcement Learning"
                
                return f"Recommended technique: {technique}\nSuggested models to face the problem: {', '.join(recommended_models)}\nRecommended data type: {', '.join(item['data_types'])}"

    return "Error: Problem type or data type not found."



# Load the DataFrame (from previous step)
@st.cache_data
def load_data():
    data = []
    for category, problem_list in problems.items():
        for problem_dict in problem_list:
            data.append({
                "Category": category,
                "Problem": problem_dict["problem"],
                "Data Types": ", ".join(problem_dict["data_types"])
            })
    return pd.DataFrame(data)

df = load_data()

# Streamlit App
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        bbackground-color: #D32F2F;
    }
    .stSelectbox div > div {
        background-color: black;
    }
    .stButton button {
        background-color: #FFFFFF";
        color: Black;
        font-weight: black;
    }
    .stMarkdown h1, h2, h3 {
        color: #2F4F4F;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🤖Smart ML Model Recommender")
    st.markdown("""
    Select your problem type and data characteristics to get machine learning recommendations.
    """)

    # Create two columns layout
    col1, col2 = st.columns([1,1])   

    with col1:
        # Problem Selection
        st.header("Problem Configuration")
        
        # Problem type selection
        problem_list = df['Problem'].unique()
        selected_problem = st.selectbox(
            "1. Select your problem type:",
            options=problem_list,
            index=0,
            help="Choose the problem type that best matches your use case"
        )

        # Data type selection
        data_types_for_problem = df[df['Problem'] == selected_problem]['Data Types'].values[0]
        selected_data_type = st.selectbox(
            "2. Select your primary data type:",
            options=data_types_for_problem.split(","),
            index=0,
            help="Select the dominant data type in your dataset"
        )

        # Recommendation button
        if st.button("Get Recommendations"):
            
            st.header("Recommendations")
            try:
                recommendations = determine_technique_model(selected_problem, selected_data_type)
                parts = recommendations.split('\n')
                    
                with st.container():
                    st.success("Suggestions Approach")
                    for part in parts:
                        if "Recommended technique" in part:
                            st.markdown(f"**{part.split(': ')[1]}**")
                        elif "Suggested models" in part:
                            st.write("**Models:**")
                            models = part.split(': ')[1].split(', ')
                            for model in models:
                                st.markdown(f"- {model}")
                        elif "Recommended data type" in part:
                            st.write("**Compatible Data Types:**")
                            st.write(part.split(': ')[1])
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")


    # Show problem catalog in sidebar
    with st.sidebar:
        st.header("📁Problem Catalog")
        st.markdown("Explore our supported problem types and data characteristics")
        
        expander = st.expander("View Full Problem List")
        with expander:
            st.dataframe(df[['Category', 'Problem', 'Data Types']], 
                        height=200,
                        use_container_width=True)

        st.markdown("---")
        st.markdown("**Supported Data Types:**")
        for dt, desc in data_types.items():
            with st.expander(f"{dt.capitalize()}"):
                st.write(", ".join(desc))

if __name__ == "__main__":
    main()

