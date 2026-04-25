import streamlit as st
import pandas as pd
import os
from streamlit_webrtc import webrtc_streamer
from processor import ProctorProcessor 

st.set_page_config(page_title="AI Proctor System", layout="wide")

# Sidebar , supports two views: Examinee and Proctor
st.sidebar.title("System Controls")
st.sidebar.markdown("Use this to simulate the two different sides of the software.")
role = st.sidebar.radio("Select View:", ["Student (Examinee)", "Proctor Dashboard"])
st.title("Real-Time AI Proctor")

# Student View
if role == "Student (Examinee)":
    st.markdown("Exam in Progress")
    st.info("Your environment is being securely monitored. Please keep your eyes on the screen and do not use unauthorized materials.")
    
    # We display the streamer centered and alone
    webrtc_streamer(
        key="proctor", 
        video_processor_factory=ProctorProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False}
    )

# Proctor View
elif role == "Proctor Dashboard":
    st.markdown("Live Monitoring Center")
    
    # To show live feed and live log, 
    # st.columns: Insert containers laid out as side-by-side columns.
    # 2, 1.5 represent the width of both col, CAN BE ADJUSTED ACCORDINGLY.
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        st.write("**Live Student Feed**")
        webrtc_streamer(
            key="proctor", 
            video_processor_factory=ProctorProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False}
        )

    with col2:
        # The auto-refreshing fragment handles both the metrics and the logs
        @st.fragment(run_every="2s")
        def proctor_dashboard():
            st.write("**Session Analytics**")
            
            if os.path.exists("session_logs.csv"):
                df = pd.read_csv("session_logs.csv")
                
                # Calculate dashboard numbers
                total_flags = len(df)
                critical_flags = len(df[df["Severity"] == "Critical"])
                high_flags = len(df[df["Severity"] == "High"])
                
                # Display metrics in a clean row
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Events", total_flags)
                m2.metric("Critical Alerts", critical_flags)
                m3.metric("High Severity", high_flags)
                
                # Display a horizontal rule.
                st.divider()
                
                # Display the live table, with the most recent transition on the top (ascending= False)
                st.write("**Live Incident Log**")
                df_sorted = df.sort_values(by="Timestamp", ascending=False)
                st.dataframe(df_sorted, use_container_width=True, hide_index=True)
            else:
                st.info("No violations logged yet.")
                
        proctor_dashboard()