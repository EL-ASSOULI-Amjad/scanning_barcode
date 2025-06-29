import cv2
from pyzbar.pyzbar import decode
import streamlit as st

st.header("TESTING")
if "testing" not in st.session_state:
    st.session_state.testing = False

if "test_results" not in st.session_state:
    st.session_state.test_results = []

if st.button("Start test"):
    st.session_state.testing = True
    st.session_state.test_results = []

if st.button("Show test_results"):
    st.write(st.session_state.test_results)

with open("test_data.csv", "w") as t:
    with open("")