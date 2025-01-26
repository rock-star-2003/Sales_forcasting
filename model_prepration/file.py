import streamlit as st

def create_page():
    st.title("Create Page")
    st.write("This is the create page.")

def delete_page():
    st.title("Delete Page")
    st.write("This is the delete page.")

# Create a sidebar for navigation
page = st.sidebar.selectbox("Select a page", ["Create", "Delete"])

# Display the selected page
if page == "Create":
    create_page()
elif page == "Delete":
    delete_page()