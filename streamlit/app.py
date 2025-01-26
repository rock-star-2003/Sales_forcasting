import streamlit as st

# def page1():
#     st.title("Page 1")
#     st.write("This is page 1.")

# def page2():
#     st.title("Page 2")
#     st.write("This is page 2.")

# # Create a sidebar for navigation
# page = st.sidebar.selectbox("Select a page", ["Page 1", "Page 2"])

# # Display the selected page
# if page == "Page 1":
#     page1()
# elif page == "Page 2":
#     page2()
st.text('Hello everyone')


create_page = st.Page("data.py", title="Data Visulization", icon=":material/add_circle:")
delete_page = st.Page("model.py", title="Future Sales", icon=":material/delete:")
home = st.Page("app.py", title="Home", icon=":material/delete:")

pg = st.navigation([home,create_page, delete_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()    