import streamlit as stm 
from streamlit_extras.let_it_rain import rain 
  
  
stm.set_page_config(page_title="This is a Simple Streamlit WebApp") 
stm.title("This is the Home Page Geeks.") 
stm.text("Geeks Home Page") 
  
  
# Raining Emoji 
  
rain( 
    emoji="😘", 
    font_size=40,  # the size of emoji 
    falling_speed=3,  # speed of raining 
    animation_length="infinite",  # for how much time the animation will happen 
) 