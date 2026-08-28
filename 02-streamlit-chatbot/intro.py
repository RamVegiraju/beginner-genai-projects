"""
Streamlit basics, before the chatbot.

Streamlit turns a Python script into a web page. No HTML, no JavaScript, no
callbacks to register -- you write top to bottom and each st.* call appends
something to the page.

One rule explains most of the surprises:

    ON EVERY INTERACTION, STREAMLIT RE-RUNS THIS ENTIRE FILE FROM THE TOP.

Move a slider and the script runs again from line 1. That is why an ordinary
Python variable cannot remember anything between clicks, and why
st.session_state exists. Section 3 below shows both.

Nothing here calls a model, so this file needs no Databricks credentials.

Run:  streamlit run intro.py
"""

import streamlit as st

# --- 1. Text ----------------------------------------------------------------
# Headings, largest to smallest, then the everyday ways to write text.

st.title("Streamlit basics")

st.header("1. Putting things on the page")
st.subheader("Calls append in the order you write them")

st.write("st.write() is the workhorse -- it takes strings, numbers, even dataframes.")
st.markdown("st.markdown() renders **bold**, *italic*, `code`, and [links](https://streamlit.io).")
st.caption("st.caption() is the small grey text, for asides.")
st.code('import streamlit as st\nst.title("Hello")', language="python")

st.divider()

# --- 2. Widgets -------------------------------------------------------------
# A widget RETURNS its current value. There is nothing to wire up: read the
# return value the way you would read any function result.

st.header("2. Widgets return their value")

name = st.text_input("Your name", placeholder="Ada")
excitement = st.slider("How excited are you?", min_value=0, max_value=10, value=7)
shout = st.checkbox("Shout it")

greeting = f"Hello, {name or 'stranger'}! Excitement: {excitement}/10"
st.write(greeting.upper() if shout else greeting)

st.divider()

# --- 3. Reruns and state ----------------------------------------------------
# The part that catches everyone out.

st.header("3. The script re-runs on every interaction")

# `clicks` is a plain local variable, so it is created again on every rerun.
# Press the button repeatedly: it shows 1, never 2. Touch any other widget and
# it drops back to 0, because that reruns the script too.
clicks = 0
if st.button("Add one (plain variable)"):
    clicks += 1
st.write(f"Plain variable: **{clicks}**")

# st.session_state is a dict that survives reruns, for as long as the browser
# tab stays open. Initialise a key once, then update it.
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Add one (session_state)"):
    st.session_state.count += 1
st.write(f"session_state: **{st.session_state.count}**")

st.info("This is exactly why app.py keeps its conversation in st.session_state.")

st.divider()

# --- 4. Layout --------------------------------------------------------------

st.header("4. Layout")

# columns() hands back one container per column. Write into them like st.
left, right = st.columns(2)
left.metric("Excitement", f"{excitement}/10")
right.metric("Button presses", st.session_state.count)

# Anything inside `with st.sidebar:` renders in the left panel instead.
with st.sidebar:
    st.header("The sidebar")
    st.write("Everything inside `with st.sidebar:` lands here. Sample 4 uses it for memory.")

st.divider()

# --- 5. Next ----------------------------------------------------------------

st.header("5. On to the chatbot")
st.write(
    "`app.py` in this folder adds just two chat-specific calls on top of all "
    "the above: `st.chat_input()` for the box pinned to the bottom, and "
    "`st.chat_message()` for the speech bubbles."
)
st.code(
    'prompt = st.chat_input("Ask something")\nwith st.chat_message("user"):\n    st.markdown(prompt)',
    language="python",
)
