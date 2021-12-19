import os
import sys

import logging
import pandas as pd
from json import JSONDecodeError
from pathlib import Path
import streamlit as st
from annotated_text import annotation
from markdown import markdown
import pickle

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import haystack_is_ready, query, send_feedback, upload_doc, haystack_version, get_backlink, reset_datastore, index_datastore


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "How do you collect my data?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "Website")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", 10))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", 3))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", Path(__file__).parent / "random_questions.csv")
FAVICON = os.getenv("FAVICON_FILE", Path(__file__).parent / "favicon.png")

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))




def main():
    reload_mode = 0
    resp_bool = reset_datastore()
    st.set_page_config(page_title='Privacy Policy QA', page_icon=FAVICON)
    # if debug:
    #
    #     if resp_bool:
    #         st.info('Deleted all documents')
    #     else:
    #         st.info('Failed to delete all documents')
    try:
        with open("/home/user/appdata/segments.pkl", "rb") as f:
            segments_dict = pickle.load(f)
        with open("/home/user/appdata/domain.pkl", "rb") as f:
            domain = pickle.load(f)
        with open("/home/user/appdata/url.pkl", "rb") as f:
            url = pickle.load(f)
    except:
        reload_mode = 1



    index_datastore(segments_dict)


    # Persistent state
    state = SessionState.get(
        question=DEFAULT_QUESTION_AT_STARTUP,
        answer=DEFAULT_ANSWER_AT_STARTUP,
        results=None,
        raw_json=None,
        random_question_requested=False
    )

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        state.answer = None
        state.results = None
        state.raw_json = None

    # Title
    st.write("# Privacy Policy Question Answering")
    if reload_mode:
        st.error("Privacy Policy state information not captured properly. Please reload the page!")
    if len(domain) and len(url):
        st.markdown("""
        <h3 style='text-align:center;padding: 0 0 1rem;'>Ask Me Anything about <a href="{1}">{0}</a>!</h3>
        
        Ask any question related to privacy practices carried out by {0} that you'd like to know about! Try clicking on Random question to see a sample query.
        
        *Note: do not use keywords, but full-fledged questions.* The underlying models are not optimized to deal with keyword queries and might misunderstand you.
        """.format(domain.capitalize(), url), unsafe_allow_html=True)
    else:
        st.markdown("""
        <h3 style='text-align:center;padding: 0 0 1rem;'>Ask Me Anything!</h3>
        
        Ask any question related to privacy practices carried out by this company that you'd like to know about! Try clicking on Random question to see a sample query.
        
        *Note: do not use keywords, but full-fledged questions.* The underlying models are not optimized to deal with keyword queries and might misunderstand you.
        """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Options")
    top_k_reader = st.sidebar.slider(
        "Max. number of answers",
        min_value=1,
        max_value=10,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        on_change=reset_results)
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=len(segments_dict),
        value=len(segments_dict),
        step=1,
        on_change=reset_results)
    #debug = st.sidebar.checkbox("Show debug info")



    # File upload block
    if not DISABLE_FILE_UPLOAD:
        st.sidebar.write("## File Upload:")
        data_files = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        for data_file in data_files:
            # Upload file
            if data_file:
                raw_json = upload_doc(data_file)
                st.sidebar.write(str(data_file.name) + " &nbsp;&nbsp; ✅ ")
                # if debug:
                #     st.subheader("REST API JSON response")
                #     st.sidebar.write(raw_json)



    hs_version = ""
    try:
        hs_version = f" <small>(v{haystack_version()})</small>"
    except Exception:
        pass

    st.sidebar.markdown(f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .haystack-footer {{
            text-align: center;
        }}
        .haystack-footer h4 {{
            margin: 0.1rem;
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    <div class="haystack-footer">
        <hr />
        <small>Detecting Textual Saliency in Privacy Policy.</small>
    </div>
    """, unsafe_allow_html=True)

    # Load csv into pandas dataframe
    try:
        df = pd.read_csv(EVAL_LABELS, sep=";")
    except Exception:
        st.error(f"The eval file was not found. Please check the demo's [README](https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")
        sys.exit(f"The eval file was not found under `{EVAL_LABELS}`. Please check the README (https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")

    # Search bar
    question = st.text_input("",
        value=state.question,
        max_chars=100,
        on_change=reset_results
    )
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    # Get next random question from the CSV
    if col2.button("Random question"):
        reset_results()
        new_row = df.sample(1)
        while new_row["Question Text"].values[0] == state.question:  # Avoid picking the same question twice (the change is not visible on the UI)
            new_row = df.sample(1)
        state.question = new_row["Question Text"].values[0]
        state.answer = new_row["Answer"].values[0]
        state.random_question_requested = True
        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
    else:
        state.random_question_requested = False
    
    run_query = (run_pressed or question != state.question) and not state.random_question_requested
    
    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Powering Up..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Cannot access Transformers API!")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        state.question = question
        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on the privacy policy document..."
        ):
            try:
                state.results, state.raw_json = query(question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Cannot access the document store!")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    if state.results:

        st.write("## Results:")

        for count, result in enumerate(state.results):
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                st.write(markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#65AAC3")) + context[end_idx:]), unsafe_allow_html=True)
                source = ""
                url, title = get_backlink(result)
                if url and title:
                    source = f"[{result['document']['meta']['title']}]({result['document']['meta']['url']})"
                else:
                    source = f"{result['source']}"
                st.markdown(f"**Relevance:** {result['relevance']} -  **Source:** {source}")

            else:
                st.write("🤔 &nbsp;&nbsp; We are unsure whether the policy document contains an answer to your question. Try to reformulate it!")
                st.write("**Relevance:** ", result["relevance"])

            st.write("___")

        # if debug:
        #     st.subheader("REST API JSON response")
        #     st.write(state.raw_json)

main()
