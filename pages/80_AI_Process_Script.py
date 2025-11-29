import json
import re
import traceback
from textwrap import dedent

import pandas as pd
import streamlit as st
from openai import OpenAI

import neqsim

st.set_page_config(page_title="AI Process Builder", page_icon="images/neqsimlogocircleflat.png")

st.title("AI-Assisted NeqSim Process Builder")

st.write(
    """
    Describe the process you want to simulate, and the assistant will draft a Python
    script that configures and runs the NeqSim process. The script is executed locally
    and the resulting data is displayed below.
    """
)

sidebar = st.sidebar
openai_api_key = sidebar.text_input("OpenAI API Key", type="password", key="ai_process_openai_key")
model_choice = sidebar.selectbox(
    "Model", ["gpt-3.5-turbo-instruct", "gpt-4o-mini"], index=0,
    help="Model used to propose the NeqSim script",
)
max_tokens = sidebar.slider("Max tokens", 200, 1200, 600, step=100)
temperature = sidebar.slider("Temperature", 0.0, 1.2, 0.2, step=0.1)

user_prompt = st.text_area(
    "Describe the process",
    value="Simulate a simple gas dehydration unit with a separator and glycol contactor.",
    height=150,
    help="Explain the units, feeds, conditions, and properties you want to report.",
)

refine_prompt = st.text_area(
    "Adjust or refine the process (optional)",
    value="",
    height=120,
    help="Add clarifications or modifications; these instructions are merged into the request.",
)

st.info(
    "The generated code is executed in a limited environment. It must define a "
    "`process_results` dictionary (JSON-serializable) and can optionally define a "
    "`process_log` string for additional notes. Failures during execution will trigger "
    "an automatic attempt to repair the script."
)


def extract_code_block(model_text: str) -> str:
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, model_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return model_text.strip()


def run_generated_code(code: str):
    local_env: dict = {
        "neqsim": neqsim,
        "json": json,
        "pd": pd,
    }

    exec(  # nosec
        code,
        {"__builtins__": {"range": range, "len": len, "min": min, "max": max}},
        local_env,
    )

    results = local_env.get("process_results", {})
    if isinstance(results, pd.DataFrame):
        results = results.to_dict(orient="records")
    elif hasattr(results, "to_json"):
        try:
            results = json.loads(results.to_json())
        except Exception:
            results = str(results)

    log_text = local_env.get("process_log", "")
    return results, log_text, local_env


def build_system_prompt() -> str:
    return dedent(
        """
        You are an expert NeqSim process engineer. Generate Python code that sets up the
        requested process using the `neqsim` library only. The code must:
        - Be fully executable without internet or file I/O.
        - Avoid placeholders; provide reasonable default values if the user is vague.
        - Finish by populating a JSON-serializable dictionary named `process_results`
          with key metrics, stream data, or tables.
        - Optionally set a `process_log` string for notes.
        Keep imports minimal and only use the modules provided in the environment.
        Wrap the code in a Python Markdown code block.
        """
    ).strip()


def request_process_script(prompt_text: str, *, model: str, api_key: str) -> str:
    OpenAI.api_key = api_key
    client = OpenAI(api_key=api_key)
    completion = client.completions.create(
        model=model,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw_text = completion.choices[0].text
    return extract_code_block(raw_text)


col_generate, col_run = st.columns(2)
generated_code = st.session_state.get("generated_process_code", "")
max_correction_attempts = 3

if col_generate.button("Generate process script", type="primary"):
    if not openai_api_key:
        st.error("Please provide an OpenAI API key in the sidebar.")
    else:
        system_prompt = build_system_prompt()
        user_message = dedent(
            f"""
            Process description: {user_prompt.strip()}
            Refinements: {refine_prompt.strip() or 'None'}
            Provide only the Python code.
            """
        ).strip()

        prompt_text = f"{system_prompt}\n\nUser request:\n{user_message}"

        generated_code = request_process_script(
            prompt_text, model=model_choice, api_key=openai_api_key
        )
        st.session_state["generated_process_code"] = generated_code

run_clicked = col_run.button("Run script", disabled=not bool(generated_code))

if run_clicked and generated_code:
    with st.spinner("Running NeqSim process with auto-correction..."):
        current_code = generated_code
        results = {}
        log_text = ""
        success = False
        attempt_details = []

        for attempt in range(1, max_correction_attempts + 1):
            try:
                results, log_text, _ = run_generated_code(current_code)
                success = True
                break
            except Exception as exc:  # pylint: disable=broad-except
                error_trace = traceback.format_exc()
                attempt_details.append((exc, error_trace))

                if not openai_api_key:
                    break

                repair_prompt = dedent(
                    f"""
                    {build_system_prompt()}

                    The previous script raised an error. Please revise the code to fix the
                    runtime issue while keeping the requested process intent. Ensure
                    `process_results` is filled with JSON-serializable data.

                    Original description: {user_prompt.strip()}
                    Refinements: {refine_prompt.strip() or 'None'}
                    Current script (attempt {attempt}):\n{current_code}
                    Error:\n{exc}\n
                    Full traceback:\n{error_trace}
                    Provide only the corrected Python code.
                    """
                ).strip()

                with st.spinner("Requesting a corrected script..."):
                    corrected_code = request_process_script(
                        repair_prompt, model=model_choice, api_key=openai_api_key
                    )
                    current_code = corrected_code

        if success:
            st.session_state["generated_process_code"] = current_code
            st.success(
                f"Process executed successfully after {attempt} attempt(s). Presenting the validated script."
            )

            if log_text:
                st.markdown("### Process log")
                st.write(log_text)

            st.markdown("### Results (JSON)")
            st.json(results)

            st.download_button(
                "Download results as JSON",
                data=json.dumps(results, indent=2),
                file_name="neqsim_process_results.json",
                mime="application/json",
            )
        else:
            st.error("Process execution failed after auto-correction attempts.")

            for idx, (exc, err_trace) in enumerate(attempt_details, start=1):
                with st.expander(f"Attempt {idx} error details", expanded=False):
                    st.exception(exc)
                    st.code(err_trace)

            if not openai_api_key:
                st.warning("Add an OpenAI API key to auto-correct the script.")
            else:
                st.info(
                    "Review the last suggested script below, adjust the description, and try again."
                )

            st.session_state["generated_process_code"] = current_code
            generated_code = current_code

if generated_code:
    st.subheader("Generated script")
    st.code(generated_code, language="python")
