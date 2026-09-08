"""Configure Java before a Streamlit page imports NeqSim and starts JPype."""

import os


def configure_java_runtime():
    """Use the Python-installed Java 21 runtime unless JAVA_HOME is supplied."""
    if not os.environ.get("JAVA_HOME"):
        from jdk4py import JAVA_HOME

        os.environ["JAVA_HOME"] = str(JAVA_HOME)

    # Preserve the app's XStream module-access flags on Java 17+.
    # This must run before any import triggers jpype.startJVM().
    if "add-opens" not in os.environ.get("JAVA_TOOL_OPTIONS", ""):
        os.environ["JAVA_TOOL_OPTIONS"] = (
            "--add-opens=java.base/java.util=ALL-UNNAMED "
            "--add-opens=java.base/java.lang=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
            "--add-opens=java.base/java.io=ALL-UNNAMED"
        )
