"""Exercise the Python-installed JVM used by Streamlit Cloud deployments."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from jdk4py import JAVA_HOME

from java_runtime import configure_java_runtime


ROOT = Path(__file__).resolve().parents[1]


class JavaRuntimeTest(unittest.TestCase):
    def test_missing_or_empty_java_home_uses_bundled_runtime(self):
        for value in (None, ""):
            with self.subTest(java_home=value), patch.dict(os.environ):
                os.environ.pop("JAVA_HOME", None)
                if value is not None:
                    os.environ["JAVA_HOME"] = value
                configure_java_runtime()
                self.assertEqual(Path(os.environ["JAVA_HOME"]), JAVA_HOME)

    def test_explicit_runtime_and_module_options_survive_reruns(self):
        settings = {
            "JAVA_HOME": "/configured/java",
            "JAVA_TOOL_OPTIONS": "--add-opens=java.base/java.util=ALL-UNNAMED",
        }
        with patch.dict(os.environ, settings):
            configure_java_runtime()
            configure_java_runtime()
            for key, value in settings.items():
                self.assertEqual(os.environ[key], value)

    def test_bundled_jvm_runs_and_restores_a_native_process(self):
        # A fresh interpreter is required: JPype can start only one JVM per
        # process. Verify java.home so a runner's preinstalled JDK cannot mask
        # an installation or discovery failure in the Python dependency.
        program = textwrap.dedent(
            """
            import json
            import math
            from pathlib import Path
            import tempfile

            from jdk4py import JAVA_HOME
            from java_runtime import configure_java_runtime

            configure_java_runtime()

            from neqsim import jneqsim, jpype, open_neqsim, save_neqsim

            java_home = Path(str(jpype.JClass("java.lang.System").getProperty("java.home")))
            assert java_home.resolve() == JAVA_HOME.resolve(), java_home

            fluid = jneqsim.thermo.system.SystemSrkEos(323.15, 50.0)
            fluid.addComponent("methane", 0.9)
            fluid.addComponent("ethane", 0.1)
            fluid.setMixingRule("classic")
            fluid.setTotalFlowRate(1.0, "kg/sec")
            operations = jneqsim.thermodynamicoperations.ThermodynamicOperations(fluid)
            operations.TPflash()
            fluid.initProperties()
            density = fluid.getDensity("kg/m3")
            assert fluid.getNumberOfPhases() == 1
            assert 20.0 < density < 60.0, density

            feed = jneqsim.process.equipment.stream.Stream("feed", fluid)
            cooler = jneqsim.process.equipment.heatexchanger.Cooler("cooler", feed)
            cooler.setOutTemperature(303.15)
            process = jneqsim.process.processmodel.ProcessSystem()
            process.add(feed)
            process.add(cooler)
            process.run()
            outlet = cooler.getOutletStream()
            assert math.isclose(outlet.getFlowRate("kg/sec"), 1.0, abs_tol=1e-8)
            assert math.isclose(outlet.getTemperature("K"), 303.15, abs_tol=1e-8)

            with tempfile.TemporaryDirectory() as directory:
                filename = str(Path(directory) / "process.neqsim")
                assert save_neqsim(process, filename)
                restored = open_neqsim(filename)
                assert restored is not None
                restored.run()
                restored_outlet = restored.getUnit("cooler").getOutletStream()
                assert math.isclose(restored_outlet.getFlowRate("kg/sec"), 1.0, abs_tol=1e-8)
                assert math.isclose(restored_outlet.getTemperature("K"), 303.15, abs_tol=1e-8)

            print(json.dumps({
                "java_version": str(jpype.JClass("java.lang.System").getProperty("java.version")),
                "gas_density_kg_m3": density,
                "process_roundtrip": "passed",
            }))
            """
        )
        environment = os.environ.copy()
        for name in ("JAVA_HOME", "JAVA_TOOL_OPTIONS", "NEQSIM_JVM_AUTOSTART"):
            environment.pop(name, None)
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
