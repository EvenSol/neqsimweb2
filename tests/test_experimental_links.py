"""Direct-link policy tests; no simulation or Streamlit runtime required."""

import unittest

from experimental_links import is_experimental_url


class ExperimentalLinkTest(unittest.TestCase):
    def test_experimental_links(self):
        for url in (
            "https://neqsim.streamlit.app/CO2_mechanisticModel",
            "https://neqsim.streamlit.app/CO2_mechanisticModel?embed=true#results",
            "https://example.org/neqsim/CO2_mechanisticModel/",
            "https://example.org/neqsim/%43O2_mechanisticModel",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_experimental_url(url, ["CO2_mechanisticModel"]))

    def test_home_stable_and_unknown_links_do_not_enable_experimental(self):
        for url in (
            None, "", "https://neqsim.streamlit.app/",
            "https://neqsim.streamlit.app/TP_flash",
            "https://neqsim.streamlit.app/unknown",
            "https://neqsim.streamlit.app/?page=CO2_mechanisticModel",
            "https://neqsim.streamlit.app/CO2_mechanisticModel_extra",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_experimental_url(url, ["CO2_mechanisticModel"]))

    def test_any_registered_experimental_route_is_supported(self):
        self.assertTrue(is_experimental_url(
            "https://neqsim.streamlit.app/NeqSim_Studio",
            iter(["CO2_mechanisticModel", "NeqSim_Studio"]),
        ))


if __name__ == "__main__":
    unittest.main()
