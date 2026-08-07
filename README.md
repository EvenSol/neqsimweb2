![NeqSim Logo](https://github.com/equinor/neqsim/blob/master/docs/wiki/neqsimlogocircleflatsmall.png)

# NeqSim Web application
The NeqSim Web project is a web user interface for the [neqsim process simulator](https://equinor.github.io/neqsimhome/).

The application is in initial development and is using the [streamlit framework](https://streamlit.io/).

The application is hosted by streamlit and can be tested at [neqsim web app link](https://neqsim.streamlit.app/).

## Process Flowsheet Studio

Process Flowsheet Studio builds and solves reusable steady-state NeqSim cases
from shared fluid packages, independent inlet streams, unit operations, and
explicit material or energy connections.

Enable an equipment design basis in a pump, two-sided heat exchanger, valve,
or pipeline to compare its solved operating point with explicit capacities.
The **Workbook · Design** view reports operating value, capacity, margin,
utilization, status, and engineering unit in a normalized table. Pipeline rows
also identify the critical native velocity-profile segment and length when the
NeqSim unit exposes that profile.

The downloadable engineering workbook contains the same review-ready data in
the **Equipment Design** worksheet together with streams, equipment,
constraints, convergence, and conservation evidence. These results support
screening and engineering studies; they are not design certification.
