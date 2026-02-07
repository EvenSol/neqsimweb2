# GitHub Copilot Instructions for NeqSim Web Application

## Project Overview

This is a **Streamlit-based web application** for the NeqSim (Non-equilibrium Simulator) thermodynamic process simulation library. The application provides an interactive interface for fluid behavior simulations, phase equilibrium calculations, and process system modeling.

- **Framework**: Streamlit
- **Backend Library**: NeqSim (Java-based, accessed via Python wrapper)
- **Language**: Python
- **Deployment**: Streamlit Cloud

## Architecture & Project Structure

```
neqsimweb2/
├── welcome.py           # Main entry point and home page
├── fluids.py            # Fluid composition data definitions
├── pages/               # Streamlit multi-page app pages
│   ├── 0_TP_flash.py    # Temperature-Pressure flash calculations
│   ├── 10_Gas_Hydrate.py
│   ├── 20_Phase_envelope.py
│   ├── 30_Water Dew Point.py
│   ├── 40_LNGageing.py
│   ├── 50_Property Generator.py
│   ├── 60_Hydrogen.py
│   └── 70_Helium.py
├── images/              # Static images and logos
├── requirements.txt     # Python dependencies
└── Dockerfile           # Container configuration
```

## Key Technologies & Libraries

- **neqsim**: Core thermodynamic simulation library (Java via JPype)
- **streamlit**: Web UI framework
- **pandas**: Data manipulation and tables
- **matplotlib/plotly**: Visualization
- **numpy/scipy**: Numerical computations
- **altair**: Declarative visualizations
- **google-genai**: Google Gemini AI integration (via `google.genai`)

## NeqSim Java Library Reference

The underlying Java library is available at: https://github.com/equinor/neqsim

**Always use this repository as the primary reference** when implementing new functionality. The Java library (`jneqsim`) provides the full API for thermodynamic calculations. When adding new features:

1. Browse the Java source code to understand available methods and classes
2. Check existing Java examples in the repository for implementation patterns
3. The Python wrapper exposes Java classes via `jneqsim` - refer to Java documentation for method signatures

## NeqSim Python Wrapper

The Python wrapper for NeqSim is available at: https://github.com/equinor/neqsim-python

This wrapper provides Pythonic access to the Java library and includes:
- Helper functions like `fluid_df`, `TPflash`, `dataFrame`, `phaseenvelope`
- Direct access to Java classes via `jneqsim` module
- Refer to this repository for Python-specific usage patterns and examples

## Coding Conventions

### Streamlit Patterns

1. **Page Configuration**: Always set page config at the top of page files:
   ```python
   st.set_page_config(page_title="Page Name", page_icon='images/neqsimlogocircleflat.png')
   ```

2. **Session State**: Use `st.session_state` for persistent data across reruns:
   ```python
   if 'activefluid_df' not in st.session_state:
       st.session_state.activefluid_df = pd.DataFrame(default_fluid)
   ```

3. **Data Editors**: Use `st.data_editor` with proper column configuration:
   ```python
   st.edited_df = st.data_editor(
       data,
       column_config={
           "ColumnName": st.column_config.NumberColumn(
               "Display Name", min_value=0, max_value=100, format="%f"
           ),
       },
       num_rows='dynamic'
   )
   ```

4. **Layout**: Use `st.divider()`, `st.sidebar`, columns, and expanders for organization

### NeqSim Integration Patterns

1. **Fluid Creation**: Use the `fluid_df` function from neqsim.thermo:
   ```python
   from neqsim.thermo import fluid_df, TPflash, dataFrame
   
   neqsim_fluid = fluid_df(df, lastIsPlusFraction=False, add_all_components=False)
   neqsim_fluid.autoSelectModel()
   ```

2. **Flash Calculations**:
   ```python
   neqsim_fluid.setPressure(pressure, 'bara')
   neqsim_fluid.setTemperature(temp, 'C')
   TPflash(neqsim_fluid)
   results = dataFrame(neqsim_fluid)
   ```

3. **Phase Envelope**:
   ```python
   from neqsim.thermo import phaseenvelope
   thermoOps = jneqsim.thermodynamicOperations.ThermodynamicOperations(fluid)
   thermoOps.calcPTphaseEnvelope()
   ```

4. **Java Interop**: Access Java classes via `jneqsim`:
   ```python
   from neqsim import jneqsim
   jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
   ```

### Fluid Data Structures

Fluid compositions use standardized DataFrames with these columns:
- `ComponentName`: Chemical component name (e.g., "methane", "CO2")
- `MolarComposition[-]`: Molar fraction (dimensionless)
- `MolarMass[kg/mol]`: Molar mass (for plus fractions)
- `RelativeDensity[-]`: Relative density (for plus fractions)

### Component Naming

Use standard NeqSim component names from the [COMP.csv database](https://github.com/equinor/neqsim/blob/master/src/main/resources/data/COMP.csv):
- Common names: "methane", "ethane", "propane", "CO2", "H2S", "nitrogen", "water"
- Plus fractions: "C7", "C8", "C9", etc.
- Specialty: "MEG", "TEG", "methanol"

## Best Practices

### Error Handling

Wrap NeqSim operations in try-except blocks:
```python
try:
    TPflash(neqsim_fluid)
    results = dataFrame(neqsim_fluid)
except Exception as e:
    st.error(f"Calculation failed: {str(e)}")
```

### User Feedback

- Use `st.spinner()` for long-running calculations
- Use `st.success()`, `st.warning()`, `st.error()` for status messages
- Validate inputs before calculations

### Data Validation

```python
if st.edited_df['MolarComposition[-]'].sum() > 0:
    # Proceed with calculation
else:
    st.warning('Please enter fluid composition')
```

### Visualization

- Use Plotly for interactive charts (preferred for phase envelopes)
- Use Matplotlib for static scientific plots
- Always include axis labels with units

## Thermodynamic Concepts

When working with this codebase, understand these key concepts:

1. **Flash Calculations**: Determine phase equilibrium at given T and P
2. **Phase Envelope**: PT diagram showing bubble/dew point curves
3. **Cricondenbar/Cricondentherm**: Maximum pressure/temperature on phase envelope
4. **Plus Fractions**: Heavy hydrocarbon pseudo-components (C7+, C10+, etc.)
5. **EoS Models**: 
   - SRK-EoS: Standard cubic equation of state
   - PR-EoS: Peng-Robinson equation of state
   - CPA-EoS: For polar components (water, MEG, methanol)
   - UMR-PRU-EoS: For accurate phase envelope calculations

## Testing & Validation

- Test with standard natural gas compositions
- Validate against published thermodynamic data
- Check units consistency (bara, C, kg/mol)

## Common Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from neqsim.thermo import fluid_df, TPflash, phaseenvelope, dataFrame
from neqsim import jneqsim
from fluids import default_fluid, lng_fluid, detailedHC_data
```

## File Naming Convention

Page files in `pages/` folder follow the pattern:
- `{order}_{feature_name}.py`
- Order numbers: 0, 10, 20, 30... (allows inserting new pages)
- Example: `25_New_Feature.py` would appear between pages 20 and 30

## Performance Considerations

- NeqSim calculations can be CPU-intensive
- Use caching with `@st.cache_data` for expensive operations when appropriate
- Avoid unnecessary recalculations on Streamlit reruns
