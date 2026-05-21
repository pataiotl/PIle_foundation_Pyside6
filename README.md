# Pile Foundation Designer

Preliminary reinforced-concrete pile-cap design aid for 2 to 12 piles, with both a Streamlit web UI and a PySide6 desktop UI.

The app supports manual service-load input and SAP2000 joint reaction imports, then performs pile reaction distribution, envelope checks, RC flexure checks, one-way shear, two-way punching shear, bearing, development length estimates, and strut-and-tie advisory output.

> Important: this software is a calculation aid. It is not a replacement for a licensed engineer's judgment, project specifications, geotechnical report, local code amendments, or final ACI 318 design review.

## Features

- Pile cap layouts for 2 to 12 piles
- Manual Dead/Live service load input with factored ultimate combinations
- SAP2000 joint reaction table import from CSV/XLSX
- Foundation group creation by selected SAP joint numbers
- Service pile reaction envelopes for pile capacity checks
- Ultimate pile reaction envelopes for RC design checks
- Pile compression and uplift checks
- Bottom and top reinforcement checks
- One-way shear and two-way punching shear checks
- Concrete bearing check
- ACI-style development length estimate
- Strut-and-tie / deep pile cap advisory
- Drawing-style plan and elevation output
- Editable tables with horizontal and vertical scrolling
- CSV, Markdown, PDF, PNG, and editable XLSX state exports

## Project Structure

```text
Pile-Foundation2/
|-- Streamlit/
|   `-- pile_foundation_app.py       # Original Streamlit application
|-- Pyside6/
|   |-- run_pyside6.py               # Desktop app entry point
|   |-- start_app.bat                # Windows double-click launcher
|   |-- main_window.py               # PySide6 UI
|   |-- pile_engine.py               # Headless engineering core
|   |-- qt_models.py                 # Editable pandas table model
|   |-- view_data.py                 # Result table builders
|   |-- requirements.txt             # Desktop app dependencies
|   `-- README.md                    # PySide-specific notes
`-- README.md
```

## Requirements

- Python 3.10 or newer recommended
- Windows is the primary tested environment for the PySide6 desktop app

Desktop dependencies:

```powershell
pip install -r .\Pyside6\requirements.txt
```

Streamlit dependencies:

```powershell
pip install streamlit pandas numpy matplotlib openpyxl fpdf
```

## Run the PySide6 Desktop App

From the repository root:

```powershell
python .\Pyside6\run_pyside6.py
```

Or on Windows, double-click:

```text
Pyside6/start_app.bat
```

## Run the Streamlit App

From the repository root:

```powershell
streamlit run .\Streamlit\pile_foundation_app.py
```

## Basic Workflow

1. Open the PySide6 desktop app or Streamlit app.
2. Set materials, pile geometry, cap geometry, column/pedestal dimensions, cover, and reinforcement.
3. Choose the load source:
   - Manual input: enter Dead and Live service loads.
   - SAP2000 import: load a joint reaction table and create foundation groups by selected joints.
4. Click `DESIGN`.
5. Review:
   - Group summary
   - Strength/service check table
   - Pile reaction envelopes
   - Calculation tab formulas
   - Drawing output
   - STM advisory
6. Export reports, tables, drawings, or editable state workbooks as needed.

## SAP2000 Import Notes

The importer expects a joint reaction table containing columns similar to:

```text
Joint, OutputCase, CaseType, F1, F2, F3, M1, M2, M3
```

The app attempts to detect equivalent column names. In the PySide6 UI, verify the column mapping before running the design:

- Vertical reaction, usually `F3`
- Moment about X, usually `M1`
- Moment about Y, usually `M2`
- Optional lateral/torsion columns
- Sign multipliers for vertical reaction and moments

SAP rows are not summed across different joints. Each selected `Joint + OutputCase` is designed separately, then enveloped for the foundation group.

## Engineering Basis

The design workflow includes:

- Elastic rigid pile-cap reaction distribution
- Compression-positive pile reactions
- ACI-style flexural strength check:

```text
phi Mn >= Mu
Mn = As fy (d - a/2)
a = As fy / (0.85 fc' b)
```

- ACI-style one-way shear check
- ACI-style two-way punching shear check
- Concrete bearing:

```text
phi Pn = phi * 0.85 fc' A1
```

- ACI-style tension development length estimate
- Practical strut-and-tie advisory for deep pile caps

Always verify final design directly against the governing ACI 318 edition and project-specific requirements.

## Outputs

The app can export:

- Markdown calculation report
- PDF calculation report
- CSV check tables
- CSV pile reaction envelopes
- PNG drawing output
- Editable XLSX state workbook

## Limitations

- This is not a final sealed design tool.
- Lateral pile group behavior, torsion design, pile head fixity, seismic detailing, settlement, geotechnical group effects, and construction tolerances require separate engineering review.
- The ACI-style formulas are implemented as a design aid and must be checked against the governing code edition, local amendments, and project criteria.
- Imported SAP reactions must be reviewed for sign convention and load-combination intent.

## Development Notes

The PySide6 version separates the desktop UI from the engineering core:

- Keep engineering changes in `Pyside6/pile_engine.py`.
- Keep UI changes in `Pyside6/main_window.py`.
- Keep table-display behavior in `Pyside6/qt_models.py` and `Pyside6/view_data.py`.

Before publishing to GitHub, avoid committing generated cache files such as `__pycache__/` and `*.pyc`.
