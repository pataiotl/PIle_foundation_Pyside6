from __future__ import annotations

import math
import traceback
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableView,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import pile_engine as eng
import view_data
from qt_models import PandasTableModel


class DesignWorker(QThread):
    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, mode: str, payload: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.mode = mode
        self.payload = payload

    def run(self) -> None:
        try:
            if self.mode == "manual":
                batch = eng.design_batch_from_manual_table(**self.payload)
            else:
                batch = eng.design_batch_from_sap_table(**self.payload)
            self.succeeded.emit(batch)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PileFoundationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(eng.APP_TITLE)
        self.resize(1480, 920)

        self.batch: Optional[Dict[str, Any]] = None
        self.current_item: Optional[Dict[str, Any]] = None
        self.sap_df = pd.DataFrame()
        self.worker: Optional[DesignWorker] = None
        self.plan_canvas: Optional[FigureCanvas] = None
        self.elev_x_canvas: Optional[FigureCanvas] = None
        self.elev_y_canvas: Optional[FigureCanvas] = None

        self._build_actions()
        self._build_layout()
        self._apply_style()
        self._connect_input_updates()
        self._refresh_pile_preview()
        self._set_idle_result_state()

    # ------------------------------------------------------------------
    # Window and common UI
    # ------------------------------------------------------------------

    def _build_actions(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.design_action = QAction("DESIGN", self)
        self.design_action.triggered.connect(self.run_design)
        toolbar.addAction(self.design_action)

        load_action = QAction("Load State", self)
        load_action.triggered.connect(self.load_state)
        toolbar.addAction(load_action)

        save_action = QAction("Save State", self)
        save_action.triggered.connect(self.save_state)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        export_report_action = QAction("Export Report", self)
        export_report_action.triggered.connect(self.save_markdown_report)
        toolbar.addAction(export_report_action)

    def _build_layout(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_tabs())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1090])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_sidebar(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setMinimumWidth(360)
        outer.setMaximumWidth(460)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        title = QLabel(eng.APP_TITLE)
        title.setObjectName("titleLabel")
        title.setWordWrap(True)
        subtitle = QLabel(eng.APP_SUBTITLE)
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.include_self_weight = QCheckBox("Include pile-cap self weight in pile reactions")
        self.include_self_weight.setChecked(True)
        layout.addWidget(self.include_self_weight)

        layout.addWidget(self._material_group())
        layout.addWidget(self._geometry_group())
        layout.addWidget(self._reinforcement_group())
        layout.addStretch(1)

        outer.setWidget(panel)
        return outer

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._input_tab(), "1 Input")
        self.tabs.addTab(self._results_tab(), "2 Design Results")
        self.tabs.addTab(self._calculation_tab(), "3 Calculation")
        self.tabs.addTab(self._drawing_tab(), "4 Drawing Output")
        self.tabs.addTab(self._stm_tab(), "5 STM Advisory")
        self.tabs.addTab(self._report_tab(), "6 Report / Export")
        self.tabs.addTab(self._basis_tab(), "7 Basis")
        return self.tabs

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                font-size: 12px;
                color: #f8fafc;
                background: #000000;
            }
            #titleLabel { font-size: 22px; font-weight: 700; }
            #sectionTitle { font-size: 16px; font-weight: 700; }
            #subtitleLabel { color: #cbd5e1; }
            QGroupBox {
                font-weight: 650;
                color: #f8fafc;
                background: #000000;
                border: 1px solid #64748b;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QTabWidget::pane {
                border: 1px solid #334155;
                background: #000000;
            }
            QTabBar::tab {
                color: #f8fafc;
                background: #111827;
                border: 1px solid #334155;
                padding: 6px 12px;
            }
            QTabBar::tab:selected {
                background: #1f2937;
                border-bottom: 1px solid #1f2937;
            }
            QLineEdit, QComboBox, QAbstractSpinBox, QListWidget {
                color: #f8fafc;
                background: #111827;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QPushButton, QToolButton {
                color: #f8fafc;
                background: #374151;
                border: 1px solid #9ca3af;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: 650;
                min-height: 24px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #4b5563;
                border: 1px solid #d1d5db;
            }
            QPushButton:pressed, QToolButton:pressed {
                background: #1f2937;
                border: 1px solid #60a5fa;
                padding-top: 7px;
                padding-left: 13px;
            }
            QPushButton:disabled, QToolButton:disabled {
                color: #9ca3af;
                background: #2f3338;
                border: 1px solid #4b5563;
            }
            QToolBar {
                spacing: 6px;
                padding: 4px;
                border-bottom: 1px solid #3f4650;
            }
            QTableView {
                color: #f8fafc;
                background: #000000;
                alternate-background-color: #050505;
                gridline-color: #334155;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                color: #f8fafc;
                background: #1f2937;
                border: 1px solid #4b5563;
                padding: 4px;
            }
            QTextEdit {
                color: #f8fafc;
                background: #000000;
                border: 1px solid #475569;
                border-radius: 5px;
            }
            QLabel[metric="true"] {
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 6px;
                background: #000000;
                padding: 8px;
            }
            QScrollArea {
                background: #000000;
                border: none;
            }
            QScrollBar:horizontal, QScrollBar:vertical {
                background: #111827;
                border: 1px solid #334155;
            }
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical {
                background: #64748b;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {
                background: #94a3b8;
            }
            """
        )

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------

    def _material_group(self) -> QGroupBox:
        group = QGroupBox("Materials")
        form = QFormLayout(group)
        self.fc_MPa = self._double_spin(15, 100, 35, 1, 1)
        self.fy_MPa = self._double_spin(275, 700, 500, 10, 1)
        self.lambda_c = self._double_spin(0.50, 1.00, 1.00, 0.05, 2)
        self.gamma_conc_kN_m3 = self._double_spin(20, 28, 24, 0.5, 1)
        self.phi_flexure = self._double_spin(0.60, 0.95, 0.90, 0.01, 2)
        self.phi_shear = self._double_spin(0.50, 0.90, 0.75, 0.01, 2)
        self.phi_bearing = self._double_spin(0.50, 0.90, 0.65, 0.01, 2)
        self.phi_stm_tie = self._double_spin(0.50, 0.90, 0.75, 0.01, 2)
        form.addRow("Concrete fc' (MPa)", self.fc_MPa)
        form.addRow("Rebar fy (MPa)", self.fy_MPa)
        form.addRow("lambda lightweight factor", self.lambda_c)
        form.addRow("Concrete unit weight (kN/m3)", self.gamma_conc_kN_m3)
        form.addRow("phi flexure", self.phi_flexure)
        form.addRow("phi shear / punching", self.phi_shear)
        form.addRow("phi bearing", self.phi_bearing)
        form.addRow("phi STM tie", self.phi_stm_tie)
        return group

    def _geometry_group(self) -> QGroupBox:
        group = QGroupBox("Pile and Cap Geometry")
        form = QFormLayout(group)
        self.n_piles = QSpinBox()
        self.n_piles.setRange(2, 12)
        self.n_piles.setValue(4)
        self.pile_diameter_mm = self._double_spin(200, 2500, 600, 50, 1)
        self.cap_thickness_mm = self._double_spin(300, 4000, 1200, 50, 1)
        self.edge_from_pile_edge_mm = self._double_spin(100, 2000, 600, 50, 1)
        self.spacing_x_mm = self._double_spin(500, 6000, 1800, 50, 1)
        self.spacing_y_mm = self._double_spin(500, 6000, 1800, 50, 1)
        self.pile_capacity_comp_kN = self._double_spin(1, 50000, 1800, 50, 1)
        self.pile_capacity_tension_kN = self._double_spin(0, 50000, 400, 25, 1)
        self.column_bx_mm = self._double_spin(200, 4000, 800, 50, 1)
        self.column_by_mm = self._double_spin(200, 4000, 800, 50, 1)
        self.pedestal_bx_mm = self._double_spin(200, 6000, 800, 50, 1)
        self.pedestal_by_mm = self._double_spin(200, 6000, 800, 50, 1)
        self.bottom_cover_mm = self._double_spin(40, 300, 100, 5, 1)
        self.top_cover_mm = self._double_spin(30, 200, 75, 5, 1)
        self.side_cover_mm = self._double_spin(30, 200, 75, 5, 1)
        self.use_pedestal_for_shear = QCheckBox("Use pedestal size for shear/bearing checks")
        self.use_pedestal_for_shear.setChecked(True)
        self.column_location = QComboBox()
        self.column_location.addItems(["Interior", "Edge", "Corner"])
        form.addRow("Number of piles", self.n_piles)
        form.addRow("Pile diameter / equivalent width (mm)", self.pile_diameter_mm)
        form.addRow("Pile cap thickness h (mm)", self.cap_thickness_mm)
        form.addRow("Edge from pile edge to cap edge (mm)", self.edge_from_pile_edge_mm)
        form.addRow("Typical pile spacing X (mm)", self.spacing_x_mm)
        form.addRow("Typical pile spacing Y (mm)", self.spacing_y_mm)
        form.addRow("Pile compression capacity (kN)", self.pile_capacity_comp_kN)
        form.addRow("Pile tension capacity (kN)", self.pile_capacity_tension_kN)
        form.addRow("Column size Bx (mm)", self.column_bx_mm)
        form.addRow("Column size By (mm)", self.column_by_mm)
        form.addRow("Pedestal / loaded area Bx (mm)", self.pedestal_bx_mm)
        form.addRow("Pedestal / loaded area By (mm)", self.pedestal_by_mm)
        form.addRow(self.use_pedestal_for_shear)
        form.addRow("Column location for punching alpha_s", self.column_location)
        form.addRow("Bottom cover (mm)", self.bottom_cover_mm)
        form.addRow("Top cover (mm)", self.top_cover_mm)
        form.addRow("Side cover (mm)", self.side_cover_mm)
        return group

    def _reinforcement_group(self) -> QGroupBox:
        group = QGroupBox("Concrete Cover and Reinforcement")
        form = QFormLayout(group)
        bars = list(eng.BAR_DATABASE_MM.keys())
        self.main_bar_x = self._combo(bars, "DB25")
        self.spacing_x_reinf_mm = self._double_spin(75, 400, 150, 25, 1)
        self.main_bar_y = self._combo(bars, "DB25")
        self.spacing_y_reinf_mm = self._double_spin(75, 400, 150, 25, 1)
        self.top_bar = self._combo(bars, "DB16")
        self.top_spacing_mm = self._double_spin(75, 400, 200, 25, 1)
        self.side_face_bar = self._combo(bars, "DB16")
        self.side_face_spacing_mm = self._double_spin(100, 400, 250, 25, 1)
        self.hook_extension_mm = self._double_spin(0, 2000, 300, 25, 1)
        self.preferred_spacing_step_mm = self._double_spin(5, 100, 25, 5, 1)
        form.addRow("Bottom bars parallel X", self.main_bar_x)
        form.addRow("Bottom X bar spacing (mm)", self.spacing_x_reinf_mm)
        form.addRow("Bottom bars parallel Y", self.main_bar_y)
        form.addRow("Bottom Y bar spacing (mm)", self.spacing_y_reinf_mm)
        form.addRow("Top bars / nominal top reinforcement", self.top_bar)
        form.addRow("Top bar spacing (mm)", self.top_spacing_mm)
        form.addRow("Side face bars", self.side_face_bar)
        form.addRow("Side face bar spacing (mm)", self.side_face_spacing_mm)
        form.addRow("Hook extension (mm)", self.hook_extension_mm)
        form.addRow("Preferred spacing step (mm)", self.preferred_spacing_step_mm)
        return group

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setAccelerated(True)
        box.setKeyboardTracking(False)
        return box

    @staticmethod
    def _combo(values: list[str], current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        index = combo.findText(current)
        combo.setCurrentIndex(max(index, 0))
        return combo

    @staticmethod
    def _manual_header_aliases() -> dict[str, str]:
        return {
            "D Ps (kN)": "Dead Ps (kN)",
            "D Msx (kN-m)": "Dead Msx (kN-m)",
            "D Msy (kN-m)": "Dead Msy (kN-m)",
            "L Ps (kN)": "Live Ps (kN)",
            "L Msx (kN-m)": "Live Msx (kN-m)",
            "L Msy (kN-m)": "Live Msy (kN-m)",
            "D Factor": "Dead factor",
            "L Factor": "Live factor",
        }

    @staticmethod
    def _manual_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = eng.normalize_manual_service_load_columns(df)
        preferred = [
            "Group Name",
            "D Ps (kN)",
            "D Msx (kN-m)",
            "D Msy (kN-m)",
            "L Ps (kN)",
            "L Msx (kN-m)",
            "L Msy (kN-m)",
            "D Factor",
            "L Factor",
            "No. Piles",
            "Thickness (mm)",
            "Pile Dia (mm)",
            "Spacing X (mm)",
            "Spacing Y (mm)",
            "Edge (mm)",
            "Pile Comp Cap (kN)",
            "Pile Tension Cap (kN)",
            "Joint IDs",
        ]
        columns = [col for col in preferred if col in df.columns]
        columns += [col for col in df.columns if col not in columns]
        return df[columns]

    @staticmethod
    def _manual_load_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = eng.normalize_manual_service_load_columns(df)
        columns = [
            "Group Name",
            "D Ps (kN)",
            "D Msx (kN-m)",
            "D Msy (kN-m)",
            "L Ps (kN)",
            "L Msx (kN-m)",
            "L Msy (kN-m)",
            "D Factor",
            "L Factor",
        ]
        return df[[col for col in columns if col in df.columns]]

    @staticmethod
    def _manual_geometry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = eng.normalize_manual_service_load_columns(df)
        columns = [
            "Group Name",
            "No. Piles",
            "Thickness (mm)",
            "Pile Dia (mm)",
            "Spacing X (mm)",
            "Spacing Y (mm)",
            "Edge (mm)",
            "Pile Comp Cap (kN)",
            "Pile Tension Cap (kN)",
            "Joint IDs",
        ]
        return df[[col for col in columns if col in df.columns]]

    def _connect_input_updates(self) -> None:
        controls = [
            self.n_piles,
            self.pile_diameter_mm,
            self.edge_from_pile_edge_mm,
            self.spacing_x_mm,
            self.spacing_y_mm,
            self.cap_thickness_mm,
        ]
        for control in controls:
            signal = control.valueChanged if hasattr(control, "valueChanged") else control.currentIndexChanged
            signal.connect(self._refresh_pile_preview)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _input_tab(self) -> QWidget:
        widget = QWidget()
        widget.setMinimumWidth(980)
        widget.setMinimumHeight(1240)
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.load_source = QComboBox()
        self.load_source.addItems(["Manual input", "SAP2000 joint reaction import"])
        self.load_source.currentIndexChanged.connect(self._switch_load_source)
        top.addWidget(QLabel("Load source"))
        top.addWidget(self.load_source, 0)
        top.addStretch(1)
        design_button = QPushButton("DESIGN")
        design_button.clicked.connect(self.run_design)
        top.addWidget(design_button)
        layout.addLayout(top)

        self.input_stack = QTabWidget()
        self.input_stack.addTab(self._manual_input_page(), "Manual")
        self.input_stack.addTab(self._sap_input_page(), "SAP2000")
        self.input_stack.setMinimumHeight(910)
        layout.addWidget(self.input_stack, 5)

        preview_group = QGroupBox("Pile layout preview")
        preview_group.setMinimumHeight(230)
        preview_layout = QVBoxLayout(preview_group)
        self.pile_preview_info = QLabel("")
        preview_layout.addWidget(self.pile_preview_info)
        self.pile_preview_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.pile_preview_table = self._table_view(self.pile_preview_model, min_height=150)
        preview_layout.addWidget(self.pile_preview_table)
        layout.addWidget(preview_group, 2)
        return self._scroll_page(widget)

    def _manual_input_page(self) -> QWidget:
        page = QWidget()
        page.setMinimumWidth(940)
        layout = QVBoxLayout(page)
        button_row = QHBoxLayout()
        for text, slot in [
            ("Apply geometry to table", self.apply_geometry_to_manual_table),
            ("Add foundation", self.add_manual_row),
            ("Duplicate selected", self.duplicate_manual_row),
            ("Remove selected", self.remove_manual_rows),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            button_row.addWidget(button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        manual_caption = QLabel(
            "Manual service loads are shown first. Enter Dead and Live service loads; the app applies the D/L factors for ultimate RC checks."
        )
        manual_caption.setWordWrap(True)
        manual_caption.setObjectName("subtitleLabel")
        layout.addWidget(manual_caption)

        default_manual = eng.default_manual_foundations(self.current_geometry())
        self.manual_load_model = PandasTableModel(
            self._manual_load_dataframe(default_manual),
            editable=True,
            header_aliases=self._manual_header_aliases(),
        )
        self.manual_geometry_model = PandasTableModel(
            self._manual_geometry_dataframe(default_manual),
            editable=True,
            header_aliases=self._manual_header_aliases(),
        )
        self.manual_table = self._table_view(self.manual_load_model, min_height=220)
        self.manual_geometry_table = self._table_view(self.manual_geometry_model, min_height=260)
        layout.addWidget(QLabel("Service load input"))
        layout.addWidget(self.manual_table)
        layout.addWidget(QLabel("Foundation geometry and pile capacity overrides"))
        layout.addWidget(self.manual_geometry_table)
        return page

    def _sap_input_page(self) -> QWidget:
        page = QWidget()
        page.setMinimumWidth(1120)
        page.setMinimumHeight(900)
        layout = QVBoxLayout(page)
        file_row = QHBoxLayout()
        load_button = QPushButton("Load SAP2000 CSV/XLSX")
        load_button.clicked.connect(self.load_sap_table)
        self.sap_file_label = QLabel("No SAP2000 table loaded")
        self.sap_file_label.setWordWrap(True)
        file_row.addWidget(load_button)
        file_row.addWidget(self.sap_file_label, 1)
        layout.addLayout(file_row)

        map_group = QGroupBox("Column mapping")
        grid = QGridLayout(map_group)
        self.vertical_col = QComboBox()
        self.mx_col = QComboBox()
        self.my_col = QComboBox()
        self.vx_col = QComboBox()
        self.vy_col = QComboBox()
        self.torsion_col = QComboBox()
        self.vertical_sign = self._combo(["1.0", "-1.0"], "1.0")
        self.moment_sign = self._combo(["1.0", "-1.0"], "1.0")
        labels = [
            ("Vertical reaction", self.vertical_col),
            ("Moment about X", self.mx_col),
            ("Moment about Y", self.my_col),
            ("Vx", self.vx_col),
            ("Vy", self.vy_col),
            ("Torsion", self.torsion_col),
            ("Vertical sign", self.vertical_sign),
            ("Moment sign", self.moment_sign),
        ]
        for i, (label, control) in enumerate(labels):
            grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            grid.addWidget(control, i // 2, (i % 2) * 2 + 1)
        layout.addWidget(map_group)

        filter_group = QGroupBox("CaseType filter")
        filter_group.setMinimumHeight(180)
        filter_layout = QVBoxLayout(filter_group)
        self.case_type_list = QListWidget()
        self.case_type_list.setMinimumHeight(120)
        self.case_type_list.setSelectionMode(QAbstractItemView.MultiSelection)
        filter_layout.addWidget(self.case_type_list)
        layout.addWidget(filter_group)

        joint_title = QLabel("Joint groups")
        joint_title.setObjectName("sectionTitle")
        layout.addWidget(joint_title)
        joint_caption = QLabel(
            "Create foundation-type groups and list SAP2000 joint numbers. "
            "Each selected Joint + OutputCase is designed separately, then the group is enveloped."
        )
        joint_caption.setWordWrap(True)
        joint_caption.setObjectName("subtitleLabel")
        layout.addWidget(joint_caption)

        self.joint_picker_group = QGroupBox("Add / update group by picking joints")
        self.joint_picker_group.setEnabled(False)
        picker_layout = QVBoxLayout(self.joint_picker_group)
        picker_form = QGridLayout()
        self.pick_group_name = QLineEdit("Group 1")
        self.pick_joint_list = QListWidget()
        self.pick_joint_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.pick_joint_list.setMinimumHeight(150)
        picker_form.addWidget(QLabel("Group name"), 0, 0)
        picker_form.addWidget(QLabel("Joint numbers"), 0, 1)
        picker_form.addWidget(self.pick_group_name, 1, 0)
        picker_form.addWidget(self.pick_joint_list, 1, 1)
        picker_form.setColumnStretch(0, 1)
        picker_form.setColumnStretch(1, 3)
        picker_layout.addLayout(picker_form)
        add_update_button = QPushButton("Add / update group")
        add_update_button.clicked.connect(self.add_update_group_from_picker)
        picker_button_row = QHBoxLayout()
        picker_button_row.addWidget(add_update_button)
        picker_button_row.addStretch(1)
        picker_layout.addLayout(picker_button_row)
        layout.addWidget(self.joint_picker_group)

        group_row = QHBoxLayout()
        for text, slot in [
            ("Apply geometry to groups", self.apply_geometry_to_sap_groups),
            ("Add group", self.add_sap_group_row),
            ("Duplicate selected", self.duplicate_sap_group_row),
            ("Remove selected", self.remove_sap_group_rows),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            group_row.addWidget(button)
        group_row.addStretch(1)
        layout.addLayout(group_row)

        self.sap_groups_model = PandasTableModel(eng.default_sap_groups(pd.DataFrame(), {}, self.current_geometry()), editable=True)
        self.sap_groups_table = self._table_view(self.sap_groups_model, min_height=150)
        layout.addWidget(QLabel("Foundation groups"))
        layout.addWidget(self.sap_groups_table, 2)

        self.sap_preview_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.sap_preview_table = self._table_view(self.sap_preview_model, min_height=220)
        layout.addWidget(QLabel("Normalized SAP2000 preview"))
        layout.addWidget(self.sap_preview_table, 2)
        return page

    def _results_tab(self) -> QWidget:
        page = QWidget()
        page.setMinimumWidth(1500)
        page.setMinimumHeight(760)
        layout = QVBoxLayout(page)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Selected group"))
        self.group_selector = QComboBox()
        self.group_selector.currentIndexChanged.connect(self.select_group_result)
        selector_row.addWidget(self.group_selector)
        selector_row.addStretch(1)
        layout.addLayout(selector_row)

        self.metric_grid = QGridLayout()
        self.metric_labels: Dict[str, QLabel] = {}
        for i, key in enumerate(["Overall max D/C", "Status", "Cap size X x Y", "Thickness h", "Pu used", "Max pile R"]):
            label = QLabel(f"{key}\n-")
            label.setProperty("metric", True)
            label.setMinimumHeight(54)
            label.setWordWrap(True)
            self.metric_labels[key] = label
            self.metric_grid.addWidget(label, 0, i)
        layout.addLayout(self.metric_grid)

        self.result_tables = QTabWidget()
        self.summary_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.checks_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.flex_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.detail_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.governing_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.pile_env_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.rc_pile_env_model = PandasTableModel(pd.DataFrame(), editable=False)
        for title, model in [
            ("Group Summary", self.summary_model),
            ("Strength / Service Checks", self.checks_model),
            ("Flexural Reinforcement", self.flex_model),
            ("Shear and Detailing", self.detail_model),
            ("Governing Combinations", self.governing_model),
            ("Service Pile Envelopes", self.pile_env_model),
            ("Ultimate Pile Envelopes", self.rc_pile_env_model),
        ]:
            self.result_tables.addTab(self._table_view(model), title)
        self.result_tables.setMinimumHeight(560)
        layout.addWidget(self.result_tables, 1)
        return self._scroll_page(page)

    def _calculation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.calculation_text = QTextEdit()
        self.calculation_text.setReadOnly(True)
        self.calculation_text.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.calculation_text)
        return page

    def _drawing_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        control_row = QHBoxLayout()
        self.show_rebar = QCheckBox("Show schematic bottom reinforcement")
        self.show_rebar.setChecked(True)
        self.show_reactions = QCheckBox("Show pile reactions on plan")
        self.show_reactions.setChecked(True)
        self.show_rebar.toggled.connect(self.update_drawing)
        self.show_reactions.toggled.connect(self.update_drawing)
        save_png = QPushButton("Save plan PNG")
        save_png.clicked.connect(self.save_plan_png)
        control_row.addWidget(self.show_rebar)
        control_row.addWidget(self.show_reactions)
        control_row.addStretch(1)
        control_row.addWidget(save_png)
        layout.addLayout(control_row)

        self.drawing_container = QWidget()
        self.drawing_layout = QVBoxLayout(self.drawing_container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.drawing_container)
        layout.addWidget(scroll, 1)
        return page

    def _stm_tab(self) -> QWidget:
        page = QWidget()
        page.setMinimumWidth(1300)
        page.setMinimumHeight(620)
        layout = QVBoxLayout(page)
        self.stm_notes = QLabel("")
        self.stm_notes.setWordWrap(True)
        layout.addWidget(self.stm_notes)
        self.stm_model = PandasTableModel(pd.DataFrame(), editable=False)
        self.stm_table = self._table_view(self.stm_model, min_height=520)
        layout.addWidget(self.stm_table, 1)
        return self._scroll_page(page)

    def _report_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        button_row = QHBoxLayout()
        for text, slot in [
            ("Save Markdown", self.save_markdown_report),
            ("Save PDF", self.save_pdf_report),
            ("Save editable state XLSX", self.save_state),
            ("Export group summary CSV", lambda: self.save_dataframe_csv("pile_cap_group_summary.csv", self.summary_model.dataframe)),
            ("Export checks CSV", lambda: self.save_dataframe_csv("pile_cap_checks.csv", self.checks_model.dataframe)),
            ("Export pile envelopes CSV", lambda: self.save_dataframe_csv("pile_reaction_envelopes.csv", self.pile_env_model.dataframe)),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            button_row.addWidget(button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.report_text, 1)
        return page

    def _basis_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "\n".join(
                [
                    "Basis / Assumptions",
                    "",
                    "This desktop version keeps the original Streamlit engineering logic and replaces only the UI layer.",
                    "Final design must be verified directly against ACI 318-25, the geotechnical report, project specifications, seismic requirements, and local regulations.",
                    "",
                    "Main assumptions:",
                    "- Geometry inputs are in mm.",
                    "- Forces are in kN and moments are in kN-m.",
                    "- Pile reactions use elastic rigid pile-cap distribution.",
                    "- Pile/geotechnical checks use service combinations where available.",
                    "- RC flexure, one-way shear, punching, bearing, detailing, STM advisory, exports, and drawings call the preserved engineering core.",
                    "- Imported SAP2000 rows are checked one-by-one by Joint + OutputCase and enveloped by foundation group.",
                ]
            )
        )
        layout.addWidget(text)
        return page

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _scroll_page(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        return scroll

    def _table_view(self, model: PandasTableModel, min_height: int = 320) -> QTableView:
        table = QTableView()
        table.setModel(model)
        table.setSortingEnabled(True)
        table.setAlternatingRowColors(True)
        if model.is_editable():
            table.setSelectionBehavior(QAbstractItemView.SelectItems)
            table.setEditTriggers(
                QAbstractItemView.DoubleClicked
                | QAbstractItemView.SelectedClicked
                | QAbstractItemView.EditKeyPressed
                | QAbstractItemView.AnyKeyPressed
            )
        else:
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setMinimumHeight(min_height)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setDefaultSectionSize(125)
        table.horizontalHeader().setMinimumSectionSize(90)
        table.horizontalHeader().setStretchLastSection(False)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        model.dataframeChanged.connect(lambda _df, table=table: self._apply_table_column_widths(table))
        self._apply_table_column_widths(table)
        return table

    def _apply_table_column_widths(self, table: QTableView) -> None:
        model = table.model()
        if not isinstance(model, PandasTableModel):
            return
        df = model.dataframe
        if df.empty:
            return
        for section, column in enumerate(df.columns):
            name = str(column).lower()
            width = 140
            if "note" in name:
                width = 950
            elif "check" in name:
                width = 260
            elif "joint ids" in name:
                width = 300
            elif "group name" in name:
                width = 190
            elif "outputcase" in name or "combo" in name or "combination" in name:
                width = 190
            elif "geometry warnings" in name or "source" in name:
                width = 260
            elif "pile tension cap" in name or "pile comp cap" in name:
                width = 190
            elif "required" in name or "provided" in name or "capacity" in name:
                width = 170
            elif "ratio" in name or "d/c" in name or "status" in name:
                width = 120
            table.setColumnWidth(section, width)

    @staticmethod
    def _selected_rows(table: QTableView) -> list[int]:
        selection = table.selectionModel()
        if selection is None:
            return []
        rows = {index.row() for index in selection.selectedRows()}
        rows.update(index.row() for index in selection.selectedIndexes())
        return sorted(rows)

    # ------------------------------------------------------------------
    # State builders
    # ------------------------------------------------------------------

    def current_material(self) -> eng.Material:
        return eng.Material(
            fc_MPa=self.fc_MPa.value(),
            fy_MPa=self.fy_MPa.value(),
            lambda_c=self.lambda_c.value(),
            gamma_conc_kN_m3=self.gamma_conc_kN_m3.value(),
            phi_flexure=self.phi_flexure.value(),
            phi_shear=self.phi_shear.value(),
            phi_bearing=self.phi_bearing.value(),
            phi_stm_tie=self.phi_stm_tie.value(),
        )

    def current_geometry(self) -> eng.Geometry:
        return eng.Geometry(
            n_piles=self.n_piles.value(),
            pile_diameter_mm=self.pile_diameter_mm.value(),
            pile_capacity_comp_kN=self.pile_capacity_comp_kN.value(),
            pile_capacity_tension_kN=self.pile_capacity_tension_kN.value(),
            cap_thickness_mm=self.cap_thickness_mm.value(),
            bottom_cover_mm=self.bottom_cover_mm.value(),
            top_cover_mm=self.top_cover_mm.value(),
            side_cover_mm=self.side_cover_mm.value(),
            column_bx_mm=self.column_bx_mm.value(),
            column_by_mm=self.column_by_mm.value(),
            pedestal_bx_mm=self.pedestal_bx_mm.value(),
            pedestal_by_mm=self.pedestal_by_mm.value(),
            edge_from_pile_edge_mm=self.edge_from_pile_edge_mm.value(),
            spacing_x_mm=self.spacing_x_mm.value(),
            spacing_y_mm=self.spacing_y_mm.value(),
            use_pedestal_for_shear=self.use_pedestal_for_shear.isChecked(),
            column_location=self.column_location.currentText(),
        )

    def current_reinforcement(self) -> eng.Reinforcement:
        return eng.Reinforcement(
            main_bar_x=self.main_bar_x.currentText(),
            main_bar_y=self.main_bar_y.currentText(),
            top_bar=self.top_bar.currentText(),
            spacing_x_mm=self.spacing_x_reinf_mm.value(),
            spacing_y_mm=self.spacing_y_reinf_mm.value(),
            top_spacing_mm=self.top_spacing_mm.value(),
            side_face_bar=self.side_face_bar.currentText(),
            side_face_spacing_mm=self.side_face_spacing_mm.value(),
            hook_extension_mm=self.hook_extension_mm.value(),
            preferred_spacing_step_mm=self.preferred_spacing_step_mm.value(),
        )

    def _switch_load_source(self, index: int) -> None:
        self.input_stack.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Manual and SAP input actions
    # ------------------------------------------------------------------

    def apply_geometry_to_manual_table(self) -> None:
        combined = self._manual_combined_dataframe()
        updated = eng.apply_sidebar_geometry_to_groups(combined, self.current_geometry())
        self._set_manual_tables(updated)

    def add_manual_row(self) -> None:
        row = eng.group_defaults_from_geometry(self.current_geometry(), f"Foundation {len(self.manual_load_model.dataframe) + 1}", "")
        row.update(eng.MANUAL_SERVICE_LOAD_DEFAULTS)
        combined = pd.concat([self._manual_combined_dataframe(), pd.DataFrame([row])], ignore_index=True)
        self._set_manual_tables(combined)

    def duplicate_manual_row(self) -> None:
        rows = self._selected_manual_rows()
        if rows:
            combined = self._manual_combined_dataframe()
            combined = pd.concat([combined, combined.iloc[[rows[0]]].copy()], ignore_index=True)
            self._set_manual_tables(combined)

    def remove_manual_rows(self) -> None:
        rows = self._selected_manual_rows()
        if not rows:
            return
        combined = self._manual_combined_dataframe().drop(self._manual_combined_dataframe().index[rows]).reset_index(drop=True)
        self._set_manual_tables(combined)

    def _selected_manual_rows(self) -> list[int]:
        rows = self._selected_rows(self.manual_table)
        if not rows:
            rows = self._selected_rows(self.manual_geometry_table)
        return rows

    def _manual_combined_dataframe(self) -> pd.DataFrame:
        loads = self.manual_load_model.dataframe.reset_index(drop=True)
        geometry = self.manual_geometry_model.dataframe.reset_index(drop=True)
        row_count = max(len(loads), len(geometry))
        loads = loads.reindex(range(row_count))
        geometry = geometry.reindex(range(row_count))
        combined = loads.copy()
        for col in geometry.columns:
            if col == "Group Name" and col in combined.columns:
                combined[col] = combined[col].fillna(geometry[col])
            else:
                combined[col] = geometry[col]
        combined = combined.fillna("")
        return eng.normalize_manual_service_load_columns(eng.ensure_group_columns(combined, self.current_geometry()))

    def _set_manual_tables(self, df: pd.DataFrame) -> None:
        df = eng.normalize_manual_service_load_columns(eng.ensure_group_columns(df, self.current_geometry()))
        self.manual_load_model.set_dataframe(self._manual_load_dataframe(df))
        self.manual_geometry_model.set_dataframe(self._manual_geometry_dataframe(df))

    def apply_geometry_to_sap_groups(self) -> None:
        self.sap_groups_model.set_dataframe(eng.apply_sidebar_geometry_to_groups(self.sap_groups_model.dataframe, self.current_geometry()))

    def add_update_group_from_picker(self) -> None:
        group_name = self.pick_group_name.text().strip() or "Unnamed Group"
        picked_joints = [
            self.pick_joint_list.item(i).text()
            for i in range(self.pick_joint_list.count())
            if self.pick_joint_list.item(i).isSelected()
        ]
        joint_ids = ", ".join(picked_joints)
        groups = eng.ensure_group_columns(self.sap_groups_model.dataframe, self.current_geometry())
        new_row = eng.group_defaults_from_geometry(self.current_geometry(), group_name, joint_ids)
        if "Group Name" in groups.columns and group_name in groups["Group Name"].astype(str).tolist():
            groups.loc[groups["Group Name"].astype(str) == group_name, "Joint IDs"] = joint_ids
        else:
            groups = pd.concat([groups, pd.DataFrame([new_row])], ignore_index=True)
        self.sap_groups_model.set_dataframe(eng.ensure_group_columns(groups, self.current_geometry()))
        self.statusBar().showMessage(f"Updated joint group '{group_name}' with {len(picked_joints)} joint(s)")

    def add_sap_group_row(self) -> None:
        row = eng.group_defaults_from_geometry(self.current_geometry(), f"Group {len(self.sap_groups_model.dataframe) + 1}", "")
        self.sap_groups_model.insert_blank_rows(1, row)

    def duplicate_sap_group_row(self) -> None:
        rows = self._selected_rows(self.sap_groups_table)
        if rows:
            self.sap_groups_model.copy_row(rows[0])

    def remove_sap_group_rows(self) -> None:
        self.sap_groups_model.remove_rows(self._selected_rows(self.sap_groups_table))

    def _populate_joint_picker(self, joints: list[str]) -> None:
        self.pick_joint_list.clear()
        for joint in joints:
            self.pick_joint_list.addItem(joint)
        self.joint_picker_group.setEnabled(bool(joints))

    def load_sap_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load SAP2000 reaction table", "", "Tables (*.csv *.xlsx *.xls)")
        if not path:
            return
        try:
            self.sap_df = eng.read_sap2000_table_path(path)
            self.sap_file_label.setText(str(path))
            self.sap_preview_model.set_dataframe(self.sap_df.head(200))
            self._populate_sap_mapping_controls()
            self.statusBar().showMessage(f"Loaded SAP2000 table with {len(self.sap_df):,} row(s)")
        except Exception as exc:
            self._error("Could not load SAP2000 table", str(exc))

    def _populate_sap_mapping_controls(self) -> None:
        columns = list(self.sap_df.columns)
        cmap = eng.sap2000_column_map(self.sap_df)

        def fill(combo: QComboBox, default: Optional[str], allow_none: bool = False) -> None:
            combo.blockSignals(True)
            combo.clear()
            if allow_none:
                combo.addItem("- none -")
            combo.addItems([str(c) for c in columns])
            if default and default in columns:
                combo.setCurrentText(str(default))
            combo.blockSignals(False)

        fill(self.vertical_col, cmap.get("F3"))
        fill(self.mx_col, cmap.get("M1"))
        fill(self.my_col, cmap.get("M2"))
        fill(self.vx_col, cmap.get("F1"), True)
        fill(self.vy_col, cmap.get("F2"), True)
        fill(self.torsion_col, cmap.get("M3"), True)
        self.sap_groups_model.set_dataframe(eng.default_sap_groups(self.sap_df, cmap, self.current_geometry()))

        self.case_type_list.clear()
        case_type_col = cmap.get("case_type")
        values = []
        if case_type_col and case_type_col in self.sap_df.columns:
            values = sorted({str(v) for v in self.sap_df[case_type_col].dropna().tolist() if str(v).strip()})
        for value in values:
            item = QListWidgetItem(value)
            item.setSelected(value.lower() == "combination")
            self.case_type_list.addItem(item)

        joint_col = cmap.get("joint")
        joints = []
        if joint_col and joint_col in self.sap_df.columns:
            joints = sorted({str(v).strip() for v in self.sap_df[joint_col].dropna().tolist() if str(v).strip()})
        self._populate_joint_picker(joints)

    # ------------------------------------------------------------------
    # Design run and result rendering
    # ------------------------------------------------------------------

    def run_design(self) -> None:
        mat = self.current_material()
        geom = self.current_geometry()
        reinf = self.current_reinforcement()
        include_self_weight = self.include_self_weight.isChecked()
        mode = "manual" if self.load_source.currentIndex() == 0 else "sap"

        if mode == "manual":
            payload = {
                "groups": self._manual_combined_dataframe(),
                "mat": mat,
                "base_geom": geom,
                "reinf": reinf,
                "include_self_weight": include_self_weight,
            }
        else:
            if self.sap_df.empty:
                self._error("No SAP2000 table", "Load a SAP2000 joint reaction CSV/XLSX before running SAP mode.")
                return
            selected_case_types = [
                self.case_type_list.item(i).text()
                for i in range(self.case_type_list.count())
                if self.case_type_list.item(i).isSelected()
            ]
            payload = {
                "groups": self.sap_groups_model.dataframe,
                "sap_df": self.sap_df,
                "mat": mat,
                "base_geom": geom,
                "reinf": reinf,
                "include_self_weight": include_self_weight,
                "vertical_col": self.vertical_col.currentText(),
                "mx_col": self.mx_col.currentText(),
                "my_col": self.my_col.currentText(),
                "vx_col": self._none_to_none(self.vx_col.currentText()),
                "vy_col": self._none_to_none(self.vy_col.currentText()),
                "torsion_col": self._none_to_none(self.torsion_col.currentText()),
                "vertical_sign": float(self.vertical_sign.currentText()),
                "moment_sign": float(self.moment_sign.currentText()),
                "selected_case_types": selected_case_types or None,
            }

        self.design_action.setEnabled(False)
        self.statusBar().showMessage("Design is running...")
        self.worker = DesignWorker(mode, payload, self)
        self.worker.succeeded.connect(self._design_finished)
        self.worker.failed.connect(self._design_failed)
        self.worker.finished.connect(lambda: self.design_action.setEnabled(True))
        self.worker.start()

    @staticmethod
    def _none_to_none(value: str) -> Optional[str]:
        return None if not value or value == "- none -" else value

    def _design_finished(self, batch: Dict[str, Any]) -> None:
        self.batch = batch
        self.summary_model.set_dataframe(batch.get("summary", pd.DataFrame()))
        self.pile_env_model.set_dataframe(batch.get("pile_envelopes", pd.DataFrame()))
        self.rc_pile_env_model.set_dataframe(batch.get("rc_pile_envelopes", pd.DataFrame()))
        self.group_selector.blockSignals(True)
        self.group_selector.clear()
        for item in batch.get("items", []):
            self.group_selector.addItem(item["group"])
        self.group_selector.blockSignals(False)
        self.group_selector.setCurrentIndex(0)
        self.select_group_result(0)
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"Design complete for {len(batch.get('items', [])):,} group(s)")

    def _design_failed(self, details: str) -> None:
        self.statusBar().showMessage("Design failed")
        self._error("Design failed", details)

    def select_group_result(self, index: int) -> None:
        if not isinstance(self.batch, dict) or not self.batch.get("items"):
            return
        if index < 0:
            index = 0
        self.current_item = self.batch["items"][index]
        state = self.current_item["state"]
        results = self.current_item["results"]
        self._render_selected_result(state, results)

    def _render_selected_result(self, state: eng.DesignState, results: Dict[str, Any]) -> None:
        metrics = view_data.metric_summary(state, results)
        for key, label in self.metric_labels.items():
            label.setText(f"{key}\n{metrics.get(key, '-')}")
        self.checks_model.set_dataframe(view_data.checks_dataframe(results))
        self.flex_model.set_dataframe(view_data.flexural_summary_dataframe(state, results))
        self.detail_model.set_dataframe(view_data.detail_summary_dataframe(state, results))
        self.governing_model.set_dataframe(view_data.governing_combos_dataframe(results))
        self.stm_model.set_dataframe(results.get("stm", pd.DataFrame()))
        notes = view_data.engineering_notes(results)
        self.stm_notes.setText("\n".join(f"- {note}" for note in notes))
        report = eng.make_markdown_report(state, results)
        self.report_text.setPlainText(report)
        self.calculation_text.setHtml(self._calculation_html(state, results, self.current_item))
        self.update_drawing()

    def _set_idle_result_state(self) -> None:
        message = "Prepare inputs and click DESIGN to generate results."
        self.calculation_text.setPlainText(message)
        self.report_text.setPlainText(message)
        self.stm_notes.setText(message)

    @staticmethod
    def _calculation_html(state: eng.DesignState, results: Dict[str, Any], item: Optional[Dict[str, Any]] = None) -> str:
        geom = state.geometry
        mat = state.material
        reinf = state.reinforcement
        checks = {check.name: check for check in results.get("checks", [])}

        def n(value: Any, nd: int = 2) -> str:
            try:
                value = float(value)
                if not math.isfinite(value):
                    return "-"
                return f"{value:,.{nd}f}"
            except Exception:
                return escape(str(value))

        def ratio_text(check: Optional[eng.CheckResult]) -> str:
            if check is None:
                return "-"
            try:
                return "-" if not math.isfinite(float(check.ratio)) else f"{float(check.ratio):.3f}"
            except Exception:
                return "-"

        def result_line(check_name: str) -> str:
            check = checks.get(check_name)
            if check is None:
                return ""
            return (
                "<div class='result'>"
                f"<b>Result:</b> Demand = {n(check.demand, 2)} {escape(check.unit)}, "
                f"Capacity = {n(check.capacity, 2)} {escape(check.unit)}, "
                f"D/C = {ratio_text(check)}, Status = <b>{escape(check.status)}</b>"
                "</div>"
            )

        def row(label: str, value: str, unit: str = "") -> str:
            return f"<tr><td>{escape(label)}</td><td>{value}</td><td>{escape(unit)}</td></tr>"

        def section(title: str, body: str) -> str:
            return f"<h2>{escape(title)}</h2>{body}"

        service_runs = item.get("service_runs", []) if isinstance(item, dict) else []

        def pile_capacity_ratio(run: Dict[str, Any]) -> float:
            run_state = run.get("state")
            if run_state is None:
                return 0.0
            max_comp_run = max([max(p.reaction_kN, 0.0) for p in run_state.piles] + [0.0])
            max_uplift_run = max([max(-p.reaction_kN, 0.0) for p in run_state.piles] + [0.0])
            return max(
                max_comp_run / max(run_state.geometry.pile_capacity_comp_kN, 1e-9),
                max_uplift_run / max(run_state.geometry.pile_capacity_tension_kN, 1e-9),
            )

        reaction_run = max(service_runs, key=pile_capacity_ratio) if service_runs else {"state": state, "results": results, "loadcase": state.loadcase.name}
        reaction_state = reaction_run.get("state", state)
        reaction_results = reaction_run.get("results", results)
        reaction_loadcase_name = str(reaction_run.get("loadcase", reaction_state.loadcase.name))

        xs_m = [eng.mm_to_m(p.x_mm) for p in reaction_state.piles]
        ys_m = [eng.mm_to_m(p.y_mm) for p in reaction_state.piles]
        sum_x2 = sum(x * x for x in xs_m)
        sum_y2 = sum(y * y for y in ys_m)
        max_comp = max([max(p.reaction_kN, 0.0) for p in reaction_state.piles] + [0.0])
        max_uplift = max([max(-p.reaction_kN, 0.0) for p in reaction_state.piles] + [0.0])
        pile_env_df = results.get("pile_reaction_envelope")
        if isinstance(pile_env_df, pd.DataFrame) and not pile_env_df.empty:
            max_comp = float(pd.to_numeric(pile_env_df.get("Max compression (kN)"), errors="coerce").max())
            max_uplift = float(pd.to_numeric(pile_env_df.get("Max uplift (kN)"), errors="coerce").max())

        pile_count = max(len(reaction_state.piles), 1)
        p_total = float(reaction_results.get("Pu_total_kN", reaction_state.loadcase.Pu_kN + reaction_state.self_weight_kN))
        base_reaction = p_total / pile_count
        cap_weight_calc = eng.self_weight_cap_kN(
            reaction_state.cap_length_x_mm,
            reaction_state.cap_width_y_mm,
            geom.cap_thickness_mm,
            mat.gamma_conc_kN_m3,
        )
        mx = float(reaction_state.loadcase.Mux_kNm)
        my = float(reaction_state.loadcase.Muy_kNm)
        pile_calc_rows = []
        pile_formula_rows = []
        for pile in reaction_state.piles:
            x_m = eng.mm_to_m(pile.x_mm)
            y_m = eng.mm_to_m(pile.y_mm)
            mx_component = mx * y_m / sum_y2 if abs(sum_y2) > 1e-12 else 0.0
            my_component = my * x_m / sum_x2 if abs(sum_x2) > 1e-12 else 0.0
            ri = base_reaction + mx_component + my_component
            mx_term = f"[{n(mx, 3)}({n(y_m, 3)}) / {n(sum_y2, 4)}]" if abs(sum_y2) > 1e-12 else "0.000 (all y = 0)"
            my_term = f"[{n(my, 3)}({n(x_m, 3)}) / {n(sum_x2, 4)}]" if abs(sum_x2) > 1e-12 else "0.000 (all x = 0)"
            pile_calc_rows.append(
                "<tr>"
                f"<td>{escape(pile.label)}</td>"
                f"<td>{n(pile.x_mm, 0)}</td>"
                f"<td>{n(pile.y_mm, 0)}</td>"
                f"<td>{n(x_m, 3)}</td>"
                f"<td>{n(y_m, 3)}</td>"
                f"<td>{n(base_reaction, 3)}</td>"
                f"<td>{n(mx_component, 3)}</td>"
                f"<td>{n(my_component, 3)}</td>"
                f"<td><b>{n(ri, 3)}</b></td>"
                "</tr>"
            )
            pile_formula_rows.append(
                "<tr>"
                f"<td>{escape(pile.label)}</td>"
                f"<td>R<sub>{escape(pile.label)}</sub> = {n(base_reaction, 3)} "
                f"+ {mx_term} "
                f"+ {my_term} "
                f"= <b>{n(ri, 3)} kN</b></td>"
                "</tr>"
            )

        common_rows = "".join(
            [
                row("RC governing load case", escape(state.loadcase.name), ""),
                row("RC P_u total", n(results["Pu_total_kN"], 2), "kN"),
                row("Pile reaction service load case", escape(reaction_loadcase_name), ""),
                row("Service P_s total", n(p_total, 2), "kN"),
                row("f'c", n(mat.fc_MPa, 1), "MPa"),
                row("f_y", n(mat.fy_MPa, 1), "MPa"),
                row("Cap X x Y x h", f"{n(state.cap_length_x_mm, 0)} x {n(state.cap_width_y_mm, 0)} x {n(geom.cap_thickness_mm, 0)}", "mm"),
                row("d bottom X / Y", f"{n(results['d_x_mm'], 0)} / {n(results['d_y_mm'], 0)}", "mm"),
                row("d top X / Y", f"{n(results['d_top_x_mm'], 0)} / {n(results['d_top_y_mm'], 0)}", "mm"),
                row("ACI close-pile provision", "YES" if results["close_pile_spacing"]["applies"] else "NO", ""),
            ]
        )

        pile_body = f"""
            <div class='formula'>R<sub>i</sub> =
            P<sub>s</sub>/n + M<sub>sx</sub> y<sub>i</sub> / &Sigma;y<sub>j</sub><sup>2</sup>
            + M<sub>sy</sub> x<sub>i</sub> / &Sigma;x<sub>j</sub><sup>2</sup></div>
            <p><b>Sign convention:</b> compression is positive. Coordinates x and y are converted from mm to m for the moment-distribution terms.</p>
            <p><b>Load level:</b> pile reactions for pile compression/uplift are calculated from service loads.
            Ultimate load effects are still used later for RC flexure, shear, punching, and bearing checks.</p>
            <p><b>Service case used here:</b> {escape(reaction_loadcase_name)}.</p>
            <h3>Step 1 - Total vertical load used for pile reactions</h3>
            <div class='formula'>W<sub>cap</sub> = L<sub>x</sub>L<sub>y</sub>h&gamma;<sub>c</sub></div>
            <p>W<sub>cap</sub> = ({n(reaction_state.cap_length_x_mm / 1000.0, 3)} m)({n(reaction_state.cap_width_y_mm / 1000.0, 3)} m)({n(geom.cap_thickness_mm / 1000.0, 3)} m)({n(mat.gamma_conc_kN_m3, 2)} kN/m<sup>3</sup>)
            = {n(cap_weight_calc, 3)} kN.</p>
            <p>Self-weight included in pile reaction distribution = {n(reaction_state.self_weight_kN, 3)} kN.</p>
            <div class='formula'>P<sub>s,total</sub> = P<sub>s</sub> + W<sub>cap</sub></div>
            <p>P<sub>s,total</sub> = {n(reaction_state.loadcase.Pu_kN, 3)} + {n(reaction_state.self_weight_kN, 3)} = {n(p_total, 3)} kN.</p>
            <h3>Step 2 - Pile coordinate properties</h3>
            <div class='formula'>&Sigma;x<sup>2</sup> = &Sigma;x<sub>i</sub><sup>2</sup>, &nbsp; &Sigma;y<sup>2</sup> = &Sigma;y<sub>i</sub><sup>2</sup></div>
            <p>n = {len(state.piles)}, &Sigma;x<sup>2</sup> = {n(sum_x2, 4)} m<sup>2</sup>,
            &Sigma;y<sup>2</sup> = {n(sum_y2, 4)} m<sup>2</sup>.</p>
            <h3>Step 3 - Direct vertical-load share</h3>
            <div class='formula'>P<sub>s,total</sub>/n = {n(p_total, 3)} / {pile_count} = {n(base_reaction, 3)} kN per pile</div>
            <h3>Step 4 - Moment distribution terms</h3>
            <div class='formula'>R<sub>Msx,i</sub> = M<sub>sx</sub>y<sub>i</sub> / &Sigma;y<sup>2</sup>, &nbsp;
            R<sub>Msy,i</sub> = M<sub>sy</sub>x<sub>i</sub> / &Sigma;x<sup>2</sup></div>
            <p>M<sub>sx</sub> = {n(mx, 3)} kN-m, M<sub>sy</sub> = {n(my, 3)} kN-m.</p>
            <h3>Step 5 - Per-pile reaction calculation</h3>
            <table>
                <tr>
                    <th>Pile</th><th>x (mm)</th><th>y (mm)</th><th>x (m)</th><th>y (m)</th>
                    <th>Ps/n (kN)</th><th>Msx y / &Sigma;y<sup>2</sup> (kN)</th><th>Msy x / &Sigma;x<sup>2</sup> (kN)</th><th>R_i (kN)</th>
                </tr>
                {''.join(pile_calc_rows)}
            </table>
            <h3>Step 6 - Formula substitution by pile</h3>
            <table>
                <tr><th>Pile</th><th>Substitution</th></tr>
                {''.join(pile_formula_rows)}
            </table>
            <p>Maximum compression envelope = {n(max_comp, 2)} kN.
            Maximum uplift envelope = {n(max_uplift, 2)} kN.</p>
            {result_line("Pile compression")}
            {result_line("Pile tension/uplift")}
        """

        def flexure_block(
            title: str,
            check_name: str,
            bar: str,
            spacing: Dict[str, Any],
            capacity: Dict[str, Any],
            flex: Dict[str, Any],
            width_mm: float,
            d_mm: float,
            note: str,
        ) -> str:
            check = checks.get(check_name)
            demand = check.demand if check else 0.0
            return section(
                title,
                f"""
                <div class='formula'>&phi;M<sub>n</sub> &ge; M<sub>u</sub></div>
                <div class='formula'>M<sub>n</sub> = A<sub>s</sub> f<sub>y</sub>(d - a/2)</div>
                <div class='formula'>a = A<sub>s</sub> f<sub>y</sub> / (0.85 f'c b)</div>
                <p>{escape(note)}</p>
                <ul>
                    <li>M<sub>u</sub> = {n(demand, 2)} kN-m.</li>
                    <li>b = {n(width_mm, 0)} mm, d = {n(d_mm, 0)} mm.</li>
                    <li>Provided reinforcement = {escape(bar)} @ {n(spacing['spacing_use_mm'], 0)} mm.</li>
                    <li>A<sub>s,prov</sub> = {n(spacing['As_prov_mm2'], 0)} mm<sup>2</sup>; A<sub>s,req</sub> = {n(flex['As_req_mm2'], 0)} mm<sup>2</sup>.</li>
                    <li>a = {n(capacity['a_mm'], 1)} mm.</li>
                    <li>&phi;M<sub>n</sub> = {n(capacity['phiMn_kNm'], 2)} kN-m.</li>
                </ul>
                {result_line(check_name)}
                """,
            )

        top_x = results["top_demand_x"]
        top_y = results["top_demand_y"]
        top_body = f"""
            <div class='formula'>M<sub>u,top</sub> =
            max(M<sup>-</sup><sub>continuous</sub>, &Sigma;T<sub>pile</sub>l)</div>
            <div class='formula'>&phi;M<sub>n</sub> = &phi;A<sub>s</sub>f<sub>y</sub>(d - a/2)</div>
            <table>
                <tr><th>Direction</th><th>Continuous M-</th><th>Uplift moment</th><th>Governing M_u</th><th>Source</th><th>&phi;M_n</th><th>D/C</th></tr>
                <tr>
                    <td>X bars</td><td>{n(top_x['continuous_negative_kNm'], 2)}</td><td>{n(top_x['uplift_tension_kNm'], 2)}</td>
                    <td>{n(top_x['demand_kNm'], 2)}</td><td>{escape(str(top_x['source']))}</td>
                    <td>{n(results['top_cap_xbars']['phiMn_kNm'], 2)}</td><td>{ratio_text(checks.get("Flexure - top X bars"))}</td>
                </tr>
                <tr>
                    <td>Y bars</td><td>{n(top_y['continuous_negative_kNm'], 2)}</td><td>{n(top_y['uplift_tension_kNm'], 2)}</td>
                    <td>{n(top_y['demand_kNm'], 2)}</td><td>{escape(str(top_y['source']))}</td>
                    <td>{n(results['top_cap_ybars']['phiMn_kNm'], 2)}</td><td>{ratio_text(checks.get("Flexure - top Y bars"))}</td>
                </tr>
            </table>
            {result_line("Flexure - top X bars")}
            {result_line("Flexure - top Y bars")}
        """

        def one_way_block(title: str, check_name: str, capacity_key: str, demand_key: str, width_mm: float, d_mm: float) -> str:
            cap = results[capacity_key]
            demand = results["shear_demands"][demand_key]
            if cap.get("use_close_pile_aci318_25"):
                vc_formula = "v<sub>c</sub> = 0.17 &lambda;&radic;f'c"
            else:
                vc_formula = (
                    "v<sub>c</sub> = min[0.66 &lambda;<sub>s</sub>&lambda;&rho;<sub>w</sub><sup>1/3</sup>&radic;f'c "
                    "+ N<sub>u</sub>/(6A<sub>g</sub>), 0.42 &lambda;&radic;f'c]"
                )
            return section(
                title,
                f"""
                <div class='formula'>V<sub>u</sub> = &Sigma;R<sub>u,i outside section</sub></div>
                <div class='formula'>&phi;V<sub>c</sub> = &phi;v<sub>c</sub>b<sub>w</sub>d</div>
                <div class='formula'>{vc_formula}</div>
                <p>{escape(results['close_pile_spacing']['note'])}</p>
                <ul>
                    <li>V<sub>u</sub> = {n(demand, 2)} kN.</li>
                    <li>b<sub>w</sub> = {n(width_mm, 0)} mm, d = {n(d_mm, 0)} mm.</li>
                    <li>&lambda;<sub>s</sub> = {n(cap['lambda_s'], 3)}, &rho;<sub>w</sub> = {n(cap['rho_w'], 5)}.</li>
                    <li>v<sub>c</sub> = {n(cap['vc_MPa'], 3)} MPa.</li>
                    <li>&phi;V<sub>c</sub> = {n(cap['phiVc_kN'], 2)} kN.</li>
                </ul>
                {result_line(check_name)}
                """,
            )

        pdm = results["punch_demand"]
        pc = results["punch_cap"]
        punching_body = f"""
            <div class='formula'>V<sub>u</sub> = P<sub>u</sub> - R<sub>inside</sub></div>
            <div class='formula'>&phi;V<sub>c</sub> = &phi;v<sub>c</sub>b<sub>o</sub>d</div>
            <div class='formula'>v<sub>c</sub> = min[
            0.33&lambda;<sub>s</sub>&lambda;&radic;f'c,
            (0.17 + 0.33/&beta;)&lambda;<sub>s</sub>&lambda;&radic;f'c,
            (0.17 + 0.083&alpha;<sub>s</sub>d/b<sub>o</sub>)&lambda;<sub>s</sub>&lambda;&radic;f'c]</div>
            <ul>
                <li>b<sub>o</sub> = {n(pdm['bo_mm'], 0)} mm, d<sub>avg</sub> = {n(results['d_avg_mm'], 0)} mm.</li>
                <li>&beta; = {n(pc['beta'], 3)}, &alpha;<sub>s</sub> = {n(pc['alpha_s'], 1)}, &lambda;<sub>s</sub> = {n(pc['lambda_s'], 3)}.</li>
                <li>R<sub>inside</sub> = {n(pdm['R_inside_kN'], 2)} kN, P<sub>u</sub> = {n(results['Pu_total_kN'], 2)} kN.</li>
                <li>V<sub>u</sub> = {n(pdm['Vu_punch_kN'], 2)} kN.</li>
                <li>Governing v<sub>c</sub> equation = {escape(str(pc['governing_equation']))}; v<sub>c</sub> = {n(pc['vc_MPa'], 3)} MPa.</li>
                <li>&phi;V<sub>c</sub> = {n(pc['phiVc_kN'], 2)} kN.</li>
            </ul>
            {result_line("Two-way punching shear")}
        """

        loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
        loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm
        area = loaded_bx * loaded_by
        bearing_body = f"""
            <div class='formula'>&phi;P<sub>n</sub> = &phi;(0.85 f'c A<sub>1</sub>)</div>
            <ul>
                <li>A<sub>1</sub> = {n(loaded_bx, 0)} x {n(loaded_by, 0)} = {n(area, 0)} mm<sup>2</sup>.</li>
                <li>P<sub>u</sub> = {n(results['Pu_total_kN'], 2)} kN.</li>
                <li>&phi;P<sub>n</sub> = {n(results['bearing']['phiPn_kN'], 2)} kN.</li>
            </ul>
            {result_line("Column/pedestal bearing")}
        """

        development_body = f"""
            <div class='formula'>l<sub>d</sub> =
            [f<sub>y</sub>&psi;<sub>t</sub>&psi;<sub>e</sub>&psi;<sub>s</sub> /
            (1.1&lambda;&radic;f'c)] d<sub>b</sub> / [(c<sub>b</sub> + K<sub>tr</sub>)/d<sub>b</sub>]</div>
            <table>
                <tr><th>Bars</th><th>d_b</th><th>c_b</th><th>(c_b+K_tr)/d_b</th><th>l_d</th></tr>
                <tr><td>Bottom X</td><td>{n(eng.bar_diameter(reinf.main_bar_x), 0)}</td><td>{n(results['ld_x']['cb_mm'], 1)}</td><td>{n(results['ld_x']['confinement_factor'], 2)}</td><td>{n(results['ld_x_mm'], 0)}</td></tr>
                <tr><td>Bottom Y</td><td>{n(eng.bar_diameter(reinf.main_bar_y), 0)}</td><td>{n(results['ld_y']['cb_mm'], 1)}</td><td>{n(results['ld_y']['confinement_factor'], 2)}</td><td>{n(results['ld_y_mm'], 0)}</td></tr>
                <tr><td>Top</td><td>{n(eng.bar_diameter(reinf.top_bar), 0)}</td><td>{n(results['ld_top']['cb_mm'], 1)}</td><td>{n(results['ld_top']['confinement_factor'], 2)}</td><td>{n(results['ld_top_mm'], 0)}</td></tr>
            </table>
        """

        html = f"""
        <html>
        <head>
        <style>
            body {{ color: #f8fafc; background: #000000; font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; }}
            h1 {{ font-size: 20px; margin: 4px 0 8px 0; }}
            h2 {{ font-size: 15px; margin: 18px 0 6px 0; color: #f8fafc; }}
            p, li {{ line-height: 1.45; }}
            table {{ border-collapse: collapse; margin: 6px 0 10px 0; width: 100%; }}
            th, td {{ border: 1px solid #334155; padding: 5px 7px; }}
            th {{ background: #1f2937; color: #f8fafc; }}
            .formula {{ font-family: "Cambria Math", "Times New Roman", serif; font-size: 15px; color: #f8fafc; background: #111827; border: 1px solid #475569; padding: 7px 9px; margin: 5px 0; }}
            .result {{ color: #dcfce7; background: #052e16; border: 1px solid #166534; padding: 7px 9px; margin: 8px 0; }}
            .note {{ color: #cbd5e1; }}
        </style>
        </head>
        <body>
            <h1>ACI-318 Style Calculation</h1>
            <p class='note'>Equations are written in engineering notation using the values from the selected governing design result. Verify final clause references and project modifiers directly against ACI 318 and local requirements.</p>
            {section("Common Inputs", f"<table>{common_rows}</table>")}
            {section("Pile Reaction Distribution", pile_body)}
            {flexure_block("Flexure - Bottom X Bars", "Flexure - bottom X bars", reinf.main_bar_x, results["spacing_x"], results["cap_xbars"], results["flex_x"], state.cap_width_y_mm, results["d_x_mm"], "Bottom bars parallel to X; cantilever action in Y.")}
            {flexure_block("Flexure - Bottom Y Bars", "Flexure - bottom Y bars", reinf.main_bar_y, results["spacing_y"], results["cap_ybars"], results["flex_y"], state.cap_length_x_mm, results["d_y_mm"], "Bottom bars parallel to Y; cantilever action in X.")}
            {section("Flexure - Top Bars", top_body)}
            {one_way_block("One-Way Shear - Section Normal to Y", "One-way shear - section normal to Y", "one_way_x", "V_for_X_bars_direction_kN", state.cap_length_x_mm, results["d_x_mm"])}
            {one_way_block("One-Way Shear - Section Normal to X", "One-way shear - section normal to X", "one_way_y", "V_for_Y_bars_direction_kN", state.cap_width_y_mm, results["d_y_mm"])}
            {section("Two-Way Punching Shear", punching_body)}
            {section("Column / Pedestal Bearing", bearing_body)}
            {section("Development Length Estimate", development_body)}
        </body>
        </html>
        """
        return html

    # ------------------------------------------------------------------
    # Drawings
    # ------------------------------------------------------------------

    def update_drawing(self) -> None:
        if not self.current_item:
            return
        state = self.current_item["state"]
        results = self.current_item["results"]
        while self.drawing_layout.count():
            item = self.drawing_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for title, fig in [
            (
                "Plan",
                eng.plot_plan(
                    state,
                    results,
                    show_rebar=self.show_rebar.isChecked(),
                    show_reactions=self.show_reactions.isChecked(),
                ),
            ),
            ("Elevation X", eng.plot_elevation(state, results, "X")),
            ("Elevation Y", eng.plot_elevation(state, results, "Y")),
        ]:
            self.drawing_layout.addWidget(QLabel(title))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            self.drawing_layout.addWidget(toolbar)
            self.drawing_layout.addWidget(canvas)
            if title == "Plan":
                self.plan_canvas = canvas
            elif title == "Elevation X":
                self.elev_x_canvas = canvas
            else:
                self.elev_y_canvas = canvas

    def save_plan_png(self) -> None:
        if not self.current_item:
            self._error("No drawing", "Run DESIGN before saving the plan drawing.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save plan PNG", "pile_cap_plan.png", "PNG (*.png)")
        if not path:
            return
        state = self.current_item["state"]
        results = self.current_item["results"]
        fig = eng.plot_plan(state, results, self.show_rebar.isChecked(), self.show_reactions.isChecked())
        fig.savefig(path, dpi=180, bbox_inches="tight")
        self.statusBar().showMessage(f"Saved {path}")

    # ------------------------------------------------------------------
    # Save / load / exports
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load editable state", "", "Excel workbook (*.xlsx)")
        if not path:
            return
        try:
            payload = eng.read_saved_state_xlsx(path)
            if not payload:
                return
            self._apply_payload_to_controls(payload)
            groups = payload.get("groups", pd.DataFrame())
            if isinstance(groups, pd.DataFrame) and not groups.empty:
                self._set_manual_tables(eng.ensure_group_columns(groups, self.current_geometry()))
                self.sap_groups_model.set_dataframe(eng.ensure_group_columns(groups, self.current_geometry()))
            sap_df = payload.get("sap2000_import", pd.DataFrame())
            if isinstance(sap_df, pd.DataFrame) and not sap_df.empty:
                self.sap_df = sap_df
                self.sap_preview_model.set_dataframe(sap_df.head(200))
                self.sap_file_label.setText(f"Loaded from state: {len(sap_df):,} row(s)")
                self._populate_sap_mapping_controls()
            piles = payload.get("piles", pd.DataFrame())
            if isinstance(piles, pd.DataFrame) and not piles.empty:
                self.pile_preview_model.set_dataframe(piles)
            else:
                self._refresh_pile_preview()
            self.statusBar().showMessage(f"Loaded {path}")
        except Exception as exc:
            self._error("Could not load state", str(exc))

    def _apply_payload_to_controls(self, payload: Dict[str, Any]) -> None:
        mat = eng.material_from_payload(payload)
        geom = eng.geometry_from_payload(payload)
        reinf = eng.reinforcement_from_payload(payload)
        self.include_self_weight.setChecked(eng.include_self_weight_from_payload(payload, True))

        for control, value in [
            (self.fc_MPa, mat.fc_MPa),
            (self.fy_MPa, mat.fy_MPa),
            (self.lambda_c, mat.lambda_c),
            (self.gamma_conc_kN_m3, mat.gamma_conc_kN_m3),
            (self.phi_flexure, mat.phi_flexure),
            (self.phi_shear, mat.phi_shear),
            (self.phi_bearing, mat.phi_bearing),
            (self.phi_stm_tie, mat.phi_stm_tie),
            (self.pile_diameter_mm, geom.pile_diameter_mm),
            (self.pile_capacity_comp_kN, geom.pile_capacity_comp_kN),
            (self.pile_capacity_tension_kN, geom.pile_capacity_tension_kN),
            (self.cap_thickness_mm, geom.cap_thickness_mm),
            (self.bottom_cover_mm, geom.bottom_cover_mm),
            (self.top_cover_mm, geom.top_cover_mm),
            (self.side_cover_mm, geom.side_cover_mm),
            (self.column_bx_mm, geom.column_bx_mm),
            (self.column_by_mm, geom.column_by_mm),
            (self.pedestal_bx_mm, geom.pedestal_bx_mm),
            (self.pedestal_by_mm, geom.pedestal_by_mm),
            (self.edge_from_pile_edge_mm, geom.edge_from_pile_edge_mm),
            (self.spacing_x_mm, geom.spacing_x_mm),
            (self.spacing_y_mm, geom.spacing_y_mm),
            (self.spacing_x_reinf_mm, reinf.spacing_x_mm),
            (self.spacing_y_reinf_mm, reinf.spacing_y_mm),
            (self.top_spacing_mm, reinf.top_spacing_mm),
            (self.side_face_spacing_mm, reinf.side_face_spacing_mm),
            (self.hook_extension_mm, reinf.hook_extension_mm),
            (self.preferred_spacing_step_mm, reinf.preferred_spacing_step_mm),
        ]:
            control.setValue(value)
        self.n_piles.setValue(geom.n_piles)
        self.use_pedestal_for_shear.setChecked(geom.use_pedestal_for_shear)
        self.column_location.setCurrentText(geom.column_location)
        self.main_bar_x.setCurrentText(reinf.main_bar_x)
        self.main_bar_y.setCurrentText(reinf.main_bar_y)
        self.top_bar.setCurrentText(reinf.top_bar)
        self.side_face_bar.setCurrentText(reinf.side_face_bar)

    def save_state(self) -> None:
        if not self.current_item:
            self._error("No design results", "Run DESIGN before saving the editable state workbook.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save editable state", "pile_foundation_state.xlsx", "Excel workbook (*.xlsx)")
        if not path:
            return
        state = self.current_item["state"]
        results = self.current_item["results"]
        groups_df = self.batch.get("groups", pd.DataFrame()) if isinstance(self.batch, dict) else pd.DataFrame()
        data = eng.make_saved_state_xlsx(
            state,
            results,
            self.include_self_weight.isChecked(),
            sap_df=self.sap_df,
            groups_df=groups_df,
            batch_design=self.batch,
        )
        Path(path).write_bytes(data)
        self.statusBar().showMessage(f"Saved {path}")

    def save_markdown_report(self) -> None:
        if not self.current_item:
            self._error("No report", "Run DESIGN before saving the report.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Markdown report", "pile_foundation_report.md", "Markdown (*.md)")
        if not path:
            return
        Path(path).write_text(self.report_text.toPlainText(), encoding="utf-8")
        self.statusBar().showMessage(f"Saved {path}")

    def save_pdf_report(self) -> None:
        if not self.current_item:
            self._error("No report", "Run DESIGN before saving the PDF.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save calculation PDF", "pile_foundation_calculation.pdf", "PDF (*.pdf)")
        if not path:
            return
        state = self.current_item["state"]
        results = self.current_item["results"]
        Path(path).write_bytes(eng.make_calculation_pdf(state, results))
        self.statusBar().showMessage(f"Saved {path}")

    def save_dataframe_csv(self, default_name: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            self._error("No table data", "Run DESIGN first or select a table with data.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV (*.csv)")
        if not path:
            return
        Path(path).write_bytes(view_data.dataframe_to_csv_bytes(df))
        self.statusBar().showMessage(f"Saved {path}")

    # ------------------------------------------------------------------
    # Preview and messaging
    # ------------------------------------------------------------------

    def _refresh_pile_preview(self) -> None:
        try:
            geom = self.current_geometry()
            piles = eng.make_piles(geom.n_piles, geom.spacing_x_mm, geom.spacing_y_mm, geom.pile_diameter_mm)
            df = eng.piles_to_dataframe(piles)
            self.pile_preview_model.set_dataframe(df)
            cap_x, cap_y = eng.cap_dimensions_from_piles(piles, geom.edge_from_pile_edge_mm)
            sw = eng.self_weight_cap_kN(cap_x, cap_y, geom.cap_thickness_mm, self.gamma_conc_kN_m3.value())
            warnings = eng.pile_spacing_warnings(geom)
            warning_text = " | ".join(warnings) if warnings else "Spacing checks: no warning"
            self.pile_preview_info.setText(f"Cap preview: {cap_x:,.0f} x {cap_y:,.0f} mm, self weight {sw:,.1f} kN. {warning_text}")
        except Exception as exc:
            self.pile_preview_info.setText(str(exc))

    def _error(self, title: str, message: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(title)
        box.setText(title)
        box.setDetailedText(message)
        box.exec()


def run() -> None:
    app = QApplication.instance() or QApplication([])
    window = PileFoundationWindow()
    window.show()
    app.exec()
