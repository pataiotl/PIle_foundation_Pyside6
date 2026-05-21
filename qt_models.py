from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor


class PandasTableModel(QAbstractTableModel):
    """A lightweight editable model for pandas DataFrames."""

    dataframeChanged = Signal(pd.DataFrame)

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        editable: bool = False,
        parent=None,
        header_aliases: Optional[dict[str, str]] = None,
    ):
        super().__init__(parent)
        self._df = dataframe.copy() if isinstance(dataframe, pd.DataFrame) else pd.DataFrame()
        self._editable = editable
        self._status_colors = True
        self._header_aliases = header_aliases or {}

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def set_dataframe(self, dataframe: Optional[pd.DataFrame]) -> None:
        self.beginResetModel()
        self._df = dataframe.copy() if isinstance(dataframe, pd.DataFrame) else pd.DataFrame()
        self.endResetModel()
        self.dataframeChanged.emit(self.dataframe)

    def set_editable(self, editable: bool) -> None:
        self._editable = editable

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or self._df.empty:
            return None
        value = self._df.iat[index.row(), index.column()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if pd.isna(value):
                return ""
            if role == Qt.DisplayRole and isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    return "-"
                if abs(float(value)) >= 100:
                    return f"{float(value):,.1f}"
                return f"{float(value):,.3f}"
            return str(value) if role == Qt.DisplayRole else value
        if role == Qt.TextAlignmentRole:
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        if role == Qt.ForegroundRole:
            return QColor(17, 24, 39)
        if role == Qt.BackgroundRole and self._status_colors:
            return self._background_for_row(index.row()) or QColor(248, 250, 252)
        return None

    def _background_for_row(self, row: int) -> Optional[QColor]:
        if row < 0 or row >= len(self._df):
            return None
        record = self._df.iloc[row]
        status = str(record.get("Status", "")).upper()
        ratio_values = []
        for col in self._df.columns:
            name = str(col).lower()
            if "ratio" in name or "d/c" in name or name == "max d/c":
                try:
                    value = float(str(record.get(col, "")).replace(",", ""))
                    if np.isfinite(value):
                        ratio_values.append(value)
                except Exception:
                    pass
        ratio = max(ratio_values) if ratio_values else np.nan
        if status == "FAIL" or (np.isfinite(ratio) and ratio > 1.0):
            return QColor(255, 225, 225)
        if status in {"NEAR", "STM", "STM REVIEW"} or (np.isfinite(ratio) and ratio >= 0.95):
            return QColor(255, 246, 211)
        if status == "PASS" or (np.isfinite(ratio) and ratio < 0.95):
            return QColor(230, 248, 235)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section < len(self._df.columns):
                column = str(self._df.columns[section])
                return self._header_aliases.get(column, column)
            return ""
        return str(section + 1)

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        if self._df.empty or column < 0 or column >= len(self._df.columns):
            return
        col = self._df.columns[column]
        ascending = order == Qt.AscendingOrder
        self.layoutAboutToBeChanged.emit()
        try:
            numeric = pd.to_numeric(self._df[col], errors="coerce")
            if numeric.notna().any():
                self._df = self._df.assign(_sort_key=numeric).sort_values("_sort_key", ascending=ascending).drop(columns="_sort_key")
            else:
                self._df = self._df.sort_values(col, ascending=ascending, key=lambda s: s.astype(str))
        except Exception:
            self._df = self._df.sort_index(ascending=ascending)
        self._df = self._df.reset_index(drop=True)
        self.layoutChanged.emit()

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        base = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if self._editable and index.isValid():
            return base | Qt.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if not self._editable or role != Qt.EditRole or not index.isValid():
            return False
        old_value = self._df.iat[index.row(), index.column()]
        self._df.iat[index.row(), index.column()] = self._coerce_value(value, old_value)
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        self.dataframeChanged.emit(self.dataframe)
        return True

    @staticmethod
    def _coerce_value(value: Any, old_value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return ""
            if isinstance(old_value, (int, float, np.integer, np.floating)) and not isinstance(old_value, bool):
                try:
                    parsed = float(text.replace(",", ""))
                    if isinstance(old_value, (int, np.integer)):
                        return int(round(parsed))
                    return parsed
                except Exception:
                    return value
        return value

    def insert_blank_rows(self, count: int = 1, defaults: Optional[dict[str, Any]] = None) -> None:
        defaults = defaults or {}
        start = len(self._df)
        self.beginInsertRows(QModelIndex(), start, start + count - 1)
        new_rows = pd.DataFrame([defaults.copy() for _ in range(count)])
        self._df = pd.concat([self._df, new_rows], ignore_index=True)
        self.endInsertRows()
        self.dataframeChanged.emit(self.dataframe)

    def remove_rows(self, rows: Iterable[int]) -> None:
        rows = sorted({r for r in rows if 0 <= r < len(self._df)}, reverse=True)
        if not rows:
            return
        self.beginResetModel()
        self._df = self._df.drop(self._df.index[rows]).reset_index(drop=True)
        self.endResetModel()
        self.dataframeChanged.emit(self.dataframe)

    def copy_row(self, row: int) -> None:
        if row < 0 or row >= len(self._df):
            return
        self.beginResetModel()
        self._df = pd.concat([self._df, self._df.iloc[[row]].copy()], ignore_index=True)
        self.endResetModel()
        self.dataframeChanged.emit(self.dataframe)
