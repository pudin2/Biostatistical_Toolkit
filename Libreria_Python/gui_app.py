import contextlib
import datetime as _dt
import io
import json
import os
import queue
import re
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Load import load_dataframe_from_path
from Caracterization import distribution_plots_from_loaded, normality_tests_from_loaded
from KDE import kde_from_loaded
from Kruscall_Wallis import kruskal_wallis_from_loaded
from Mann_Whitney import mann_whitney_from_loaded
from DB_Scan import dbscan_from_loaded


APP_TITLE = "Microbiota Statistical Workbench"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs_gui"


def sanitize_name(value, fallback="artifact"):
    text = str(value).strip()
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.ASCII)
    text = text.strip("._")
    return text[:120] or fallback


def unique_name(name, existing):
    base = sanitize_name(name, "dataset")
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def split_list(text):
    if text is None:
        return None
    items = [x.strip() for x in str(text).replace(";", ",").split(",")]
    items = [x for x in items if x]
    return items or None


def parse_optional_float(text):
    text = str(text).strip()
    if not text:
        return None
    return float(text)


def parse_bool(value):
    return bool(value)


def parse_tuple(text):
    items = split_list(text)
    return tuple(items) if items else None


def parse_json_dict(text):
    text = str(text).strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Debe ser un objeto JSON, por ejemplo: {\"perplexity\": 20}")
    return data


def parse_bandwidths(text):
    text = str(text).strip()
    if not text:
        return None
    if text.startswith("{"):
        data = json.loads(text)
        return {str(k): float(v) for k, v in data.items()}

    result = {}
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("Usa kernel=valor, por ejemplo gaussian=1.5")
        key, value = part.split("=", 1)
        result[key.strip()] = float(value.strip())
    return result or None


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        safe = {}
        for k, v in value.items():
            if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
                safe[str(k)] = f"<{type(v).__name__} exported separately>"
            else:
                safe[str(k)] = json_safe(v)
        return safe
    if isinstance(value, (list, tuple)):
        if len(value) > 100:
            return f"<{type(value).__name__} length={len(value)} exported separately>"
        return [json_safe(v) for v in value]
    return repr(value)


class FigureCapture:
    def __init__(self, figure_dir):
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.saved = []
        self.counter = 0
        self._old_show = None

    def __enter__(self):
        plt.switch_backend("Agg")
        fig = plt.figure(figsize=(0.1, 0.1))
        plt.close(fig)
        self._old_show = plt.show
        plt.show = self.show
        return self

    def __exit__(self, exc_type, exc, tb):
        self.save_open_figures()
        plt.show = self._old_show
        plt.close("all")

    def _title_for(self, fig):
        if getattr(fig, "_suptitle", None) is not None:
            text = fig._suptitle.get_text()
            if text:
                return text
        for ax in fig.axes:
            text = ax.get_title()
            if text:
                return text
        return "figure"

    def save_open_figures(self):
        for num in list(plt.get_fignums()):
            fig = plt.figure(num)
            self.counter += 1
            name = sanitize_name(self._title_for(fig), "figure")
            path = self.figure_dir / f"{self.counter:02d}_{name}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            self.saved.append(path)
            plt.close(fig)

    def show(self, *args, **kwargs):
        self.save_open_figures()


class ArtifactExporter:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.tables_dir = self.run_dir / "tables"
        self.arrays_dir = self.run_dir / "arrays"
        self.objects_dir = self.run_dir / "objects"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {"tables": [], "arrays": [], "objects": []}
        self.excel_tables = []

    def export(self, obj, prefix="result"):
        self._export_obj(obj, sanitize_name(prefix, "result"))
        self._write_excel_book()
        return self.manifest

    def _export_obj(self, obj, prefix):
        if isinstance(obj, pd.DataFrame):
            path = self.tables_dir / f"{prefix}.csv"
            obj.to_csv(path, index=False, encoding="utf-8-sig")
            self.excel_tables.append((prefix, obj))
            self.manifest["tables"].append({"name": prefix, "path": str(path), "rows": int(obj.shape[0]), "columns": int(obj.shape[1])})
            return

        if isinstance(obj, pd.Series):
            path = self.tables_dir / f"{prefix}.csv"
            obj.to_frame().to_csv(path, index=True, encoding="utf-8-sig")
            self.excel_tables.append((prefix, obj.to_frame()))
            self.manifest["tables"].append({"name": prefix, "path": str(path), "rows": int(obj.shape[0]), "columns": 1})
            return

        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj)
            if arr.ndim <= 2:
                path = self.arrays_dir / f"{prefix}.csv"
                pd.DataFrame(arr).to_csv(path, index=False, encoding="utf-8-sig")
            else:
                path = self.arrays_dir / f"{prefix}.npy"
                np.save(path, arr)
            self.manifest["arrays"].append({"name": prefix, "path": str(path), "shape": list(arr.shape)})
            return

        if isinstance(obj, dict):
            scalar_items = {}
            for key, value in obj.items():
                child_prefix = sanitize_name(f"{prefix}_{key}", prefix)
                if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, dict, list, tuple)):
                    self._export_obj(value, child_prefix)
                else:
                    scalar_items[str(key)] = json_safe(value)
            if scalar_items:
                self._write_json(prefix, scalar_items)
            return

        if isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj, start=1):
                self._export_obj(value, sanitize_name(f"{prefix}_{i:02d}", prefix))
            return

        self._write_json(prefix, json_safe(obj))

    def _write_json(self, name, payload):
        path = self.objects_dir / f"{sanitize_name(name)}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        self.manifest["objects"].append({"name": name, "path": str(path)})

    def _write_excel_book(self):
        if not self.excel_tables:
            return
        path = self.run_dir / "tables.xlsx"
        try:
            with pd.ExcelWriter(path) as writer:
                used = set()
                for name, df in self.excel_tables:
                    sheet = sanitize_name(name, "sheet")[:31] or "sheet"
                    base = sheet
                    i = 2
                    while sheet in used:
                        suffix = f"_{i}"
                        sheet = f"{base[:31 - len(suffix)]}{suffix}"
                        i += 1
                    used.add(sheet)
                    df.to_excel(writer, sheet_name=sheet, index=False)
            self.manifest["excel_workbook"] = str(path)
        except Exception as exc:
            self.manifest["excel_workbook_error"] = str(exc)


class ScrollFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)


class MicrobiotaGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x820")
        self.minsize(1120, 720)

        self.dfs = {}
        self.results = {}
        self.last_run_dir = None
        self.worker = None
        self.msg_queue = queue.Queue()
        self.inputs = {}
        self.df_combos = []
        self.column_combos = []

        self._configure_style()
        self._build_ui()
        self.after(150, self._poll_queue)

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f5f6f8")
        style.configure("Sidebar.TFrame", background="#eef1f4")
        style.configure("Header.TLabel", background="#f5f6f8", foreground="#20242a", font=("Segoe UI", 16, "bold"))
        style.configure("Subtle.TLabel", background="#f5f6f8", foreground="#5d6673")
        style.configure("TLabelframe", background="#f5f6f8")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=24)

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, minsize=340)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root, style="Sidebar.TFrame", padding=12)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(5, weight=1)

        ttk.Label(sidebar, text="Microbiota Workbench", font=("Segoe UI", 15, "bold"), background="#eef1f4").grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, text="Datos en memoria, exportes por corrida", background="#eef1f4", foreground="#5d6673").grid(row=1, column=0, sticky="w", pady=(2, 12))

        data_box = ttk.LabelFrame(sidebar, text="Datasets cargados", padding=8)
        data_box.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        data_box.grid_columnconfigure(0, weight=1)

        btns = ttk.Frame(data_box)
        btns.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        btns.grid_columnconfigure((0, 1, 2), weight=1)
        ttk.Button(btns, text="Cargar", command=self.load_files).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(btns, text="Vista", command=self.preview_selected_dataset).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(btns, text="Quitar", command=self.remove_selected_dataset).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.dataset_tree = ttk.Treeview(data_box, columns=("shape",), show="tree headings", height=8)
        self.dataset_tree.heading("#0", text="Nombre")
        self.dataset_tree.heading("shape", text="Shape")
        self.dataset_tree.column("#0", width=190, stretch=True)
        self.dataset_tree.column("shape", width=95, anchor="center", stretch=False)
        self.dataset_tree.grid(row=1, column=0, sticky="nsew")

        output_box = ttk.LabelFrame(sidebar, text="Salida", padding=8)
        output_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        output_box.grid_columnconfigure(0, weight=1)
        self.output_dir_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        ttk.Entry(output_box, textvariable=self.output_dir_var).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        output_btns = ttk.Frame(output_box)
        output_btns.grid(row=1, column=0, sticky="ew")
        output_btns.grid_columnconfigure((0, 1), weight=1)
        ttk.Button(output_btns, text="Cambiar", command=self.choose_output_dir).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(output_btns, text="Abrir", command=self.open_output_dir).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        memory_box = ttk.LabelFrame(sidebar, text="Resultados en memoria", padding=8)
        memory_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        memory_box.grid_columnconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(memory_box, columns=("time",), show="tree headings", height=5)
        self.result_tree.heading("#0", text="Analisis")
        self.result_tree.heading("time", text="Hora")
        self.result_tree.column("#0", width=180, stretch=True)
        self.result_tree.column("time", width=85, stretch=False)
        self.result_tree.grid(row=0, column=0, sticky="ew")

        log_box = ttk.LabelFrame(sidebar, text="Log", padding=8)
        log_box.grid(row=5, column=0, sticky="nsew")
        log_box.grid_columnconfigure(0, weight=1)
        log_box.grid_rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_box, height=12, wrap="word", relief="flat", bg="#ffffff", fg="#20242a")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        main = ttk.Frame(root, padding=16)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        ttk.Label(main, text="Panel de analisis", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="Escoge parametros, ejecuta y guarda tablas/figuras automaticamente.", style="Subtle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 12))

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=2, column=0, sticky="nsew")

        self._build_characterization_tab()
        self._build_normality_tab()
        self._build_kde_tab()
        self._build_kruskal_tab()
        self._build_mann_whitney_tab()
        self._build_dbscan_tab()

        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(main, textvariable=self.status_var, style="Subtle.TLabel").grid(row=3, column=0, sticky="ew", pady=(10, 0))

    def _new_tab(self, title):
        frame = ScrollFrame(self.notebook)
        self.notebook.add(frame, text=title)
        body = frame.inner
        body.grid_columnconfigure(0, weight=1)
        return body

    def _section(self, parent, title, row):
        box = ttk.LabelFrame(parent, text=title, padding=12)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        box.grid_columnconfigure(1, weight=1)
        box.grid_columnconfigure(3, weight=1)
        return box

    def _add_entry(self, box, group, key, label, default="", row=0, col=0, width=22):
        ttk.Label(box, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(box, textvariable=var, width=width)
        entry.grid(row=row, column=col + 1, sticky="ew", pady=4, padx=(0, 16))
        self.inputs.setdefault(group, {})[key] = var
        return entry

    def _add_combo(self, box, group, key, label, values, default="", row=0, col=0, width=22, dataset_combo=False, column_for=None):
        ttk.Label(box, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=default)
        combo = ttk.Combobox(box, textvariable=var, values=values, width=width)
        combo.grid(row=row, column=col + 1, sticky="ew", pady=4, padx=(0, 16))
        self.inputs.setdefault(group, {})[key] = var
        if dataset_combo:
            self.df_combos.append(combo)
            combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_columns())
            combo.bind("<FocusOut>", lambda _event: self.refresh_columns())
        if column_for:
            self.column_combos.append((combo, column_for))
        return combo

    def _add_check(self, box, group, key, label, default=True, row=0, col=0):
        var = tk.BooleanVar(value=default)
        check = ttk.Checkbutton(box, text=label, variable=var)
        check.grid(row=row, column=col, columnspan=2, sticky="w", pady=4, padx=(0, 16))
        self.inputs.setdefault(group, {})[key] = var
        return check

    def _run_button(self, parent, row, label, command):
        btn = ttk.Button(parent, text=label, style="Accent.TButton", command=command)
        btn.grid(row=row, column=0, sticky="ew", pady=(4, 0))
        return btn

    def _build_characterization_tab(self):
        group = "characterization"
        tab = self._new_tab("Caracterizacion")
        box = self._section(tab, "Parametros", 0)
        self._add_combo(box, group, "df_name", "Dataset", [], "anthro_data", 0, 0, dataset_combo=True)
        self._add_combo(box, group, "analysis_mode", "Modo", ["by_column", "full_matrix", "both"], "both", 0, 2)
        self._add_entry(box, group, "numeric_cols", "Columnas numericas", "", 1, 0)
        self._add_entry(box, group, "bins", "Bins", "80", 1, 2)
        self._add_check(box, group, "plot_positive_hist", "Graficar histograma X > 0", True, 2, 0)
        self._add_check(box, group, "verbose", "Mostrar resumen en log", True, 2, 2)
        self._run_button(tab, 1, "Ejecutar caracterizacion", lambda: self.run_analysis("characterization"))

    def _build_normality_tab(self):
        group = "normality"
        tab = self._new_tab("Normalidad")
        box = self._section(tab, "Parametros", 0)
        self._add_combo(box, group, "df_name", "Dataset", [], "anthro_data", 0, 0, dataset_combo=True)
        self._add_combo(box, group, "analysis_mode", "Modo", ["by_column", "full_matrix", "both"], "both", 0, 2)
        self._add_entry(box, group, "numeric_cols", "Columnas numericas", "", 1, 0)
        self._add_combo(box, group, "value_mode", "Valores", ["all", "positive", "both"], "both", 1, 2)
        self._add_combo(box, group, "test_method", "Prueba", ["shapiro", "anderson", "both"], "both", 2, 0)
        self._add_entry(box, group, "alpha", "Alpha", "0.0003", 2, 2)
        self._add_check(box, group, "verbose", "Mostrar resumen en log", True, 3, 0)
        self._run_button(tab, 1, "Ejecutar normalidad", lambda: self.run_analysis("normality"))

    def _build_kde_tab(self):
        group = "kde"
        tab = self._new_tab("KDE")
        box = self._section(tab, "Parametros", 0)
        self._add_combo(box, group, "data_df_name", "Dataset OTU", [], "otu_data_converted", 0, 0, dataset_combo=True)
        self._add_entry(box, group, "grid_size", "Grid size", "1000", 0, 2)
        self._add_entry(box, group, "cv_subsample", "CV subsample", "1000", 1, 0)
        self._add_entry(box, group, "cv_folds", "CV folds", "3", 1, 2)
        self._add_entry(box, group, "cv_bw_grid", "CV BW grid", "8", 2, 0)
        self._add_entry(box, group, "min_bandwidth", "Min bandwidth", "1.0", 2, 2)
        self._add_entry(box, group, "cv_max_expansions", "Max expansions", "4", 3, 0)
        self._add_entry(box, group, "test_kernel_bandwidths", "BW por kernel", "", 3, 2)
        self._add_check(box, group, "verbose", "Mostrar resumen en log", True, 4, 0)
        self._run_button(tab, 1, "Ejecutar KDE", lambda: self.run_analysis("kde"))

    def _build_kruskal_tab(self):
        group = "kruskal"
        tab = self._new_tab("Kruskal-Wallis")
        box = self._section(tab, "Parametros", 0)
        self._add_combo(box, group, "group_df_name", "Dataset grupos", [], "anthro_data", 0, 0, dataset_combo=True)
        self._add_combo(box, group, "value_df_name", "Dataset valores", [], "anthro_data", 0, 2, dataset_combo=True)
        self._add_combo(box, group, "group_col", "Columna grupo", [], "bmi_class", 1, 0, column_for=("kruskal", "group_df_name"))
        self._add_combo(box, group, "id_col_group", "ID grupos", [], "ID", 1, 2, column_for=("kruskal", "group_df_name"))
        self._add_combo(box, group, "id_col_value", "ID valores", [], "ID", 2, 0, column_for=("kruskal", "value_df_name"))
        self._add_entry(box, group, "value_cols", "Variables", "glucose, HDL, LDL, waist, body_fat, HOMA_IR", 2, 2)
        self._add_entry(box, group, "alpha", "Alpha", "0.0003", 3, 0)
        self._add_entry(box, group, "min_group_size", "Min grupo", "3", 3, 2)
        self._add_check(box, group, "apply_fdr", "Aplicar FDR", True, 4, 0)
        self._add_check(box, group, "verbose", "Mostrar resumen en log", True, 4, 2)
        self._run_button(tab, 1, "Ejecutar Kruskal-Wallis", lambda: self.run_analysis("kruskal"))

    def _build_mann_whitney_tab(self):
        group = "mann_whitney"
        tab = self._new_tab("Mann-Whitney")
        box = self._section(tab, "Parametros", 0)
        self._add_combo(box, group, "group_df_name", "Dataset grupos", [], "anthro_data", 0, 0, dataset_combo=True)
        self._add_combo(box, group, "value_df_name", "Dataset valores", [], "otu_data_converted", 0, 2, dataset_combo=True)
        self._add_combo(box, group, "group_col", "Columna grupo", [], "sex", 1, 0, column_for=("mann_whitney", "group_df_name"))
        self._add_entry(box, group, "groups_to_compare", "Grupos", "Male, Female", 1, 2)
        self._add_combo(box, group, "id_col_group", "ID grupos", [], "ID", 2, 0, column_for=("mann_whitney", "group_df_name"))
        self._add_combo(box, group, "id_col_value", "ID valores", [], "ID", 2, 2, column_for=("mann_whitney", "value_df_name"))
        self._add_entry(box, group, "value_cols", "Variables", "", 3, 0)
        self._add_combo(box, group, "alternative", "Alternativa", ["two-sided", "less", "greater"], "two-sided", 3, 2)
        self._add_entry(box, group, "alpha", "Alpha", "0.0003", 4, 0)
        self._add_entry(box, group, "min_group_size", "Min grupo", "3", 4, 2)
        self._add_check(box, group, "apply_fdr", "Aplicar FDR", True, 5, 0)
        self._add_check(box, group, "verbose", "Mostrar resumen en log", True, 5, 2)
        self._run_button(tab, 1, "Ejecutar Mann-Whitney", lambda: self.run_analysis("mann_whitney"))

    def _build_dbscan_tab(self):
        group = "dbscan"
        tab = self._new_tab("DBSCAN")
        data_box = self._section(tab, "Datos y limpieza", 0)
        self._add_combo(data_box, group, "data_df_name", "Dataset datos", [], "anthro_data", 0, 0, dataset_combo=True)
        self._add_combo(data_box, group, "id_col", "ID datos", [], "ID", 0, 2, column_for=("dbscan", "data_df_name"))
        self._add_entry(data_box, group, "feature_cols", "Features", "waist, age, HDL", 1, 0)
        self._add_combo(data_box, group, "meta_df_name", "Dataset meta", [], "anthro_data", 1, 2, dataset_combo=True)
        self._add_combo(data_box, group, "meta_id_col", "ID meta", [], "ID", 2, 0, column_for=("dbscan", "meta_df_name"))
        self._add_combo(data_box, group, "missing_strategy", "Faltantes", ["fill_zero", "drop_rows", "median"], "fill_zero", 2, 2)
        self._add_check(data_box, group, "drop_non_numeric", "Quitar no numericas", True, 3, 0)
        self._add_check(data_box, group, "remove_zero_rows", "Quitar filas suma 0", True, 3, 2)
        self._add_entry(data_box, group, "min_prevalence", "Min prevalence", "", 4, 0)
        self._add_entry(data_box, group, "min_total_abundance", "Min abundance", "", 4, 2)

        model_box = self._section(tab, "Modelo", 1)
        self._add_entry(model_box, group, "eps", "eps", "1.0", 0, 0)
        self._add_entry(model_box, group, "min_samples", "Min samples", "3", 0, 2)
        self._add_combo(model_box, group, "transform_method", "Transformacion", ["none", "log1p", "clr"], "none", 1, 0)
        self._add_entry(model_box, group, "pseudocount", "Pseudocount", "1.0", 1, 2)
        self._add_check(model_box, group, "scale", "Escalar variables", True, 2, 0)
        self._add_combo(model_box, group, "embedding_method", "Embedding", ["none", "pca", "kpca", "isomap", "mds", "tsne", "umap"], "pca", 2, 2)
        self._add_entry(model_box, group, "n_components", "Componentes", "3", 3, 0)
        self._add_entry(model_box, group, "random_state", "Random state", "42", 3, 2)
        self._add_entry(model_box, group, "embedding_kwargs", "Embedding JSON", "", 4, 0)

        out_box = self._section(tab, "Figuras y resumen", 2)
        self._add_check(out_box, group, "calculate_k_distance", "Calcular k-distance", True, 0, 0)
        self._add_entry(out_box, group, "k_distance_min_samples", "K-distance k", "8", 0, 2)
        self._add_check(out_box, group, "plot_k_distance_graph", "Guardar figura k-distance", True, 1, 0)
        self._add_check(out_box, group, "plot_embedding_graph", "Guardar figura embedding", True, 1, 2)
        self._add_entry(out_box, group, "summary_numeric_cols", "Resumen numerico", "glucose, waist", 2, 0)
        self._add_entry(out_box, group, "summary_categorical_cols", "Resumen categorico", "bmi_class", 2, 2)
        self._add_entry(out_box, group, "summary_numeric_aggs", "Agregaciones", "median", 3, 0)
        self._add_check(out_box, group, "verbose", "Mostrar resumen en log", True, 3, 2)
        self._run_button(tab, 3, "Ejecutar DBSCAN", lambda: self.run_analysis("dbscan"))

    def load_files(self):
        paths = filedialog.askopenfilenames(
            title="Selecciona datasets",
            filetypes=[
                ("Archivos soportados", "*.csv *.otus *.txt *.meta *.taxonomy"),
                ("CSV", "*.csv"),
                ("Tabulados", "*.otus *.txt *.meta *.taxonomy"),
                ("Todos", "*.*"),
            ],
        )
        if not paths:
            return
        for path_text in paths:
            path = Path(path_text)
            try:
                df = load_dataframe_from_path(path)
                name = unique_name(path.stem, self.dfs)
                self.dfs[name] = df
                self._log(f"Cargado: {name} -> {df.shape}")
            except Exception as exc:
                self._log(f"Error cargando {path.name}: {exc}")
        self.refresh_datasets()

    def remove_selected_dataset(self):
        selected = self.dataset_tree.selection()
        if not selected:
            return
        for item in selected:
            name = self.dataset_tree.item(item, "text")
            self.dfs.pop(name, None)
            self._log(f"Dataset quitado de memoria: {name}")
        self.refresh_datasets()

    def preview_selected_dataset(self):
        selected = self.dataset_tree.selection()
        if not selected:
            messagebox.showinfo(APP_TITLE, "Selecciona un dataset para previsualizar.")
            return
        name = self.dataset_tree.item(selected[0], "text")
        df = self.dfs.get(name)
        if df is None:
            return
        top = tk.Toplevel(self)
        top.title(f"Vista previa - {name}")
        top.geometry("980x520")
        frame = ttk.Frame(top, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"{name} | shape {df.shape}", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        tree = ttk.Treeview(frame, show="headings")
        tree.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        preview = df.head(100).copy()
        columns = [str(c) for c in preview.columns[:60]]
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=130, stretch=False)
        for _, row in preview.iloc[:, :60].iterrows():
            values = ["" if pd.isna(v) else str(v) for v in row.tolist()]
            tree.insert("", "end", values=values)

    def refresh_datasets(self):
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
        for name, df in sorted(self.dfs.items()):
            self.dataset_tree.insert("", "end", text=name, values=(f"{df.shape[0]} x {df.shape[1]}",))

        names = sorted(self.dfs.keys())
        for combo in self.df_combos:
            combo.configure(values=names)
        self.refresh_columns()

    def refresh_columns(self):
        for combo, (group, dataset_key) in self.column_combos:
            dataset_name = self.inputs.get(group, {}).get(dataset_key, tk.StringVar()).get()
            df = self.dfs.get(dataset_name)
            combo.configure(values=[] if df is None else list(map(str, df.columns)))

    def choose_output_dir(self):
        path = filedialog.askdirectory(title="Carpeta de salida", initialdir=self.output_dir_var.get() or str(DEFAULT_OUTPUT_DIR))
        if path:
            self.output_dir_var.set(path)

    def open_output_dir(self):
        path = Path(self.output_dir_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"No se pudo abrir la carpeta:\n{exc}")

    def run_analysis(self, analysis):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo(APP_TITLE, "Ya hay un analisis en ejecucion.")
            return
        if not self.dfs:
            messagebox.showinfo(APP_TITLE, "Carga al menos un dataset antes de ejecutar.")
            return
        self.refresh_columns()
        try:
            params = self._collect_params(analysis)
            output_root = Path(self.output_dir_var.get()).expanduser()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Revisa los parametros:\n{exc}")
            return
        self.status_var.set(f"Ejecutando {analysis}...")
        self._log(f"\n=== Ejecutando {analysis} ===")
        self.worker = threading.Thread(target=self._worker_run, args=(analysis, params, output_root), daemon=True)
        self.worker.start()

    def _worker_run(self, analysis, params, output_root):
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_root / f"{stamp}_{sanitize_name(analysis)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            figure_dir = run_dir / "figures"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout), FigureCapture(figure_dir) as figs:
                result = self._execute(analysis, params)

            log_text = stdout.getvalue()
            if log_text:
                (run_dir / "execution_log.txt").write_text(log_text, encoding="utf-8")

            exporter = ArtifactExporter(run_dir)
            manifest = exporter.export(result, prefix=analysis)
            manifest.update({
                "analysis": analysis,
                "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "parameters": json_safe(params),
                "figures": [str(p) for p in figs.saved],
            })
            with (run_dir / "manifest.json").open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)
            with (run_dir / "parameters.json").open("w", encoding="utf-8") as fh:
                json.dump(json_safe(params), fh, ensure_ascii=False, indent=2)

            self.msg_queue.put(("done", analysis, result, run_dir, manifest, log_text))
        except Exception:
            self.msg_queue.put(("error", analysis, traceback.format_exc()))

    def _collect_params(self, analysis):
        values = {key: var.get() for key, var in self.inputs[analysis].items()}

        if analysis == "characterization":
            return {
                "df_name": values["df_name"],
                "numeric_cols": split_list(values["numeric_cols"]),
                "analysis_mode": values["analysis_mode"],
                "bins": int(values["bins"]),
                "plot_positive_hist": parse_bool(values["plot_positive_hist"]),
                "verbose": parse_bool(values["verbose"]),
            }

        if analysis == "normality":
            return {
                "df_name": values["df_name"],
                "numeric_cols": split_list(values["numeric_cols"]),
                "analysis_mode": values["analysis_mode"],
                "value_mode": values["value_mode"],
                "test_method": values["test_method"],
                "alpha": float(values["alpha"]),
                "verbose": parse_bool(values["verbose"]),
            }

        if analysis == "kde":
            return {
                "data_df_name": values["data_df_name"],
                "grid_size": int(values["grid_size"]),
                "cv_subsample": int(values["cv_subsample"]),
                "cv_folds": int(values["cv_folds"]),
                "cv_bw_grid": int(values["cv_bw_grid"]),
                "min_bandwidth": float(values["min_bandwidth"]),
                "cv_max_expansions": int(values["cv_max_expansions"]),
                "test_kernel_bandwidths": parse_bandwidths(values["test_kernel_bandwidths"]),
                "verbose": parse_bool(values["verbose"]),
            }

        if analysis == "kruskal":
            return {
                "alpha": float(values["alpha"]),
                "group_df_name": values["group_df_name"],
                "value_df_name": values["value_df_name"],
                "group_col": values["group_col"],
                "id_col_group": values["id_col_group"],
                "id_col_value": values["id_col_value"],
                "value_cols": split_list(values["value_cols"]),
                "min_group_size": int(values["min_group_size"]),
                "apply_fdr": parse_bool(values["apply_fdr"]),
                "verbose": parse_bool(values["verbose"]),
            }

        if analysis == "mann_whitney":
            groups = split_list(values["groups_to_compare"])
            return {
                "alpha": float(values["alpha"]),
                "group_df_name": values["group_df_name"],
                "value_df_name": values["value_df_name"],
                "group_col": values["group_col"],
                "groups_to_compare": tuple(groups) if groups else None,
                "id_col_group": values["id_col_group"],
                "id_col_value": values["id_col_value"],
                "value_cols": split_list(values["value_cols"]),
                "min_group_size": int(values["min_group_size"]),
                "alternative": values["alternative"],
                "apply_fdr": parse_bool(values["apply_fdr"]),
                "verbose": parse_bool(values["verbose"]),
            }

        if analysis == "dbscan":
            meta_df_name = values["meta_df_name"].strip() or None
            return {
                "data_df_name": values["data_df_name"],
                "id_col": values["id_col"].strip() or None,
                "feature_cols": split_list(values["feature_cols"]),
                "meta_df_name": meta_df_name,
                "meta_id_col": values["meta_id_col"].strip() or None,
                "eps": float(values["eps"]),
                "min_samples": int(values["min_samples"]),
                "calculate_k_distance": parse_bool(values["calculate_k_distance"]),
                "k_distance_min_samples": int(values["k_distance_min_samples"]),
                "drop_non_numeric": parse_bool(values["drop_non_numeric"]),
                "missing_strategy": values["missing_strategy"],
                "remove_zero_rows": parse_bool(values["remove_zero_rows"]),
                "min_prevalence": parse_optional_float(values["min_prevalence"]),
                "min_total_abundance": parse_optional_float(values["min_total_abundance"]),
                "transform_method": values["transform_method"],
                "pseudocount": float(values["pseudocount"]),
                "scale": parse_bool(values["scale"]),
                "embedding_method": values["embedding_method"],
                "n_components": int(values["n_components"]),
                "random_state": int(values["random_state"]),
                "embedding_kwargs": parse_json_dict(values["embedding_kwargs"]),
                "plot_k_distance_graph": parse_bool(values["plot_k_distance_graph"]),
                "plot_embedding_graph": parse_bool(values["plot_embedding_graph"]),
                "summary_numeric_cols": split_list(values["summary_numeric_cols"]),
                "summary_categorical_cols": split_list(values["summary_categorical_cols"]),
                "summary_numeric_aggs": parse_tuple(values["summary_numeric_aggs"]) or ("median",),
                "verbose": parse_bool(values["verbose"]),
            }

        raise ValueError(f"Analisis desconocido: {analysis}")

    def _execute(self, analysis, params):
        if analysis == "characterization":
            return distribution_plots_from_loaded(dfs=self.dfs, **params)
        if analysis == "normality":
            return normality_tests_from_loaded(dfs=self.dfs, **params)
        if analysis == "kde":
            return kde_from_loaded(dfs=self.dfs, **params)
        if analysis == "kruskal":
            return kruskal_wallis_from_loaded(dfs=self.dfs, **params)
        if analysis == "mann_whitney":
            return mann_whitney_from_loaded(dfs=self.dfs, **params)
        if analysis == "dbscan":
            return dbscan_from_loaded(dfs=self.dfs, **params)
        raise ValueError(f"Analisis desconocido: {analysis}")

    def _poll_queue(self):
        try:
            while True:
                message = self.msg_queue.get_nowait()
                kind = message[0]
                if kind == "done":
                    _, analysis, result, run_dir, manifest, log_text = message
                    key = f"{analysis}_{_dt.datetime.now().strftime('%H%M%S')}"
                    self.results[key] = result
                    self.last_run_dir = Path(run_dir)
                    self.result_tree.insert("", "end", text=key, values=(_dt.datetime.now().strftime("%H:%M:%S"),))
                    if log_text:
                        self._log(log_text.rstrip())
                    self._log(f"Terminado: {analysis}")
                    self._log(f"Salida: {run_dir}")
                    self._log(f"Tablas: {len(manifest.get('tables', []))} | Arrays: {len(manifest.get('arrays', []))} | Figuras: {len(manifest.get('figures', []))}")
                    self.status_var.set(f"Listo. Ultima salida: {run_dir}")
                elif kind == "error":
                    _, analysis, trace = message
                    self._log(f"Error en {analysis}:\n{trace}")
                    self.status_var.set(f"Error en {analysis}")
                    messagebox.showerror(APP_TITLE, f"El analisis fallo. Revisa el log.\n\n{trace.splitlines()[-1]}")
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _log(self, text):
        self.log_text.insert("end", str(text) + "\n")
        self.log_text.see("end")


def main():
    app = MicrobiotaGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
