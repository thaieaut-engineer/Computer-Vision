from pathlib import Path
import math
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QDesktopServices
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QSplitter, QStatusBar,
    QProgressBar, QTabWidget, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.core.image_io import read_image, write_image
from src.core.pipeline import process_image
from src.core.experiment import run_experiment


class ProcessingWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, settings, reference_image=None):
        super().__init__()
        self.image = image.copy()
        self.reference_image = reference_image.copy() if reference_image is not None else None
        self.settings = dict(settings)

    def run(self):
        try:
            self.completed.emit(process_image(self.image, self.settings, self.reference_image))
        except Exception as exc:
            self.failed.emit(str(exc))


class ExperimentWorker(QThread):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image):
        super().__init__()
        self.image = image.copy()

    def run(self):
        try:
            result = run_experiment(
                self.image,
                progress_callback=lambda done, total, label: self.progress.emit(done, total, label),
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageView(QLabel):
    def __init__(self, empty_text="Chưa có ảnh"):
        super().__init__(empty_text)
        self.empty_text = empty_text
        self._pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 170)
        self.setStyleSheet("background:#151c29;color:#8290a6;border:1px solid #303d50;border-radius:8px;padding:5px;")
        self.setWordWrap(True)

    def set_image(self, image):
        if image is None:
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText(self.empty_text)
            return
        arr = np.asarray(image)
        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._render_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_pixmap()

    def _render_pixmap(self):
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.setText("")


class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.plot_images(None, None)

    def plot_images(self, original, processed):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if original is None:
            ax.text(0.5, 0.5, "Mở ảnh để xem histogram", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            def gray(img):
                if img.ndim == 2:
                    return img
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax.hist(gray(original).ravel(), bins=256, range=(0, 256), density=True,
                    histtype="step", linewidth=1.5, label="Ảnh gốc")
            if processed is not None:
                ax.hist(gray(processed).ravel(), bins=256, range=(0, 256), density=True,
                        histtype="step", linewidth=1.5, label="Ảnh kết quả")
            ax.set_title("Histogram mức xám")
            ax.set_xlabel("Mức xám (0–255)")
            ax.set_ylabel("Tần suất chuẩn hóa")
            ax.grid(True, alpha=0.2)
            ax.legend()
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageEnhancement | Nâng cao chất lượng ảnh")
        self.resize(1500, 960)
        self.original_image = None
        self.reference_image = None
        self.reference_path = None
        self.noisy_image = None
        self.result_image = None
        self.display_image = None
        self.processing_worker = None
        self.experiment_worker = None
        self.last_experiment = None
        self._build_ui()
        self._build_menu()
        self._apply_style()

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("Tệp")
        open_action = QAction("Mở ảnh...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        save_action = QAction("Lưu kết quả...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        exit_action = QAction("Thoát", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("IMAGE ENHANCEMENT")
        title.setObjectName("appTitle")
        subtitle = QLabel("Nâng cao chất lượng ảnh bằng biến đổi điểm ảnh và lọc trong miền không gian")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.open_btn = QPushButton("＋  Mở ảnh")
        self.open_btn.clicked.connect(self.open_image)
        self.save_btn = QPushButton("↓  Lưu ảnh kết quả")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        self.reset_btn = QPushButton("↺  Đặt lại")
        self.reset_btn.clicked.connect(self.reset_controls)
        header.addWidget(self.open_btn)
        header.addWidget(self.save_btn)
        header.addWidget(self.reset_btn)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_processing_tab(), "Xử lý ảnh")
        self.tabs.addTab(self._build_experiment_tab(), "Thực nghiệm")
        self.tabs.addTab(self._build_histogram_tab(), "Histogram")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Sẵn sàng | Mở ảnh để bắt đầu")

    def _build_processing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        preview_split = QSplitter(Qt.Orientation.Horizontal)
        original_group = QGroupBox("ẢNH ĐẦU VÀO")
        original_layout = QVBoxLayout(original_group)
        self.original_view = ImageView("Mở ảnh để bắt đầu")
        original_layout.addWidget(self.original_view)
        result_group = QGroupBox("ẢNH KẾT QUẢ")
        result_layout = QVBoxLayout(result_group)
        self.result_view = ImageView("Kết quả xử lý sẽ hiển thị tại đây")
        result_layout.addWidget(self.result_view)
        preview_split.addWidget(original_group)
        preview_split.addWidget(result_group)
        left_layout.addWidget(preview_split, 3)

        edge_group = QGroupBox("SO SÁNH ĐƯỜNG BIÊN SOBEL – TÁC ĐỘNG RIÊNG CỦA BỘ LỌC")
        edge_layout = QGridLayout(edge_group)
        self.edge_views = {}
        for index, name in enumerate(("Ảnh tham chiếu", "Ảnh nhiễu", "Sau Mean", "Sau Median")):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view = ImageView("Chưa có dữ liệu")
            view.setMinimumHeight(110)
            cell_layout.addWidget(label)
            cell_layout.addWidget(view)
            edge_layout.addWidget(cell, index // 2, index % 2)
            self.edge_views[name] = view
        left_layout.addWidget(edge_group, 2)
        splitter.addWidget(left)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(310)
        control_scroll.setMaximumWidth(390)
        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(8, 0, 0, 0)
        control_layout.setSpacing(9)

        reference_group = QGroupBox("ẢNH THAM CHIẾU SẠCH (GROUND TRUTH)")
        reference_layout = QVBoxLayout(reference_group)
        self.reference_status = QLabel("Ảnh đang mở là ảnh được xử lý. Khi mô phỏng nhiễu, nên mở ảnh sạch; ảnh tham chiếu riêng (nếu chọn) được dùng làm ground truth để đánh giá.")
        self.reference_status.setWordWrap(True)
        self.reference_status.setObjectName("hint")
        reference_layout.addWidget(self.reference_status)
        reference_buttons = QHBoxLayout()
        self.reference_btn = QPushButton("Chọn ảnh tham chiếu sạch...")
        self.reference_btn.clicked.connect(self.open_reference_image)
        self.clear_reference_btn = QPushButton("Bỏ chọn")
        self.clear_reference_btn.clicked.connect(self.clear_reference_image)
        self.clear_reference_btn.setEnabled(False)
        reference_buttons.addWidget(self.reference_btn)
        reference_buttons.addWidget(self.clear_reference_btn)
        reference_layout.addLayout(reference_buttons)
        control_layout.addWidget(reference_group)

        transform_group = QGroupBox("1. BIẾN ĐỔI ĐIỂM ẢNH")
        transform_form = QFormLayout(transform_group)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setToolTip("Gamma < 1 thường làm sáng; Gamma > 1 thường làm tối.")
        transform_form.addRow("Gamma:", self.gamma_spin)
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.1, 3.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(1.0)
        transform_form.addRow("Tương phản (α):", self.contrast_spin)
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(-100, 100)
        transform_form.addRow("Độ sáng (β):", self.brightness_spin)
        control_layout.addWidget(transform_group)

        noise_group = QGroupBox("2. MÔ PHỎNG NHIỄU")
        noise_form = QFormLayout(noise_group)
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["Không thêm nhiễu", "Gaussian", "Salt & Pepper"])
        noise_form.addRow("Loại nhiễu:", self.noise_combo)
        self.noise_level = QDoubleSpinBox()
        self.noise_level.setRange(1, 100)
        self.noise_level.setValue(5)
        self.noise_level.setDecimals(1)
        noise_form.addRow("Mức nhiễu:", self.noise_level)
        self.noise_hint = QLabel("Gaussian: độ lệch chuẩn σ. Salt & Pepper: tỷ lệ pixel nhiễu (%).")
        self.noise_hint.setWordWrap(True)
        self.noise_hint.setObjectName("hint")
        noise_form.addRow(self.noise_hint)
        control_layout.addWidget(noise_group)

        filter_group = QGroupBox("3. BỘ LỌC KHÔNG GIAN")
        filter_form = QFormLayout(filter_group)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Không lọc", "Mean", "Median"])
        filter_form.addRow("Bộ lọc:", self.filter_combo)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["3", "5", "7"])
        filter_form.addRow("Kích thước kernel:", self.kernel_combo)
        control_layout.addWidget(filter_group)

        edge_group = QGroupBox("4. PHÁT HIỆN ĐƯỜNG BIÊN SOBEL")
        edge_form = QFormLayout(edge_group)
        self.sobel_combo = QComboBox()
        self.sobel_combo.addItems(["Không dùng", "Sobel X", "Sobel Y", "Sobel Magnitude"])
        edge_form.addRow("Hiển thị:", self.sobel_combo)
        control_layout.addWidget(edge_group)

        self.apply_btn = QPushButton("▶  ÁP DỤNG XỬ LÝ")
        self.apply_btn.setObjectName("applyButton")
        self.apply_btn.clicked.connect(self.apply_processing)
        control_layout.addWidget(self.apply_btn)
        self.process_progress = QProgressBar()
        self.process_progress.setRange(0, 0)
        self.process_progress.hide()
        control_layout.addWidget(self.process_progress)
        metrics_group = QGroupBox("5. CHỈ SỐ CHẤT LƯỢNG")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_label = QLabel("MSE: –   |   PSNR: –   |   SSIM: –")
        self.metrics_label.setObjectName("metricsLabel")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)
        self.edge_metric_label = QLabel("Edge F1: –")
        self.edge_metric_label.setObjectName("edgeMetricLabel")
        metrics_layout.addWidget(self.edge_metric_label)
        metric_note = QLabel("Chỉ số chỉ được tính khi có ảnh tham chiếu sạch hợp lệ. Gamma/Contrast được áp dụng tương ứng lên ảnh tham chiếu trước khi so sánh để đánh giá riêng tác động khử nhiễu.")
        metric_note.setObjectName("hint")
        metric_note.setWordWrap(True)
        metrics_layout.addWidget(metric_note)
        control_layout.addWidget(metrics_group)
        control_layout.addStretch()
        control_scroll.setWidget(controls)
        splitter.addWidget(control_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter)
        return tab

    def _build_experiment_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        intro = QLabel(
            "Chạy tự động 72 ca: Gaussian/Salt & Pepper ở hai mức nhiễu; Mean/Median với "
            "kernel 3×3, 5×5, 7×7; Gamma 0.8, 1.0, 1.2. Ảnh sạch dùng để tạo nhiễu và làm "
            "ground truth là ảnh tham chiếu riêng nếu đã chọn, nếu không sẽ dùng ảnh đang mở. "
            "Ảnh tham chiếu được biến đổi Gamma tương ứng trước khi tính chỉ số."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.run_experiment_btn = QPushButton("▶  CHẠY 72 CA THỰC NGHIỆM")
        self.run_experiment_btn.setObjectName("applyButton")
        self.run_experiment_btn.clicked.connect(self.start_experiment)
        row.addWidget(self.run_experiment_btn)
        self.open_results_btn = QPushButton("Mở thư mục kết quả")
        self.open_results_btn.setEnabled(False)
        self.open_results_btn.clicked.connect(self.open_experiment_folder)
        row.addWidget(self.open_results_btn)
        row.addStretch()
        layout.addLayout(row)
        self.experiment_progress = QProgressBar()
        self.experiment_progress.setRange(0, 100)
        self.experiment_progress.setValue(0)
        layout.addWidget(self.experiment_progress)
        self.experiment_status = QLabel("Chưa chạy thực nghiệm.")
        layout.addWidget(self.experiment_status)
        self.experiment_table = QTableWidget(0, 9)
        self.experiment_table.setHorizontalHeaderLabels([
            "Nhiễu", "Mức", "Gamma", "Bộ lọc", "Kernel", "MSE", "PSNR (dB)", "SSIM", "Edge F1"
        ])
        self.experiment_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.experiment_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.experiment_table.setAlternatingRowColors(True)
        layout.addWidget(self.experiment_table, 1)
        self.experiment_folder = None
        return tab

    def _build_histogram_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        note = QLabel("Histogram cho biết phân bố mức xám của ảnh gốc và ảnh kết quả.")
        note.setObjectName("hint")
        layout.addWidget(note)
        self.histogram_canvas = HistogramCanvas(tab)
        layout.addWidget(self.histogram_canvas, 1)
        return tab

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#0f1420; color:#e7ecf5; font-family:'Segoe UI'; font-size:10pt; }
            QMenuBar, QMenu { background:#151d2b; color:#e7ecf5; }
            QMenuBar::item:selected, QMenu::item:selected { background:#2878d0; }
            QLabel#appTitle { font-size:20pt; font-weight:700; color:#f3f6fc; }
            QLabel#subtitle { color:#9ba8bc; font-size:9pt; }
            QLabel#hint { color:#9aa9bf; font-size:9pt; }
            QGroupBox { border:1px solid #303d50; border-radius:9px; margin-top:11px; padding:10px; font-weight:600; color:#bdcbe0; }
            QGroupBox::title { subcontrol-origin:margin; left:11px; padding:0 5px; }
            QPushButton { background:#26364b; border:1px solid #3b4e68; border-radius:7px; padding:8px 12px; color:#edf3ff; }
            QPushButton:hover { background:#344b68; }
            QPushButton:disabled { color:#69758a; background:#1b2432; }
            QPushButton#applyButton { background:#2878d0; border:0; font-weight:700; padding:11px; }
            QPushButton#applyButton:hover { background:#368be7; }
            QComboBox, QSpinBox, QDoubleSpinBox { background:#1a2535; border:1px solid #394b63; border-radius:5px; padding:5px; min-height:22px; }
            QComboBox QAbstractItemView { background:#1a2535; selection-background-color:#2878d0; }
            QTabWidget::pane { border:1px solid #303d50; border-radius:7px; }
            QTabBar::tab { background:#1a2535; color:#aebbd0; padding:9px 18px; margin-right:3px; border-top-left-radius:6px; border-top-right-radius:6px; }
            QTabBar::tab:selected { background:#2878d0; color:white; }
            QLabel#metricsLabel { color:#8fd3b2; font-size:10pt; font-weight:700; }
            QLabel#edgeMetricLabel { color:#a9c7ff; font-weight:600; }
            QTableWidget { background:#151d2b; alternate-background-color:#1b2636; gridline-color:#344156; }
            QHeaderView::section { background:#26364b; color:#edf3ff; padding:7px; border:0; }
            QProgressBar { border:1px solid #344156; border-radius:4px; background:#1a2535; text-align:center; min-height:12px; }
            QProgressBar::chunk { background:#368be7; border-radius:3px; }
            QStatusBar { color:#9ba8bc; }
        """)

    def _set_busy(self, busy):
        self.open_btn.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy)
        self.run_experiment_btn.setEnabled(not busy)
        self.reset_btn.setEnabled(not busy)
        if hasattr(self, "reference_btn"):
            self.reference_btn.setEnabled(not busy)
            self.clear_reference_btn.setEnabled((not busy) and self.reference_image is not None)
        if busy:
            self.process_progress.show()
        else:
            self.process_progress.hide()

    def open_image(self):
        if self._workers_running():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            image = read_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "Không mở được ảnh", str(exc))
            return
        self.original_image = image
        self.reference_image = None
        self.reference_path = None
        if hasattr(self, "reference_status"):
            self.reference_status.setText("Ảnh đang mở là ảnh được xử lý. Khi mô phỏng nhiễu, nên mở ảnh sạch; ảnh tham chiếu riêng (nếu chọn) được dùng làm ground truth để đánh giá.")
            self.clear_reference_btn.setEnabled(False)
        self.noisy_image = None
        self.result_image = None
        self.display_image = None
        self.original_view.set_image(image)
        self.result_view.set_image(None)
        for view in self.edge_views.values():
            view.set_image(None)
        self.save_btn.setEnabled(False)
        self.metrics_label.setText("MSE: –   |   PSNR: –   |   SSIM: –")
        self.edge_metric_label.setText("Edge F1: –")
        self.histogram_canvas.plot_images(image, None)
        self.statusBar().showMessage(f"Đã mở {Path(path).name} | {image.shape[1]} × {image.shape[0]} pixels")
        self.tabs.setCurrentIndex(0)

    def open_reference_image(self):
        if self._workers_running():
            return
        if self.original_image is None:
            QMessageBox.information(self, "Chưa có ảnh đầu vào", "Hãy mở ảnh cần xử lý trước, sau đó chọn ảnh tham chiếu sạch.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh tham chiếu sạch", "", "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            reference = read_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "Không mở được ảnh tham chiếu", str(exc))
            return
        if self.original_image is not None and reference.shape[:2] != self.original_image.shape[:2]:
            QMessageBox.warning(
                self, "Kích thước không khớp",
                "Ảnh tham chiếu và ảnh đầu vào phải có cùng chiều rộng và chiều cao. "
                f"Đầu vào: {self.original_image.shape[1]}×{self.original_image.shape[0]}, "
                f"tham chiếu: {reference.shape[1]}×{reference.shape[0]}."
            )
            return
        self.reference_image = reference
        self.reference_path = path
        self.reference_status.setText(
            f"Đã chọn ảnh tham chiếu sạch: {Path(path).name} "
            f"({reference.shape[1]}×{reference.shape[0]})."
        )
        self.clear_reference_btn.setEnabled(True)
        self.statusBar().showMessage(f"Đã chọn ảnh tham chiếu sạch: {Path(path).name}")

    def clear_reference_image(self):
        self.reference_image = None
        self.reference_path = None
        self.clear_reference_btn.setEnabled(False)
        self.reference_status.setText(
            "Đã bỏ ảnh tham chiếu riêng. Nếu mô phỏng nhiễu, ảnh đang mở sẽ được xem "
            "là ảnh sạch ban đầu; nếu ảnh vốn đã nhiễu thì không bật mô phỏng nhiễu."
        )
        self.metrics_label.setText("MSE: –   |   PSNR: –   |   SSIM: –")
        self.edge_metric_label.setText("Edge F1: –")

    def reset_controls(self):
        self.gamma_spin.setValue(1.0)
        self.contrast_spin.setValue(1.0)
        self.brightness_spin.setValue(0)
        self.noise_combo.setCurrentIndex(0)
        self.noise_level.setValue(5)
        self.filter_combo.setCurrentIndex(0)
        self.kernel_combo.setCurrentIndex(0)
        self.sobel_combo.setCurrentIndex(0)
        if self.original_image is not None:
            self.result_image = None
            self.display_image = None
            self.result_view.set_image(None)
            self.save_btn.setEnabled(False)
            self.metrics_label.setText("MSE: –   |   PSNR: –   |   SSIM: –")
            self.edge_metric_label.setText("Edge F1: –")
            self.histogram_canvas.plot_images(self.original_image, None)

    def apply_processing(self):
        if self.original_image is None:
            QMessageBox.information(self, "Chưa có ảnh", "Vui lòng mở ảnh trước khi xử lý.")
            return
        if self._workers_running():
            return
        settings = {
            "noise": self.noise_combo.currentText(),
            "noise_level": self.noise_level.value(),
            "gamma": self.gamma_spin.value(),
            "contrast": self.contrast_spin.value(),
            "brightness": self.brightness_spin.value(),
            "filter": self.filter_combo.currentText(),
            "kernel": int(self.kernel_combo.currentText()),
            "sobel": self.sobel_combo.currentText(),
        }
        self._set_busy(True)
        self.statusBar().showMessage("Đang xử lý ảnh trong luồng nền...")
        self.processing_worker = ProcessingWorker(
            self.original_image, settings, reference_image=self.reference_image
        )
        self.processing_worker.completed.connect(self.on_processing_completed)
        self.processing_worker.failed.connect(self.on_worker_failed)
        self.processing_worker.finished.connect(lambda: self._set_busy(False))
        self.processing_worker.start()

    def on_processing_completed(self, result):
        self.noisy_image = result["noisy"]
        self.result_image = result["enhanced"]
        self.display_image = result["display"]
        self.result_view.set_image(self.display_image)
        q = result["quality"]
        if q.get("valid", False):
            psnr_text = "∞" if math.isinf(q["psnr"]) else f"{q['psnr']:.3f} dB"
            self.metrics_label.setText(
                f"MSE: {q['mse']:.3f}    |    PSNR: {psnr_text}    |    SSIM cục bộ: {q['ssim']:.5f}"
            )
            e = q["edge"]
            self.edge_metric_label.setText(
                f"Edge Precision: {e['precision']:.4f}    Recall: {e['recall']:.4f}    F1: {e['f1']:.4f}"
            )
        else:
            self.metrics_label.setText("MSE: N/A   |   PSNR: N/A   |   SSIM: N/A")
            self.edge_metric_label.setText("Edge F1: N/A")
            QMessageBox.information(self, "Chưa có ảnh tham chiếu sạch", q.get("reason", "Không thể đánh giá định lượng."))
        for name, image in result["edge_panels"].items():
            self.edge_views[name].set_image(image)
        self.save_btn.setEnabled(True)
        self.histogram_canvas.plot_images(self.original_image, self.result_image)
        self.statusBar().showMessage("Xử lý hoàn tất. Chỉ số chất lượng được tính trước bước hiển thị Sobel.")

    def on_worker_failed(self, message):
        self._set_busy(False)
        QMessageBox.critical(self, "Lỗi xử lý", message)
        self.statusBar().showMessage("Tác vụ thất bại.")

    def save_result(self):
        if self.display_image is None:
            QMessageBox.information(self, "Chưa có kết quả", "Hãy xử lý ảnh trước khi lưu.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Lưu ảnh đang hiển thị", "ket_qua.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if not path:
            return
        try:
            saved = write_image(path, self.display_image)
            self.statusBar().showMessage(f"Đã lưu ảnh: {saved}")
        except Exception as exc:
            QMessageBox.warning(self, "Không lưu được ảnh", str(exc))

    def start_experiment(self):
        if self.original_image is None:
            QMessageBox.information(self, "Chưa có ảnh", "Vui lòng mở ảnh trước khi chạy thực nghiệm.")
            return
        if self._workers_running():
            return
        if self.reference_image is None:
            answer = QMessageBox.question(
                self, "Xác nhận ảnh sạch",
                "Bạn chưa chọn ảnh tham chiếu riêng. Chương trình sẽ coi ảnh đang mở là ảnh sạch "
                "để tự tạo nhiễu và tính các chỉ số. Ảnh đang mở có chắc chắn là ảnh sạch không?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.run_experiment_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.reference_btn.setEnabled(False)
        self.clear_reference_btn.setEnabled(False)
        self.experiment_progress.setValue(0)
        clean_source = self.reference_image if self.reference_image is not None else self.original_image
        source_label = Path(self.reference_path).name if self.reference_path else "ảnh đang mở (đã xác nhận sạch)"
        self.experiment_status.setText(f"Đang chạy 72 ca trên ground truth: {source_label}...")
        self.experiment_worker = ExperimentWorker(clean_source)
        self.experiment_worker.progress.connect(self.on_experiment_progress)
        self.experiment_worker.completed.connect(self.on_experiment_completed)
        self.experiment_worker.failed.connect(self.on_worker_failed)
        self.experiment_worker.finished.connect(self.on_experiment_thread_finished)
        self.experiment_worker.start()

    def on_experiment_progress(self, done, total, label):
        self.experiment_progress.setValue(int(done * 100 / total))
        self.experiment_status.setText(f"Đang chạy {done}/{total}: {label}")

    def on_experiment_completed(self, result):
        self.last_experiment = result
        self.experiment_folder = result["output_dir"]
        rows = result["rows"]
        self.experiment_table.setRowCount(len(rows))
        for r, data in enumerate(rows):
            values = [data["noise"], f"{data['noise_level']:g}", f"{data['gamma']:.1f}",
                      data["method"], f"{data['kernel']}×{data['kernel']}",
                      f"{data['mse']:.3f}", f"{data['psnr_db']:.3f}",
                      f"{data['ssim_local']:.5f}", f"{data['edge_f1']:.5f}"]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.experiment_table.setItem(r, c, item)
        self.open_results_btn.setEnabled(True)
        self.experiment_progress.setValue(100)
        self.experiment_status.setText(
            f"Hoàn thành {len(rows)} ca (3 Gamma × 2 nhiễu × 2 mức × 2 bộ lọc × 3 kernel). Kết quả: {result['output_dir']}"
        )
        self.statusBar().showMessage("Thực nghiệm hoàn tất.")

    def on_experiment_thread_finished(self):
        self.run_experiment_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.reference_btn.setEnabled(True)
        self.clear_reference_btn.setEnabled(self.reference_image is not None)

    def open_experiment_folder(self):
        if self.experiment_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.experiment_folder))

    def _workers_running(self):
        return ((self.processing_worker is not None and self.processing_worker.isRunning()) or
                (self.experiment_worker is not None and self.experiment_worker.isRunning()))

    def closeEvent(self, event):
        # Chờ worker kết thúc an toàn trước khi hủy cửa sổ.
        for worker in (self.processing_worker, self.experiment_worker):
            if worker is not None and worker.isRunning():
                worker.wait()
        event.accept()
