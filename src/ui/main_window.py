from pathlib import Path
import math
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QSplitter, QStatusBar,
    QProgressBar, QTabWidget, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame, QSizePolicy
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.core.image_io import read_image, write_image
from src.core.pipeline import process_image
from src.core.experiment import run_experiment


class ProcessingWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, settings):
        super().__init__()
        self.image = image.copy()
        self.settings = dict(settings)

    def run(self):
        try:
            self.completed.emit(process_image(self.image, self.settings))
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
        self.setMinimumSize(40, 30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#0c0d16;color:#6f7292;border:1px solid #202236;border-radius:8px;padding:3px;")
        self.setWordWrap(True)

    def sizeHint(self):
        return QSize(220, 150)

    def minimumSizeHint(self):
        return QSize(40, 30)

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
        if self._pixmap is not None and self.width() > 2 and self.height() > 2:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.setText("")


class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4), facecolor="#0c0d16", tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setStyleSheet("background:#0c0d16;")
        self.plot_images(None, None)

    def plot_images(self, original, processed):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#121320")
        if original is None:
            ax.text(0.5, 0.5, "Mở ảnh để xem histogram", ha="center", va="center",
                    transform=ax.transAxes, color="#6f7292", fontsize=11)
            ax.set_axis_off()
        else:
            def gray(img):
                if img.ndim == 2:
                    return img
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax.hist(gray(original).ravel(), bins=256, range=(0, 256), density=True,
                    histtype="step", linewidth=1.6, color="#c084fc", label="Ảnh gốc")
            if processed is not None:
                ax.hist(gray(processed).ravel(), bins=256, range=(0, 256), density=True,
                    histtype="step", linewidth=1.6, color="#38bdf8", label="Ảnh kết quả")
            ax.set_title("Histogram mức xám", color="#f1f3fa", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("Mức xám (0–255)", color="#a5a8c9", fontsize=9.5)
            ax.set_ylabel("Tần suất chuẩn hóa", color="#a5a8c9", fontsize=9.5)
            ax.tick_params(colors="#8f91ad", labelsize=8.5)
            for spine in ax.spines.values():
                spine.set_color("#23253c")
            ax.grid(True, color="#23253c", alpha=0.6, linestyle="--")
            ax.legend(facecolor="#18192a", edgecolor="#2d304a", labelcolor="#e2e4f0", fontsize=9)
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageEnhancement | Nâng cao chất lượng ảnh")
        self.setMinimumSize(850, 520)
        self._init_window_geometry()
        self.original_image = None
        self.noisy_image = None
        self.result_image = None
        self.display_image = None
        self.processing_worker = None
        self.experiment_worker = None
        self.last_experiment = None
        self._build_ui()
        self._build_menu()
        self._apply_style()

    def _init_window_geometry(self):
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1360, max(850, int(avail.width() * 0.88)))
            h = min(860, max(520, int(avail.height() * 0.88)))
            self.resize(w, h)
        else:
            self.resize(1180, 720)

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
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("✦  IMAGE ENHANCEMENT")
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
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Trái: Ảnh gốc + kết quả (trên) và So sánh Sobel (dưới), ngăn cách bởi vertical splitter
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setChildrenCollapsible(False)

        preview_split = QSplitter(Qt.Orientation.Horizontal)
        preview_split.setChildrenCollapsible(False)
        original_group = QGroupBox("ẢNH ĐẦU VÀO")
        original_layout = QVBoxLayout(original_group)
        original_layout.setContentsMargins(6, 6, 6, 6)
        self.original_view = ImageView("Mở ảnh để bắt đầu")
        original_layout.addWidget(self.original_view)
        result_group = QGroupBox("ẢNH KẾT QUẢ")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(6, 6, 6, 6)
        self.result_view = ImageView("Kết quả xử lý sẽ hiển thị tại đây")
        result_layout.addWidget(self.result_view)
        preview_split.addWidget(original_group)
        preview_split.addWidget(result_group)
        preview_split.setStretchFactor(0, 1)
        preview_split.setStretchFactor(1, 1)
        left_splitter.addWidget(preview_split)

        edge_group = QGroupBox("SO SÁNH ĐƯỜNG BIÊN SOBEL – TÁC ĐỘNG RIÊNG CỦA BỘ LỌC")
        edge_layout = QGridLayout(edge_group)
        edge_layout.setContentsMargins(6, 6, 6, 6)
        edge_layout.setSpacing(4)
        self.edge_views = {}
        for index, name in enumerate(("Ảnh tham chiếu", "Ảnh nhiễu", "Sau Mean", "Sau Median")):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color:#a0b0c6; font-size:9pt;")
            view = ImageView("Chưa có dữ liệu")
            cell_layout.addWidget(label)
            cell_layout.addWidget(view)
            edge_layout.addWidget(cell, index // 2, index % 2)
            self.edge_views[name] = view
        left_splitter.addWidget(edge_group)
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)
        splitter.addWidget(left_splitter)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setMinimumWidth(280)
        control_scroll.setMaximumWidth(420)
        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(6, 4, 6, 4)
        control_layout.setSpacing(6)

        transform_group = QGroupBox("1. BIẾN ĐỔI ĐIỂM ẢNH")
        transform_form = QFormLayout(transform_group)
        transform_form.setVerticalSpacing(4)
        transform_form.setHorizontalSpacing(8)
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
        noise_form.setVerticalSpacing(4)
        noise_form.setHorizontalSpacing(8)
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
        filter_form.setVerticalSpacing(4)
        filter_form.setHorizontalSpacing(8)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Không lọc", "Mean", "Median"])
        filter_form.addRow("Bộ lọc:", self.filter_combo)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["3", "5", "7"])
        filter_form.addRow("Kích thước kernel:", self.kernel_combo)
        control_layout.addWidget(filter_group)

        edge_group = QGroupBox("4. PHÁT HIỆN ĐƯỜNG BIÊN SOBEL")
        edge_form = QFormLayout(edge_group)
        edge_form.setVerticalSpacing(4)
        edge_form.setHorizontalSpacing(8)
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
        metrics_layout.setContentsMargins(8, 8, 8, 8)
        metrics_layout.setSpacing(4)
        self.metrics_label = QLabel("MSE: –   |   PSNR: –\nSSIM: –   |   Edge F1: –")
        self.metrics_label.setObjectName("metricsLabel")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)
        self.edge_metric_label = QLabel("Biên: Precision: –   |   Recall: –")
        self.edge_metric_label.setObjectName("edgeMetricLabel")
        self.edge_metric_label.setWordWrap(True)
        metrics_layout.addWidget(self.edge_metric_label)
        metric_note = QLabel("Chỉ số chất lượng được tính so với ảnh gốc (cùng Gamma/Contrast).")
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
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        intro = QLabel(
            "Chạy tự động 72 ca: Gaussian/Salt & Pepper ở hai mức nhiễu; Mean/Median với "
            "kernel 3×3, 5×5, 7×7; Gamma 0.8, 1.0, 1.2. Ảnh đang mở được dùng làm ảnh gốc "
            "để tạo nhiễu và làm ground truth. Ảnh gốc được biến đổi Gamma tương ứng trước khi tính chỉ số."
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
        self.experiment_table.horizontalHeader().setMinimumSectionSize(65)
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
            /* Tổng thể giao diện tối phong cách Obsidian / Neon Violet */
            QMainWindow, QWidget {
                background: #090a10;
                color: #f1f3fa;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                font-size: 9.5pt;
            }

            /* Thanh Menu */
            QMenuBar {
                background: #0e0f18;
                color: #e2e4f0;
                border-bottom: 1px solid #1f2133;
                padding: 2px 6px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 5px 10px;
                border-radius: 5px;
            }
            QMenuBar::item:selected {
                background: #22243a;
                color: #ffffff;
            }
            QMenu {
                background: #141523;
                color: #f1f3fa;
                border: 1px solid #2b2d45;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 5px;
            }
            QMenu::item:selected {
                background: #7c3aed;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #24263b;
                margin: 4px 6px;
            }

            /* Tiêu đề & phụ đề */
            QLabel#appTitle {
                font-size: 16pt;
                font-weight: 800;
                color: #ffffff;
            }
            QLabel#subtitle {
                color: #8b8ea8;
                font-size: 8.5pt;
            }
            QLabel#hint {
                color: #7b7e9b;
                font-size: 8.5pt;
            }

            /* Card GroupBox viền cong nhẹ, bóng tối hiện đại */
            QGroupBox {
                background: #12131f;
                border: 1px solid #222438;
                border-radius: 9px;
                margin-top: 10px;
                padding: 8px 10px 8px 10px;
                font-weight: 700;
                color: #c4b5fd;
                font-size: 9pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                background: #12131f;
                border-radius: 4px;
            }

            /* Nút phụ (Mở ảnh, Lưu kết quả, Đặt lại) */
            QPushButton {
                background: #1c1d2e;
                border: 1px solid #2d304a;
                border-radius: 7px;
                padding: 6px 14px;
                color: #e6e8f7;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #26283f;
                border-color: #8b5cf6;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #161726;
                border-color: #7c3aed;
            }
            QPushButton:disabled {
                color: #4f516d;
                background: #12131e;
                border-color: #1c1d2c;
            }

            /* Nút chính Hành động (Gradient Tím Neon rực rỡ như mẫu) */
            QPushButton#applyButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:0.5 #8b5cf6, stop:1 #a855f7);
                border: none;
                font-weight: 700;
                padding: 9px;
                font-size: 10pt;
                color: #ffffff;
                border-radius: 8px;
            }
            QPushButton#applyButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:0.5 #9d6cf8, stop:1 #b86bfc);
            }
            QPushButton#applyButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6d28d9, stop:0.5 #7c3aed, stop:1 #8b5cf6);
            }

            /* Các ô nhập liệu (ComboBox, SpinBox) */
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #171827;
                border: 1px solid #2b2d45;
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 22px;
                color: #f1f3fa;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #4c4f72;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #8b5cf6;
                background: #1b1c2e;
            }
            QComboBox QAbstractItemView {
                background: #151624;
                border: 1px solid #2b2d45;
                selection-background-color: #7c3aed;
                color: #f1f3fa;
                border-radius: 6px;
                padding: 4px;
            }

            /* Tabs điều hướng */
            QTabWidget::pane {
                border: 1px solid #202236;
                border-radius: 8px;
                background: #0d0e17;
                top: -1px;
            }
            QTabBar::tab {
                background: #141523;
                color: #8b8da8;
                padding: 8px 18px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid #202236;
                border-bottom: none;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background: #1b1c2e;
                color: #c4b5fd;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #8b5cf6);
                color: #ffffff;
                border-color: #7c3aed;
            }

            /* Chỉ số chất lượng nổi bật */
            QLabel#metricsLabel {
                color: #c084fc;
                font-size: 9.5pt;
                font-weight: 700;
            }
            QLabel#edgeMetricLabel {
                color: #818cf8;
                font-size: 9pt;
                font-weight: 600;
            }

            /* Bảng kết quả thực nghiệm */
            QTableWidget {
                background: #0d0e17;
                alternate-background-color: #131422;
                gridline-color: #202236;
                border: 1px solid #202236;
                border-radius: 6px;
                color: #e2e4f0;
            }
            QHeaderView::section {
                background: #171828;
                color: #c4b5fd;
                padding: 7px;
                border: none;
                border-right: 1px solid #202236;
                border-bottom: 1px solid #202236;
                font-weight: 600;
            }
            QTableWidget::item:selected {
                background: #7c3aed;
                color: #ffffff;
            }

            /* Thanh tiến trình Progress Bar */
            QProgressBar {
                border: 1px solid #282a42;
                border-radius: 5px;
                background: #121320;
                text-align: center;
                color: #ffffff;
                font-size: 8pt;
                min-height: 12px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #a855f7);
                border-radius: 4px;
            }

            /* Status Bar */
            QStatusBar {
                color: #8588a4;
                background: #090a10;
                border-top: 1px solid #171826;
                font-size: 8.5pt;
            }

            /* Thanh kéo Splitter */
            QSplitter::handle {
                background: #171828;
            }
            QSplitter::handle:hover {
                background: #8b5cf6;
            }
            QSplitter::handle:horizontal {
                width: 4px;
            }
            QSplitter::handle:vertical {
                height: 4px;
            }

            /* Thanh cuộn mượt mà */
            QScrollBar:vertical {
                background: #0a0b12;
                width: 9px;
                border-radius: 4px;
            }
            QScrollBar:horizontal {
                background: #0a0b12;
                height: 9px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #25273d;
                min-height: 24px;
                min-width: 24px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: #8b5cf6;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                height: 0;
                width: 0;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background: none;
            }
        """)

    def _set_busy(self, busy):
        self.open_btn.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy)
        self.run_experiment_btn.setEnabled(not busy)
        self.reset_btn.setEnabled(not busy)
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
        self.noisy_image = None
        self.result_image = None
        self.display_image = None
        self.original_view.set_image(image)
        self.result_view.set_image(None)
        for view in self.edge_views.values():
            view.set_image(None)
        self.save_btn.setEnabled(False)
        self.metrics_label.setText("MSE: –   |   PSNR: –\nSSIM: –   |   Edge F1: –")
        self.edge_metric_label.setText("Biên: Precision: –   |   Recall: –")
        self.histogram_canvas.plot_images(image, None)
        self.statusBar().showMessage(f"Đã mở {Path(path).name} | {image.shape[1]} × {image.shape[0]} pixels")
        self.tabs.setCurrentIndex(0)

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
            self.metrics_label.setText("MSE: –   |   PSNR: –\nSSIM: –   |   Edge F1: –")
            self.edge_metric_label.setText("Biên: Precision: –   |   Recall: –")
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
            self.original_image, settings
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
                f"MSE: {q['mse']:.3f}   |   PSNR: {psnr_text}\n"
                f"SSIM cục bộ: {q['ssim']:.5f}   |   Edge F1: {q['edge']['f1']:.4f}"
            )
            e = q["edge"]
            self.edge_metric_label.setText(
                f"Biên: Precision: {e['precision']:.4f}   |   Recall: {e['recall']:.4f}"
            )
        else:
            self.metrics_label.setText("MSE: N/A   |   PSNR: N/A\nSSIM: N/A   |   Edge F1: N/A")
            self.edge_metric_label.setText("Biên: Precision: N/A   |   Recall: N/A")
            QMessageBox.information(self, "Thông báo", q.get("reason", "Không thể đánh giá định lượng."))
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
        self.run_experiment_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.experiment_progress.setValue(0)
        self.experiment_status.setText("Đang chạy 72 ca trên ảnh đang mở...")
        self.experiment_worker = ExperimentWorker(self.original_image)
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
