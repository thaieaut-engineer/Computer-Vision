from pathlib import Path
import math
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QSize, QRectF, QPointF
from PyQt6.QtGui import (
    QImage, QPixmap, QAction, QKeySequence, QDesktopServices,
    QPainter, QColor, QFont, QPen, QCursor
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QSplitter, QStatusBar,
    QProgressBar, QTabWidget, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame, QSizePolicy
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.core.image_io import read_image, write_image
from src.core.pipeline import process_image, compute_difference_map
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


class ImageView(QWidget):
    """Khung hiển thị ảnh tương tác chuyên sâu: hỗ trợ Phóng to (Zoom) và Di chuyển (Pan).
    
    - Cuộn chuột: Phóng to / Thu nhỏ theo vị trí con trỏ chuột (1.0x - 10.0x).
    - Giữ chuột trái và kéo: Di chuyển khung nhìn (Pan).
    - Nhấp đúp chuột: Đặt lại tỷ lệ gốc 1.0x vừa khung hình.
    - Phát tín hiệu zoom_changed để đồng bộ chuyển động giữa nhiều ảnh (Sync Zoom).
    """
    zoom_changed = pyqtSignal(float, float, float)  # (zoom_factor, center_x, center_y)

    def __init__(self, empty_text="Chưa có ảnh", parent=None):
        super().__init__(parent)
        self.empty_text = empty_text
        self._image = None
        self._qimage = None
        self.zoom_factor = 1.0
        self.center_x = 0.5
        self.center_y = 0.5
        self._dragging = False
        self._drag_start = None
        self._block_sync = False
        self.setMouseTracking(True)
        self.setMinimumSize(40, 30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#0c0d16; border:1px solid #202236; border-radius:8px;")

    def sizeHint(self):
        return QSize(220, 150)

    def minimumSizeHint(self):
        return QSize(40, 30)

    def set_image(self, image):
        self._image = image
        if image is None:
            self._qimage = None
            self.update()
            return
        arr = np.asarray(image)
        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self._qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.center_x = 0.5
        self.center_y = 0.5
        self.update()

    def set_view_transform(self, zoom, cx, cy):
        if self._block_sync:
            return
        self._block_sync = True
        try:
            self.zoom_factor = max(1.0, min(10.0, zoom))
            self.center_x = max(0.0, min(1.0, cx))
            self.center_y = max(0.0, min(1.0, cy))
            self.update()
        finally:
            self._block_sync = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0c0d16"))
        if self._qimage is None:
            painter.setPen(QColor("#6f7292"))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return

        W, H = self.width(), self.height()
        img_w, img_h = self._qimage.width(), self._qimage.height()
        scale = min(W / img_w, H / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        offset_x = (W - disp_w) / 2.0
        offset_y = (H - disp_h) / 2.0
        target_rect = QRectF(offset_x, offset_y, disp_w, disp_h)

        src_w = img_w / self.zoom_factor
        src_h = img_h / self.zoom_factor
        src_cx = self.center_x * img_w
        src_cy = self.center_y * img_h
        src_left = max(0.0, min(img_w - src_w, src_cx - src_w / 2.0))
        src_top = max(0.0, min(img_h - src_h, src_cy - src_h / 2.0))
        src_rect = QRectF(src_left, src_top, src_w, src_h)

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target_rect, self._qimage, src_rect)

        # Hiển thị huy hiệu mức Zoom ở góc dưới phải khi đang phóng to
        if self.zoom_factor > 1.01:
            badge_text = f"{int(round(self.zoom_factor * 100))}%"
            badge_w, badge_h = 52, 22
            badge_rect = QRectF(W - badge_w - 6, H - badge_h - 6, badge_w, badge_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(18, 19, 32, 220))
            painter.drawRoundedRect(badge_rect, 5, 5)
            painter.setPen(QColor("#c084fc"))
            font = QFont("Segoe UI", 8, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

    def wheelEvent(self, event):
        if self._qimage is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else (1.0 / 1.2)

        W, H = self.width(), self.height()
        img_w, img_h = self._qimage.width(), self._qimage.height()
        scale = min(W / img_w, H / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        offset_x = (W - disp_w) / 2.0
        offset_y = (H - disp_h) / 2.0

        pos = event.position()
        mx, my = pos.x(), pos.y()
        norm_mx = (mx - offset_x) / disp_w if disp_w > 0 else 0.5
        norm_my = (my - offset_y) / disp_h if disp_h > 0 else 0.5
        norm_mx = max(0.0, min(1.0, norm_mx))
        norm_my = max(0.0, min(1.0, norm_my))

        src_w = img_w / self.zoom_factor
        src_h = img_h / self.zoom_factor
        src_left = max(0.0, min(img_w - src_w, self.center_x * img_w - src_w / 2.0))
        src_top = max(0.0, min(img_h - src_h, self.center_y * img_h - src_h / 2.0))
        img_pt_x = src_left + norm_mx * src_w
        img_pt_y = src_top + norm_my * src_h

        new_zoom = max(1.0, min(10.0, self.zoom_factor * factor))
        if abs(new_zoom - 1.0) < 0.05:
            new_zoom = 1.0
            new_cx, new_cy = 0.5, 0.5
        else:
            new_src_w = img_w / new_zoom
            new_src_h = img_h / new_zoom
            src_cx_new = img_pt_x + (0.5 - norm_mx) * new_src_w
            src_cy_new = img_pt_y + (0.5 - norm_my) * new_src_h
            new_cx = max(0.0, min(1.0, src_cx_new / img_w))
            new_cy = max(0.0, min(1.0, src_cy_new / img_h))

        self.zoom_factor = new_zoom
        self.center_x = new_cx
        self.center_y = new_cy
        self.update()
        if not self._block_sync:
            self.zoom_changed.emit(self.zoom_factor, self.center_x, self.center_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.zoom_factor > 1.0:
            self._dragging = True
            self._drag_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start is not None and self._qimage is not None:
            delta = event.position() - self._drag_start
            self._drag_start = event.position()
            img_w, img_h = self._qimage.width(), self._qimage.height()
            scale = min(self.width() / img_w, self.height() / img_h)
            dx_img = - delta.x() / (scale * self.zoom_factor) if scale > 0 else 0
            dy_img = - delta.y() / (scale * self.zoom_factor) if scale > 0 else 0
            self.center_x = max(0.0, min(1.0, self.center_x + dx_img / img_w))
            self.center_y = max(0.0, min(1.0, self.center_y + dy_img / img_h))
            self.update()
            if not self._block_sync:
                self.zoom_changed.emit(self.zoom_factor, self.center_x, self.center_y)
        elif self.zoom_factor > 1.01:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoom_factor > 1.01 else Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        self.reset_zoom()
        if not self._block_sync:
            self.zoom_changed.emit(1.0, 0.5, 0.5)


class MetricCard(QFrame):
    """Thẻ hiển thị chỉ số thực nghiệm nổi bật (Obsidian / Neon Accent)."""
    def __init__(self, title, initial_val="–", val_color="#c084fc", subtitle="", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setStyleSheet(f"""
            QFrame#metricCard {{
                background: #111220;
                border: 1px solid #23253c;
                border-radius: 8px;
                padding: 4px;
            }}
            QFrame#metricCard:hover {{
                border-color: #7c3aed;
                background: #151627;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #8b8da8; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.5px;")

        self.lbl_val = QLabel(initial_val)
        self.lbl_val.setStyleSheet(f"color: {val_color}; font-size: 13pt; font-weight: 800; font-family: 'Segoe UI', system-ui, sans-serif;")

        self.lbl_sub = QLabel(subtitle)
        self.lbl_sub.setStyleSheet("color: #636683; font-size: 7.5pt;")
        self.lbl_sub.setWordWrap(True)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_val)
        layout.addWidget(self.lbl_sub)

    def set_value(self, val_text, sub_text=None):
        self.lbl_val.setText(val_text)
        if sub_text is not None:
            self.lbl_sub.setText(sub_text)


class ImageCard(QFrame):
    """Card hiển thị ảnh theo chuẩn UI hiện đại: Khung ảnh bo góc, tiêu đề bước và huy hiệu (Pill Badge)."""
    def __init__(self, step_title="1. Ảnh gốc", badge_text="Sạch", badge_color="green", empty_text="Chưa có ảnh", parent=None):
        super().__init__(parent)
        self.setObjectName("imageCard")
        self.setStyleSheet("""
            QFrame#imageCard {
                background: #10111d;
                border: 1px solid #202236;
                border-radius: 12px;
                padding: 6px;
            }
            QFrame#imageCard:hover {
                border-color: #3b3e66;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 8)
        layout.setSpacing(6)

        # Khung ảnh tương tác (Zoom & Pan)
        self.view = ImageView(empty_text)
        layout.addWidget(self.view, 1)

        # Hàng thông tin dưới ảnh: Tiêu đề + Huy hiệu (Pill Badge)
        self.info_layout = QHBoxLayout()
        self.info_layout.setContentsMargins(2, 2, 2, 2)
        self.info_layout.setSpacing(8)

        self.lbl_title = QLabel(step_title)
        self.lbl_title.setStyleSheet("color: #f1f3fa; font-size: 10pt; font-weight: 700;")
        self.info_layout.addWidget(self.lbl_title)

        self.info_layout.addStretch()

        self.lbl_badge = QLabel(badge_text)
        self.set_badge(badge_text, badge_color)
        self.info_layout.addWidget(self.lbl_badge)

        layout.addLayout(self.info_layout)

    def set_title(self, title):
        self.lbl_title.setText(title)

    def set_badge(self, text, color="green"):
        self.lbl_badge.setText(text)
        if color == "green":
            bg = "rgba(16, 185, 129, 0.16)"
            border = "rgba(16, 185, 129, 0.35)"
            fg = "#34d399"
        elif color in ("orange", "peach"):
            bg = "rgba(249, 115, 22, 0.16)"
            border = "rgba(249, 115, 22, 0.35)"
            fg = "#fb923c"
        elif color == "purple":
            bg = "rgba(139, 92, 246, 0.18)"
            border = "rgba(139, 92, 246, 0.38)"
            fg = "#c084fc"
        elif color == "cyan":
            bg = "rgba(56, 189, 248, 0.16)"
            border = "rgba(56, 189, 248, 0.35)"
            fg = "#38bdf8"
        else:
            bg = "rgba(100, 116, 139, 0.16)"
            border = "rgba(100, 116, 139, 0.3)"
            fg = "#94a3b8"

        self.lbl_badge.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 9px;
                color: {fg};
                font-size: 8pt;
                font-weight: 600;
                padding: 2px 8px;
            }}
        """)




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
        self.last_result = None
        self.showing_diff = False
        self.result_title_text = "3. Sau Median"
        self.result_badge_text = "Đã lọc"
        self.result_badge_color = "green"
        self.processing_worker = None
        self.experiment_worker = None
        self.last_experiment = None
        self._build_ui()
        self._build_menu()
        self._apply_style()
        self._init_zoom_sync()

    def _init_zoom_sync(self):
        self.preview_views = [self.original_view, self.noisy_view, self.result_view]
        for v in self.preview_views:
            v.zoom_changed.connect(self._on_view_zoom_changed)

    def _on_view_zoom_changed(self, zoom, cx, cy):
        if not self.sync_zoom_cb.isChecked():
            return
        sender = self.sender()
        for v in self.preview_views:
            if v is not sender:
                v.set_view_transform(zoom, cx, cy)

    def reset_all_zoom(self):
        for v in self.preview_views:
            v.reset_zoom()

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

        # Trái: 3 ảnh xem trước (trên) và So sánh Sobel (dưới), ngăn cách bởi vertical splitter
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setChildrenCollapsible(False)

        # Container chứa tiêu đề nhóm 'KẾT QUẢ XỬ LÝ ẢNH' và 3 card ảnh
        preview_container = QWidget()
        preview_container_layout = QVBoxLayout(preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)
        preview_container_layout.setSpacing(6)

        # Hàng tiêu đề trên cùng: 'KẾT QUẢ XỬ LÝ ẢNH' (trái) + Thanh công cụ Zoom (phải)
        top_header_layout = QHBoxLayout()
        top_header_layout.setContentsMargins(4, 2, 4, 2)
        top_header_layout.setSpacing(10)

        section_title = QLabel("KẾT QUẢ XỬ LÝ ẢNH")
        section_title.setStyleSheet("color: #f1f3fa; font-size: 11pt; font-weight: 800; letter-spacing: 0.5px;")
        top_header_layout.addWidget(section_title)

        top_header_layout.addStretch()

        self.sync_zoom_cb = QCheckBox("🔗 Đồng bộ Zoom")
        self.sync_zoom_cb.setChecked(True)
        self.sync_zoom_cb.setToolTip("Đồng bộ vị trí và tỷ lệ thu phóng giữa 3 ảnh.")

        self.reset_zoom_btn = QPushButton("↺ 100%")
        self.reset_zoom_btn.setStyleSheet("""
            QPushButton {
                background: #171828; color: #a5a8c9; border: 1px solid #282a42;
                border-radius: 5px; padding: 2px 10px; font-size: 8pt;
            }
            QPushButton:hover { background: #22243a; color: #e2e4f0; border-color: #7c3aed; }
        """)
        self.reset_zoom_btn.clicked.connect(self.reset_all_zoom)

        zoom_hint = QLabel("💡 Cuộn: Zoom • Kéo: Pan • Đúp: Reset")
        zoom_hint.setStyleSheet("color: #5d607e; font-size: 8pt;")

        top_header_layout.addWidget(self.sync_zoom_cb)
        top_header_layout.addWidget(self.reset_zoom_btn)
        top_header_layout.addWidget(zoom_hint)
        preview_container_layout.addLayout(top_header_layout)

        # Khu vực 3 ảnh song song với thẻ bo góc và huy hiệu trạng thái
        preview_split = QSplitter(Qt.Orientation.Horizontal)
        preview_split.setChildrenCollapsible(False)

        # 1. Ảnh gốc [Sạch]
        self.card_clean = ImageCard("1. Ảnh gốc", "Sạch", "green", "Mở ảnh để bắt đầu")
        self.original_view = self.card_clean.view

        # 2. Ảnh nhiễu [Salt & Pepper / Gaussian / Không nhiễu]
        self.card_noisy = ImageCard("2. Ảnh nhiễu", "Không nhiễu", "gray", "Ảnh nhiễu sẽ hiển thị tại đây")
        self.noisy_view = self.card_noisy.view

        # 3. Sau Median [Đã lọc]
        self.card_result = ImageCard("3. Sau Median", "Đã lọc", "green", "Kết quả xử lý sẽ hiển thị tại đây")
        self.result_view = self.card_result.view

        # Nút chuyển đổi nhanh xem Ảnh sai khác trên Card 3
        self.diff_toggle_btn = QPushButton("◩ Sai khác")
        self.diff_toggle_btn.setToolTip("Chuyển đổi giữa Ảnh kết quả lọc và Ảnh sai khác (Difference Image)")
        self.diff_toggle_btn.setStyleSheet("""
            QPushButton {
                background: #171828; color: #a5a8c9; border: 1px solid #2b2d45;
                border-radius: 9px; padding: 2px 8px; font-size: 7.5pt; font-weight: 600;
            }
            QPushButton:hover { background: #22243c; color: #c084fc; border-color: #7c3aed; }
        """)
        self.diff_toggle_btn.clicked.connect(self.toggle_result_diff)
        self.card_result.info_layout.insertWidget(2, self.diff_toggle_btn)

        preview_split.addWidget(self.card_clean)
        preview_split.addWidget(self.card_noisy)
        preview_split.addWidget(self.card_result)
        preview_split.setStretchFactor(0, 1)
        preview_split.setStretchFactor(1, 1)
        preview_split.setStretchFactor(2, 1)

        preview_container_layout.addWidget(preview_split, 1)
        left_splitter.addWidget(preview_container)

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

        # 5. Thông số thực nghiệm nổi bật (Metric Cards)
        metrics_group = QGroupBox("5. CHỈ SỐ CHẤT LƯỢNG")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setContentsMargins(6, 6, 6, 6)
        metrics_layout.setSpacing(6)

        cards_grid = QGridLayout()
        cards_grid.setContentsMargins(0, 0, 0, 0)
        cards_grid.setSpacing(6)

        self.card_psnr = MetricCard("PSNR (ĐỘ SẮC NÉT)", "–", "#38bdf8", "Tín hiệu / nhiễu (dB)")
        self.card_ssim = MetricCard("SSIM CỤC BỘ", "–", "#c084fc", "Tương đồng cấu trúc [0–1]")
        self.card_mse = MetricCard("MSE TOÀN PHƯƠNG", "–", "#fb7185", "Sai số trung bình")
        self.card_edge = MetricCard("EDGE F1 (SOBEL)", "–", "#a78bfa", "Bảo toàn biên [0–1]")

        cards_grid.addWidget(self.card_psnr, 0, 0)
        cards_grid.addWidget(self.card_ssim, 0, 1)
        cards_grid.addWidget(self.card_mse, 1, 0)
        cards_grid.addWidget(self.card_edge, 1, 1)
        metrics_layout.addLayout(cards_grid)

        details_box = QVBoxLayout()
        details_box.setSpacing(2)
        self.edge_metric_label = QLabel("Biên Sobel: Precision: –   |   Recall: –")
        self.edge_metric_label.setStyleSheet("color: #9498b8; font-size: 8pt; font-weight: 600;")
        self.time_label = QLabel("⚡ Thời gian xử lý: –")
        self.time_label.setStyleSheet("color: #34d399; font-size: 8.5pt; font-weight: 700;")
        details_box.addWidget(self.edge_metric_label)
        details_box.addWidget(self.time_label)
        metrics_layout.addLayout(details_box)

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

        # Thẻ tóm tắt thông số tối ưu nổi bật
        exp_summary_layout = QHBoxLayout()
        exp_summary_layout.setSpacing(6)
        self.exp_card_psnr = MetricCard("CẤU HÌNH PSNR TỐT NHẤT", "–", "#38bdf8", "Chưa chạy")
        self.exp_card_ssim = MetricCard("CẤU HÌNH SSIM TỐT NHẤT", "–", "#c084fc", "Chưa chạy")
        self.exp_card_edge = MetricCard("BẢO TOÀN BIÊN TỐT NHẤT (F1)", "–", "#a78bfa", "Chưa chạy")
        self.exp_card_speed = MetricCard("HIỆU NĂNG THỰC NGHIỆM", "–", "#34d399", "Tối ưu hóa học thuật")
        exp_summary_layout.addWidget(self.exp_card_psnr)
        exp_summary_layout.addWidget(self.exp_card_ssim)
        exp_summary_layout.addWidget(self.exp_card_edge)
        exp_summary_layout.addWidget(self.exp_card_speed)
        layout.addLayout(exp_summary_layout)

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
        self.last_result = None
        self.showing_diff = False
        self.diff_toggle_btn.setText("◩ Sai khác")

        self.original_view.set_image(image)
        self.card_clean.set_title("1. Ảnh gốc")
        self.card_clean.set_badge("Sạch", "green")

        self.noisy_view.set_image(None)
        self.card_noisy.set_title("2. Ảnh nhiễu")
        self.card_noisy.set_badge("Không nhiễu", "gray")

        self.result_view.set_image(None)
        self.card_result.set_title("3. Sau Median")
        self.card_result.set_badge("Chờ xử lý", "gray")

        self.reset_all_zoom()
        for view in self.edge_views.values():
            view.set_image(None)
        self.save_btn.setEnabled(False)
        self.card_psnr.set_value("–", "Tín hiệu / nhiễu (dB)")
        self.card_ssim.set_value("–", "Tương đồng cấu trúc [0–1]")
        self.card_mse.set_value("–", "Sai số trung bình")
        self.card_edge.set_value("–", "Bảo toàn biên [0–1]")
        self.edge_metric_label.setText("Biên Sobel: Precision: –   |   Recall: –")
        self.time_label.setText("⚡ Thời gian xử lý: –")
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
        self.showing_diff = False
        self.diff_toggle_btn.setText("◩ Sai khác")
        if self.original_image is not None:
            self.result_image = None
            self.display_image = None
            self.last_result = None
            self.original_view.set_image(self.original_image)
            self.card_clean.set_title("1. Ảnh gốc")
            self.card_clean.set_badge("Sạch", "green")

            self.noisy_view.set_image(None)
            self.card_noisy.set_title("2. Ảnh nhiễu")
            self.card_noisy.set_badge("Không nhiễu", "gray")

            self.result_view.set_image(None)
            self.card_result.set_title("3. Sau Median")
            self.card_result.set_badge("Chờ xử lý", "gray")

            self.reset_all_zoom()
            for view in self.edge_views.values():
                view.set_image(None)
            self.save_btn.setEnabled(False)
            self.card_psnr.set_value("–", "Tín hiệu / nhiễu (dB)")
            self.card_ssim.set_value("–", "Tương đồng cấu trúc [0–1]")
            self.card_mse.set_value("–", "Sai số trung bình")
            self.card_edge.set_value("–", "Bảo toàn biên [0–1]")
            self.edge_metric_label.setText("Biên Sobel: Precision: –   |   Recall: –")
            self.time_label.setText("⚡ Thời gian xử lý: –")
            self.histogram_canvas.plot_images(self.original_image, None)

    def toggle_result_diff(self):
        if not hasattr(self, "last_result") or self.last_result is None:
            return
        self.showing_diff = not self.showing_diff
        if self.showing_diff:
            diff_img = self.last_result.get("difference")
            if diff_img is not None:
                self.result_view.set_image(diff_img)
            self.card_result.set_title("3. Ảnh sai khác")
            self.card_result.set_badge("Sai khác 3×", "orange")
            self.diff_toggle_btn.setText("✦ Kết quả")
        else:
            self.result_view.set_image(self.display_image)
            self.card_result.set_title(getattr(self, "result_title_text", "3. Sau Median"))
            self.card_result.set_badge(getattr(self, "result_badge_text", "Đã lọc"), getattr(self, "result_badge_color", "green"))
            self.diff_toggle_btn.setText("◩ Sai khác")

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
        self.last_result = result
        self.noisy_image = result["noisy"]
        self.result_image = result["enhanced"]
        self.display_image = result["display"]
        self.showing_diff = False
        self.diff_toggle_btn.setText("◩ Sai khác")

        # 1. Ảnh gốc [Sạch]
        self.original_view.set_image(self.original_image)
        self.card_clean.set_title("1. Ảnh gốc")
        self.card_clean.set_badge("Sạch", "green")

        # 2. Ảnh nhiễu
        noise_name = self.noise_combo.currentText()
        if noise_name != "Không thêm nhiễu":
            self.noisy_view.set_image(result["transformed"])
            self.card_noisy.set_title("2. Ảnh nhiễu")
            self.card_noisy.set_badge(noise_name, "orange")
        else:
            has_transform = (self.gamma_spin.value() != 1.0 or self.contrast_spin.value() != 1.0 or self.brightness_spin.value() != 0)
            if has_transform:
                self.noisy_view.set_image(result["transformed"])
                self.card_noisy.set_title("2. Biến đổi tông")
                self.card_noisy.set_badge("Đã chỉnh tông", "cyan")
            else:
                self.noisy_view.set_image(self.original_image)
                self.card_noisy.set_title("2. Ảnh nhiễu")
                self.card_noisy.set_badge("Không nhiễu", "gray")

        # 3. Kết quả xử lý (Sau Median / Sau Mean / Sobel)
        filter_name = self.filter_combo.currentText()
        if filter_name in ("Mean", "Median"):
            self.result_title_text = f"3. Sau {filter_name}"
            self.result_badge_text = "Đã lọc"
            self.result_badge_color = "green"
        elif self.sobel_combo.currentText() != "Không dùng":
            sobel_name = self.sobel_combo.currentText()
            self.result_title_text = f"3. {sobel_name}"
            self.result_badge_text = "Biên Sobel"
            self.result_badge_color = "purple"
        else:
            self.result_title_text = "3. Ảnh kết quả"
            self.result_badge_text = "Đã xử lý"
            self.result_badge_color = "green"

        self.card_result.set_title(self.result_title_text)
        self.card_result.set_badge(self.result_badge_text, self.result_badge_color)
        self.result_view.set_image(self.display_image)

        q = result["quality"]
        if q.get("valid", False):
            psnr_val = q["psnr"]
            psnr_text = "∞" if math.isinf(psnr_val) else f"{psnr_val:.2f} dB"
            psnr_eval = "Rất tốt (>30dB)" if psnr_val >= 30 else ("Khá (20–30dB)" if psnr_val >= 20 else "Nhiều biến dạng (<20dB)")
            self.card_psnr.set_value(psnr_text, psnr_eval)

            ssim_val = q["ssim"]
            ssim_eval = "Bảo toàn cấu trúc rất cao" if ssim_val >= 0.85 else ("Cấu trúc khá tốt" if ssim_val >= 0.65 else "Biến đổi cấu trúc lớn")
            self.card_ssim.set_value(f"{ssim_val:.4f}", ssim_eval)

            self.card_mse.set_value(f"{q['mse']:.2f}", "Sai số toàn phương")

            e = q["edge"]
            self.card_edge.set_value(f"{e['f1']:.4f}", f"P: {e['precision']:.3f} | R: {e['recall']:.3f}")
            self.edge_metric_label.setText(
                f"Biên Sobel: Precision: {e['precision']:.4f}   |   Recall: {e['recall']:.4f}"
            )
        else:
            self.card_psnr.set_value("N/A")
            self.card_ssim.set_value("N/A")
            self.card_mse.set_value("N/A")
            self.card_edge.set_value("N/A")
            self.edge_metric_label.setText("Biên Sobel: Precision: N/A   |   Recall: N/A")
            QMessageBox.information(self, "Thông báo", q.get("reason", "Không thể đánh giá định lượng."))

        elapsed = result.get("elapsed_ms", 0.0)
        self.time_label.setText(f"⚡ Thời gian xử lý: {elapsed:.1f} ms (Tự cài đặt)")

        for name, image in result["edge_panels"].items():
            self.edge_views[name].set_image(image)
        self.save_btn.setEnabled(True)
        self.histogram_canvas.plot_images(self.original_image, self.result_image)
        self.statusBar().showMessage(f"Xử lý hoàn tất trong {elapsed:.1f} ms. Đã cập nhật 3 ảnh và so sánh Sobel.")

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

        # Cập nhật các thẻ thông số thực nghiệm nổi bật
        valid_psnr = [r for r in rows if not math.isinf(r["psnr_db"])]
        if valid_psnr:
            best_p = max(valid_psnr, key=lambda r: r["psnr_db"])
            self.exp_card_psnr.set_value(
                f"{best_p['psnr_db']:.2f} dB",
                f"{best_p['noise']} {best_p['noise_level']:g} → {best_p['method']} {best_p['kernel']}×{best_p['kernel']} (γ={best_p['gamma']:.1f})"
            )
        best_s = max(rows, key=lambda r: r["ssim_local"])
        self.exp_card_ssim.set_value(
            f"{best_s['ssim_local']:.4f}",
            f"{best_s['noise']} {best_s['noise_level']:g} → {best_s['method']} {best_s['kernel']}×{best_s['kernel']} (γ={best_s['gamma']:.1f})"
        )
        best_e = max(rows, key=lambda r: r["edge_f1"])
        self.exp_card_edge.set_value(
            f"{best_e['edge_f1']:.4f}",
            f"{best_e['noise']} {best_e['noise_level']:g} → {best_e['method']} {best_e['kernel']}×{best_e['kernel']} (γ={best_e['gamma']:.1f})"
        )
        total_time_s = result.get("total_time_s", 0.0)
        per_case_ms = (total_time_s / len(rows) * 1000.0) if len(rows) > 0 else 0.0
        self.exp_card_speed.set_value(
            f"{len(rows)} ca hoàn tất",
            f"Tổng: {total_time_s:.2f}s (~{per_case_ms:.1f}ms/ca)"
        )

        self.open_results_btn.setEnabled(True)
        self.experiment_progress.setValue(100)
        self.experiment_status.setText(
            f"Hoàn thành {len(rows)} ca (3 Gamma × 2 nhiễu × 2 mức × 2 bộ lọc × 3 kernel) trong {total_time_s:.2f}s. Kết quả: {result['output_dir']}"
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
