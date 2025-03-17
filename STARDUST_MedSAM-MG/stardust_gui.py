import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, 
                             QCheckBox, QTabWidget, QGroupBox, QFormLayout, QSpinBox, 
                             QDoubleSpinBox, QMessageBox, QTextEdit, QScrollArea, QGridLayout,
                             QRadioButton, QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class SettingsManager:
    def __init__(self, settings_file="stardust_gui_settings.json"):
        self.settings_file = settings_file
        self.settings = {
            "data_root": "",
            "base_output_dir": "",
            "metrics_output_dir": "",  
            "sam2_checkpoint": "",
            "medsam2_checkpoint": "",
            "model_cfg": "sam2_hiera_t.yaml",
            "bbox_shift": 5,
            "num_workers": 10,
            "save_nii": False,
            "include_ct": False,
            "use_point_prompts": False,
            "label": "1,2",  
            "visualization_slices": "0 24 48",
            "visualization_rotation": 0,
            "run_sam2_2d": True,
            "run_medsam2_2d": True,
            "run_sam2_3d": True,
            "run_medsam2_3d": True,
            "label_names": {
                "1": "GTV",
                "2": "Right Kidney",
                "3": "Spleen",
                "4": "Pancreas",
                "5": "Aorta",
                "6": "Inferior Vena Cava",
                "7": "Right Adrenal Gland",
                "8": "Left Adrenal Gland",
                "9": "Gallbladder",
                "10": "Esophagus",
                "11": "Stomach",
                "13": "Left Kidney"
            },
            "prompt_type": "box",
            "num_pos_points": 3,
            "num_neg_points": 1,
            "min_dist_from_mask_edge": 3,
            "debug_mode": False
        }
        self.load_settings()
    
    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    for key in self.settings.keys():
                        if key in loaded_settings:
                            self.settings[key] = loaded_settings[key]
                return True
            except Exception as e:
                print(f"Error loading settings: {e}")
        return False

class CommandExecutor(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
    
    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True
            )
            
            for line in iter(process.stdout.readline, ''):
                self.update_signal.emit(line.rstrip())
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                self.finished_signal.emit(True, "Command executed successfully")
            else:
                self.finished_signal.emit(False, f"Command failed with return code {return_code}")
        except Exception as e:
            self.finished_signal.emit(False, f"Error executing command: {e}")

class CommandQueue(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, commands):
        super().__init__()
        self.commands = commands
        self.current_index = 0
        self.total_commands = len(commands)
    
    def run(self):
        success = True
        error_message = ""
        
        for i, command in enumerate(self.commands):
            self.current_index = i
            self.progress_signal.emit(i + 1, self.total_commands)
            self.update_signal.emit(f"\n--- Executing command {i+1}/{self.total_commands} ---\n{command}\n")
            
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    shell=True
                )
                
                for line in iter(process.stdout.readline, ''):
                    self.update_signal.emit(line.rstrip())
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code != 0:
                    success = False
                    error_message = f"Command {i+1} failed with return code {return_code}"
                    self.update_signal.emit(f"ERROR: {error_message}")
            except Exception as e:
                success = False
                error_message = f"Error executing command {i+1}: {e}"
                self.update_signal.emit(f"EXCEPTION: {error_message}")
        
        if success:
            self.finished_signal.emit(True, "All commands executed successfully")
        else:
            self.finished_signal.emit(False, f"Command execution completed with errors: {error_message}")

class SAMInferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_manager = SettingsManager()
        self.init_ui()
        self.load_settings_to_ui()
    
    def init_ui(self):
        self.setWindowTitle("STARDUST-MedSAM2 Inference Tool")
        self.setGeometry(100, 100, 1200, 900)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        tabs = QTabWidget()
        
        inference_tab = QWidget()
        inference_layout = QVBoxLayout()
        inference_tab.setLayout(inference_layout)
        
        dir_group = QGroupBox("Directories")
        dir_layout = QVBoxLayout()
        
        data_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit()
        data_btn = QPushButton("Browse...")
        data_btn.clicked.connect(lambda: self.browse_directory(self.data_dir_edit))
        data_layout.addWidget(QLabel("Data Directory:"))
        data_layout.addWidget(self.data_dir_edit)
        data_layout.addWidget(data_btn)
        
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(lambda: self.browse_directory(self.output_dir_edit))
        output_layout.addWidget(QLabel("Base Output Directory:"))
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_btn)
        
        metrics_output_layout = QHBoxLayout()
        self.metrics_output_dir_edit = QLineEdit()
        metrics_output_btn = QPushButton("Browse...")
        metrics_output_btn.clicked.connect(lambda: self.browse_directory(self.metrics_output_dir_edit))
        metrics_output_layout.addWidget(QLabel("Metrics Output Directory:"))
        metrics_output_layout.addWidget(self.metrics_output_dir_edit)
        metrics_output_layout.addWidget(metrics_output_btn)
        
        dir_layout.addLayout(data_layout)
        dir_layout.addLayout(output_layout)
        dir_layout.addLayout(metrics_output_layout)
        dir_group.setLayout(dir_layout)
        
        checkpoint_group = QGroupBox("Model Checkpoints")
        checkpoint_layout = QVBoxLayout()
        
        sam2_layout = QHBoxLayout()
        self.sam2_ckpt_edit = QLineEdit()
        sam2_btn = QPushButton("Browse...")
        sam2_btn.clicked.connect(lambda: self.browse_file(self.sam2_ckpt_edit))
        sam2_layout.addWidget(QLabel("SAM2 Checkpoint:"))
        sam2_layout.addWidget(self.sam2_ckpt_edit)
        sam2_layout.addWidget(sam2_btn)
        
        medsam2_layout = QHBoxLayout()
        self.medsam2_ckpt_edit = QLineEdit()
        medsam2_btn = QPushButton("Browse...")
        medsam2_btn.clicked.connect(lambda: self.browse_file(self.medsam2_ckpt_edit))
        medsam2_layout.addWidget(QLabel("MedSAM2 Checkpoint:"))
        medsam2_layout.addWidget(self.medsam2_ckpt_edit)
        medsam2_layout.addWidget(medsam2_btn)
        
        config_layout = QHBoxLayout()
        self.config_edit = QLineEdit()
        config_btn = QPushButton("Browse...")
        config_btn.clicked.connect(lambda: self.browse_file(self.config_edit))
        config_layout.addWidget(QLabel("Model Config:"))
        config_layout.addWidget(self.config_edit)
        config_layout.addWidget(config_btn)
        
        checkpoint_layout.addLayout(sam2_layout)
        checkpoint_layout.addLayout(medsam2_layout)
        checkpoint_layout.addLayout(config_layout)
        checkpoint_group.setLayout(checkpoint_layout)
        
        common_group = QGroupBox("Common Parameters")
        common_layout = QFormLayout()
        
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        common_layout.addRow("Number of Workers:", self.workers_spin)
        
        self.labels_edit = QLineEdit()
        common_layout.addRow("Label (comma separated):", self.labels_edit)
        
        # Neue GUI-Elemente für Prompt-Typ und Debug-Modus
        prompt_group = QGroupBox("Prompt Settings")
        prompt_layout = QVBoxLayout()
        
        # Radio-Buttons für Prompt-Typ (Box oder Point)
        prompt_type_layout = QHBoxLayout()
        prompt_type_label = QLabel("Prompt Type:")
        self.box_prompt_radio = QRadioButton("Box")
        self.point_prompt_radio = QRadioButton("Point")
        
        # Standardmäßig Box auswählen
        self.box_prompt_radio.setChecked(True)
        
        prompt_type_layout.addWidget(prompt_type_label)
        prompt_type_layout.addWidget(self.box_prompt_radio)
        prompt_type_layout.addWidget(self.point_prompt_radio)
        prompt_type_layout.addStretch()
        
        prompt_layout.addLayout(prompt_type_layout)
        
        # Stack-Widget für die verschiedenen Prompt-Einstellungen
        self.prompt_settings_stack = QStackedWidget()
        
        # Box-Einstellungen
        box_settings_widget = QWidget()
        box_settings_layout = QFormLayout()
        
        # Box-Shift-Einstellungen
        self.box_shift_spin = QSpinBox()
        self.box_shift_spin.setRange(0, 50)
        self.box_shift_spin.setValue(5)
        self.box_shift_spin.setToolTip("Amount to expand the bounding box. Higher values may improve robustness but decrease precision.")
        box_settings_layout.addRow("Box Shift (px):", self.box_shift_spin)
        
        box_settings_widget.setLayout(box_settings_layout)
        
        # Point-Einstellungen
        point_settings_widget = QWidget()
        point_settings_layout = QFormLayout()
        
        # Einstellungen für positive und negative Punkte
        self.pos_points_spin = QSpinBox()
        self.pos_points_spin.setRange(1, 20)
        self.pos_points_spin.setValue(3)
        point_settings_layout.addRow("Number of Positive Points:", self.pos_points_spin)
        
        self.neg_points_spin = QSpinBox()
        self.neg_points_spin.setRange(0, 20)
        self.neg_points_spin.setValue(1)
        point_settings_layout.addRow("Number of Negative Points:", self.neg_points_spin)
        
        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(1, 50)
        self.min_dist_spin.setValue(3)
        point_settings_layout.addRow("Min Distance from Mask Edge (px):", self.min_dist_spin)
        
        point_settings_widget.setLayout(point_settings_layout)
        
        # Füge beide Einstellungswidgets zum Stack hinzu
        self.prompt_settings_stack.addWidget(box_settings_widget)  # Index 0
        self.prompt_settings_stack.addWidget(point_settings_widget)  # Index 1
        
        # Verbinde Radio-Buttons mit Stack-Widget
        self.box_prompt_radio.toggled.connect(lambda checked: self.prompt_settings_stack.setCurrentIndex(0) if checked else None)
        self.point_prompt_radio.toggled.connect(lambda checked: self.prompt_settings_stack.setCurrentIndex(1) if checked else None)
        
        prompt_layout.addWidget(self.prompt_settings_stack)
        
        # Debug-Mode-Checkbox
        debug_layout = QHBoxLayout()
        self.debug_mode_check = QCheckBox("Debug Mode (Save Prompt Visualizations)")
        debug_layout.addWidget(self.debug_mode_check)
        
        prompt_layout.addLayout(debug_layout)
        prompt_group.setLayout(prompt_layout)
        
        self.save_nii_check = QCheckBox("Save NII")
        self.include_ct_check = QCheckBox("Include CT")
        
        options_layout = QVBoxLayout()
        options_row1 = QHBoxLayout()
        options_row1.addWidget(self.save_nii_check)
        options_row1.addWidget(self.include_ct_check)
        options_row1.addStretch()
        
        options_layout.addLayout(options_row1)
        
        common_layout.addRow("Options:", options_layout)
        
        common_group.setLayout(common_layout)
        
        inference_select_group = QGroupBox("Inference Selection")
        inference_select_layout = QGridLayout()
        
        self.sam2_2d_check = QCheckBox("SAM2 2D")
        self.medsam2_2d_check = QCheckBox("MedSAM2 2D")
        self.sam2_3d_check = QCheckBox("SAM2 3D")
        self.medsam2_3d_check = QCheckBox("MedSAM2 3D")
        
        inference_select_layout.addWidget(QLabel("2D:"), 0, 0)
        inference_select_layout.addWidget(self.sam2_2d_check, 0, 1)
        inference_select_layout.addWidget(self.medsam2_2d_check, 0, 2)
        inference_select_layout.addWidget(QLabel("3D:"), 1, 0)
        inference_select_layout.addWidget(self.sam2_3d_check, 1, 1)
        inference_select_layout.addWidget(self.medsam2_3d_check, 1, 2)
        
        inference_select_group.setLayout(inference_select_layout)
        
        vis_group = QGroupBox("Visualization Settings")
        vis_layout = QGridLayout()
        vis_layout.setColumnStretch(1, 1)
        
        slices_label = QLabel("Slice Indices:")
        self.vis_slices_edit = QLineEdit()
        self.vis_slices_edit.setPlaceholderText("Optional: e.g. 0 24 48 (leave empty for auto-selection)")
        vis_layout.addWidget(slices_label, 0, 0)
        vis_layout.addWidget(self.vis_slices_edit, 0, 1)
        
        rotation_label = QLabel("Image Rotation:")
        self.vis_rotation_combo = QComboBox()
        self.vis_rotation_combo.addItem("0° (keine Rotation)", 0)
        self.vis_rotation_combo.addItem("90°", 1)
        self.vis_rotation_combo.addItem("180°", 2)
        self.vis_rotation_combo.addItem("270°", 3)
        vis_layout.addWidget(rotation_label, 1, 0)
        vis_layout.addWidget(self.vis_rotation_combo, 1, 1)
        
        vis_group.setLayout(vis_layout)
        
        buttons_group = QGroupBox("Execute")
        buttons_layout = QVBoxLayout()
        
        run_all_inference_btn = QPushButton("Run All Selected Inference")
        run_all_inference_btn.clicked.connect(self.run_all_inference)
        run_all_inference_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        
        compute_all_metrics_btn = QPushButton("Compute All Metrics")
        compute_all_metrics_btn.clicked.connect(self.compute_all_metrics)
        
        visualize_metrics_btn = QPushButton("Visualize Individual Metrics")
        visualize_metrics_btn.clicked.connect(self.visualize_metrics)
        
        visualize_comparison_btn = QPushButton("Visualize Comparison")
        visualize_comparison_btn.clicked.connect(self.visualize_comparison)
        
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        
        buttons_layout.addWidget(run_all_inference_btn)
        buttons_layout.addWidget(compute_all_metrics_btn)
        buttons_layout.addWidget(visualize_metrics_btn)
        buttons_layout.addWidget(visualize_comparison_btn)
        buttons_layout.addWidget(save_settings_btn)
        buttons_group.setLayout(buttons_layout)
        
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        console_layout.addWidget(self.console_output)
        
        console_group.setLayout(console_layout)
        
        inference_layout.addWidget(dir_group)
        inference_layout.addWidget(checkpoint_group)
        inference_layout.addWidget(common_group)
        inference_layout.addWidget(prompt_group)
        inference_layout.addWidget(inference_select_group)
        inference_layout.addWidget(vis_group)
        
        bottom_layout = QHBoxLayout()
        buttons_console_layout = QVBoxLayout()
        buttons_console_layout.addWidget(buttons_group)
        buttons_console_layout.addStretch()
        
        bottom_layout.addLayout(buttons_console_layout)
        bottom_layout.addWidget(console_group, 2)  
        
        inference_layout.addLayout(bottom_layout)
        
        tabs.addTab(inference_tab, "Inference")
        
        label_tab = QWidget()
        label_layout = QVBoxLayout()
        label_tab.setLayout(label_layout)
        
        label_management_group = QGroupBox("Label Management")
        label_management_layout = QVBoxLayout()
        
        intro_label = QLabel("Hier können Sie die Namen für jedes Label definieren. Diese werden für die korrekte Speicherung der Segmentierungen verwendet.")
        intro_label.setWordWrap(True)
        label_management_layout.addWidget(intro_label)
        
        label_grid = QGridLayout()
        
        self.label_edits = {}
        row = 0
        col = 0
        max_cols = 2  
        
        label_ids = sorted([int(k) for k in self.settings_manager.settings["label_names"].keys()])
        
        for label_id in label_ids:
            label_id_str = str(label_id)
            label_form = QFormLayout()
            self.label_edits[label_id_str] = QLineEdit()
            self.label_edits[label_id_str].setText(self.settings_manager.settings["label_names"][label_id_str])
            label_form.addRow(f"Label {label_id_str}:", self.label_edits[label_id_str])
            
            label_grid.addLayout(label_form, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        label_management_layout.addLayout(label_grid)
        
        save_labels_btn = QPushButton("Label-Namen speichern")
        save_labels_btn.clicked.connect(self.save_label_names)
        label_management_layout.addWidget(save_labels_btn)
        
        label_management_group.setLayout(label_management_layout)
        label_layout.addWidget(label_management_group)
        
        tabs.addTab(label_tab, "Label Management")
        
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        metrics_button_layout = QHBoxLayout()
        self.compute_metrics_btn = QPushButton("Compute Metrics")
        self.compute_metrics_btn.clicked.connect(self.compute_all_metrics)
        self.visualize_metrics_btn = QPushButton("Visualize Individual Metrics")
        self.visualize_metrics_btn.clicked.connect(self.visualize_metrics)
        self.visualize_comparison_btn = QPushButton("Visualize Model Comparison")
        self.visualize_comparison_btn.clicked.connect(self.visualize_comparison)
        metrics_button_layout.addWidget(self.compute_metrics_btn)
        metrics_button_layout.addWidget(self.visualize_metrics_btn)
        metrics_button_layout.addWidget(self.visualize_comparison_btn)
        metrics_layout.addLayout(metrics_button_layout)
        
        metrics_results_group = QGroupBox("Metrics Results")
        metrics_results_layout = QVBoxLayout(metrics_results_group)
        
        metrics_results_controls = QHBoxLayout()
        self.refresh_metrics_btn = QPushButton("Refresh Results")
        self.refresh_metrics_btn.clicked.connect(self.refresh_metrics_results)
        metrics_results_controls.addWidget(self.refresh_metrics_btn)
        metrics_results_layout.addLayout(metrics_results_controls)
        
        self.metrics_results_display = QTextEdit()
        self.metrics_results_display.setReadOnly(True)
        metrics_results_layout.addWidget(self.metrics_results_display)
        
        metrics_layout.addWidget(metrics_results_group)
        
        tabs.addTab(metrics_tab, "Metrics")
        
        main_layout.addWidget(tabs)
    
    def browse_directory(self, line_edit):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)
    
    def browse_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            line_edit.setText(file_path)
    
    def save_settings(self):
        self.settings_manager.settings["data_root"] = self.data_dir_edit.text()
        self.settings_manager.settings["base_output_dir"] = self.output_dir_edit.text()
        self.settings_manager.settings["metrics_output_dir"] = self.metrics_output_dir_edit.text()  
        self.settings_manager.settings["sam2_checkpoint"] = self.sam2_ckpt_edit.text()
        self.settings_manager.settings["medsam2_checkpoint"] = self.medsam2_ckpt_edit.text()
        self.settings_manager.settings["model_cfg"] = self.config_edit.text()
        self.settings_manager.settings["num_workers"] = self.workers_spin.value()
        self.settings_manager.settings["save_nii"] = self.save_nii_check.isChecked()
        self.settings_manager.settings["include_ct"] = self.include_ct_check.isChecked()
        self.settings_manager.settings["label"] = self.labels_edit.text()
        self.settings_manager.settings["visualization_slices"] = self.vis_slices_edit.text()
        self.settings_manager.settings["visualization_rotation"] = self.vis_rotation_combo.currentData()
        self.settings_manager.settings["run_sam2_2d"] = self.sam2_2d_check.isChecked()
        self.settings_manager.settings["run_medsam2_2d"] = self.medsam2_2d_check.isChecked()
        self.settings_manager.settings["run_sam2_3d"] = self.sam2_3d_check.isChecked()
        self.settings_manager.settings["run_medsam2_3d"] = self.medsam2_3d_check.isChecked()
        self.settings_manager.settings["prompt_type"] = "box" if self.box_prompt_radio.isChecked() else "point"
        self.settings_manager.settings["box_shift"] = self.box_shift_spin.value()
        self.settings_manager.settings["num_pos_points"] = self.pos_points_spin.value()
        self.settings_manager.settings["num_neg_points"] = self.neg_points_spin.value()
        self.settings_manager.settings["min_dist_from_mask_edge"] = self.min_dist_spin.value()
        self.settings_manager.settings["debug_mode"] = self.debug_mode_check.isChecked()
        
        for label_id, edit in self.label_edits.items():
            self.settings_manager.settings["label_names"][label_id] = edit.text()
        
        if self.settings_manager.save_settings():
            QMessageBox.information(self, "Success", "Settings saved successfully.")
        else:
            QMessageBox.warning(self, "Error", "Failed to save settings.")
    
    def load_settings_to_ui(self):
        self.data_dir_edit.setText(self.settings_manager.settings["data_root"])
        self.output_dir_edit.setText(self.settings_manager.settings["base_output_dir"])
        self.metrics_output_dir_edit.setText(self.settings_manager.settings.get("metrics_output_dir", ""))  
        self.sam2_ckpt_edit.setText(self.settings_manager.settings["sam2_checkpoint"])
        self.medsam2_ckpt_edit.setText(self.settings_manager.settings["medsam2_checkpoint"])
        self.config_edit.setText(self.settings_manager.settings["model_cfg"])
        self.workers_spin.setValue(self.settings_manager.settings["num_workers"])
        self.save_nii_check.setChecked(self.settings_manager.settings["save_nii"])
        self.include_ct_check.setChecked(self.settings_manager.settings["include_ct"])
        self.labels_edit.setText(self.settings_manager.settings["label"])
        self.vis_slices_edit.setText(self.settings_manager.settings["visualization_slices"])
        
        # Setze die Rotation, falls in den Einstellungen vorhanden
        rotation_value = self.settings_manager.settings.get("visualization_rotation", 0)
        index = self.vis_rotation_combo.findData(rotation_value)
        if index >= 0:
            self.vis_rotation_combo.setCurrentIndex(index)
        
        self.sam2_2d_check.setChecked(self.settings_manager.settings["run_sam2_2d"])
        self.medsam2_2d_check.setChecked(self.settings_manager.settings["run_medsam2_2d"])
        self.sam2_3d_check.setChecked(self.settings_manager.settings["run_sam2_3d"])
        self.medsam2_3d_check.setChecked(self.settings_manager.settings["run_medsam2_3d"])
        
        prompt_type = self.settings_manager.settings.get("prompt_type", "box")
        if prompt_type == "box":
            self.box_prompt_radio.setChecked(True)
            self.prompt_settings_stack.setCurrentIndex(0)
        else:
            self.point_prompt_radio.setChecked(True)
            self.prompt_settings_stack.setCurrentIndex(1)
        
        self.box_shift_spin.setValue(self.settings_manager.settings.get("box_shift", 5))
        self.pos_points_spin.setValue(self.settings_manager.settings.get("num_pos_points", 3))
        self.neg_points_spin.setValue(self.settings_manager.settings.get("num_neg_points", 1))
        self.min_dist_spin.setValue(self.settings_manager.settings.get("min_dist_from_mask_edge", 3))
        self.debug_mode_check.setChecked(self.settings_manager.settings.get("debug_mode", False))
    
    def get_output_path(self, dimension, model):
        """
        Generiert den Ausgabepfad basierend auf dem Basisverzeichnis, der Dimension und dem Modell.
        Gibt None zurück, wenn kein Basisverzeichnis angegeben ist.
        """
        base_dir = self.output_dir_edit.text().strip()
        if not base_dir:
            return None
        return os.path.join(base_dir, dimension, model.lower())
    
    def build_inference_command(self, model, dimension):
        """
        Erstellt den Inferenzbefehl für ein bestimmtes Modell und eine bestimmte Dimension.
        Überprüft zuerst, ob ein gültiges Ausgabeverzeichnis vorhanden ist.
        """
        # Überprüfe, ob die Eingabeverzeichnisse vorhanden sind
        data_root = self.data_dir_edit.text().strip()
        if not data_root:
            QMessageBox.warning(self, "Warning", "Please specify a data directory")
            return None
            
        output_dir = self.get_output_path(dimension, model)
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Please specify a base output directory")
            return None
            
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            QMessageBox.critical(self, "Error", f"Permission denied when creating directory: {output_dir}")
            return None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating output directory: {str(e)}")
            return None
        
        if model == "MedSAM2" and dimension == "2D":
            script_path = os.path.join("eval_inference", "infer_medsam2_2d.py")
            command = [
                f"python {script_path}",
                f"-data_root {self.data_dir_edit.text()}",
                f"-pred_save_dir {output_dir}",
                f"-sam2_checkpoint {self.sam2_ckpt_edit.text()}",
                f"-medsam2_checkpoint {self.medsam2_ckpt_edit.text()}",
                f"-model_cfg {self.config_edit.text()}",
                f"-bbox_shift {self.box_shift_spin.value()}",
                f"-num_workers {self.workers_spin.value()}"
            ]
            
            if self.save_nii_check.isChecked():
                command.append("--save_nii")
            
            if self.include_ct_check.isChecked():
                command.append("--include_ct")
            
            if self.point_prompt_radio.isChecked():  
                command.append("-prompt_type point")
                command.append(f"-num_pos_points {self.pos_points_spin.value()}")
                command.append(f"-num_neg_points {self.neg_points_spin.value()}")
                command.append(f"-min_dist_from_edge {self.min_dist_spin.value()}")
            else:
                command.append("-prompt_type box")
            
            if self.debug_mode_check.isChecked():
                command.append("-debug_mode")
            
            if self.labels_edit.text():
                command.append(f"--label {self.labels_edit.text()}")
                
        elif model == "SAM2" and dimension == "2D":
            script_path = os.path.join("eval_inference", "infer_sam2_2d.py")
            command = [
                f"python {script_path}",
                f"-data_root {self.data_dir_edit.text()}",
                f"-pred_save_dir {output_dir}",
                f"-sam2_checkpoint {self.sam2_ckpt_edit.text()}",
                f"-model_cfg {self.config_edit.text()}",
                f"-bbox_shift {self.box_shift_spin.value()}",
                f"-num_workers {self.workers_spin.value()}"
            ]
            
            if self.save_nii_check.isChecked():
                command.append("--save_nii")
            
            if self.include_ct_check.isChecked():
                command.append("--include_ct")
            
            if self.point_prompt_radio.isChecked():  
                command.append("-prompt_type point")
                command.append(f"-num_pos_points {self.pos_points_spin.value()}")
                command.append(f"-num_neg_points {self.neg_points_spin.value()}")
                command.append(f"-min_dist_from_edge {self.min_dist_spin.value()}")
            else:
                command.append("-prompt_type box")
            
            if self.debug_mode_check.isChecked():
                command.append("-debug_mode")
            
            if self.labels_edit.text():
                command.append(f"--label {self.labels_edit.text()}")
                
        elif model == "SAM2" and dimension == "3D":
            script_path = os.path.join("eval_inference", "infer_SAM2_3D.py")
            command = [
                f"python {script_path}",
                f"--checkpoint {self.sam2_ckpt_edit.text()}",
                f"--cfg {self.config_edit.text()}",
                f"--imgs_path {self.data_dir_edit.text()}",
                f"--gts_path {self.data_dir_edit.text()}",
                f"--pred_save_dir {output_dir}"
            ]
            
            vis_dir = os.path.join("results", "overlay_3D", "sam2")
            os.makedirs(vis_dir, exist_ok=True)
            command.append(f"--save_overlay")
            command.append(f"--png_save_dir {vis_dir}")
            
            if self.save_nii_check.isChecked():
                nifti_dir = os.path.join("results", "segs_nifti", "sam2")
                os.makedirs(nifti_dir, exist_ok=True)
                command.append("--save_nifti")
                command.append(f"--nifti_path {nifti_dir}")
            
            if self.labels_edit.text():
                command.append(f"--label {self.labels_edit.text()}")
        
        return " ".join(command)
    
    def build_metrics_command(self, dimension, model):
        """
        Erstellt den Befehl zur Berechnung von Metriken für ein bestimmtes Modell und eine Dimension.
        """
        # Überprüfe, ob die Eingabeverzeichnisse vorhanden sind
        data_root = self.data_dir_edit.text().strip()
        if not data_root:
            return None
            
        output_dir = self.get_output_path(dimension, model)
        if not output_dir:
            return None
            
        metrics_output_dir = self.metrics_output_dir_edit.text().strip()
        if metrics_output_dir:
            metrics_dir = os.path.join(metrics_output_dir, dimension, model.lower())
        else:
            metrics_dir = os.path.join("metric_results", dimension, model.lower())
        
        try:
            os.makedirs(metrics_dir, exist_ok=True)
        except Exception as e:
            self.console_output.append(f"Error creating metrics directory {metrics_dir}: {str(e)}")
            return None
        
        command = [
            f"python ./eval_metrics/compute_metrics.py",
            f"-s {output_dir}",
            f"-g {self.data_dir_edit.text()}",
            f"-csv_dir {metrics_dir}",
            f"-nw {self.workers_spin.value()}"
        ]
        
        return " ".join(command)
    
    def run_all_inference(self):
        self.console_output.clear()
        self.console_output.append("Starting inference...")
        
        commands = []
        
        if self.sam2_2d_check.isChecked():
            cmd = self.build_inference_command("SAM2", "2D")
            if cmd:
                commands.append(cmd)
        
        if self.medsam2_2d_check.isChecked():
            cmd = self.build_inference_command("MedSAM2", "2D")
            if cmd:
                commands.append(cmd)
        
        if self.sam2_3d_check.isChecked():
            cmd = self.build_inference_command("SAM2", "3D")
            if cmd:
                commands.append(cmd)
        
        if self.medsam2_3d_check.isChecked():
            # MedSAM2 3D hat andere Parameter
            output_dir = self.get_output_path("3D", "MedSAM2")
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    vis_dir = os.path.join("results", "overlay_3D", "medsam2")
                    os.makedirs(vis_dir, exist_ok=True)
                    command = [
                        f"python ./eval_inference/infer_MedSAM2_3D.py",
                        f"--checkpoint {self.medsam2_ckpt_edit.text()}",
                        f"--cfg {self.config_edit.text()}",
                        f"--imgs_path {self.data_dir_edit.text()}",
                        f"--gts_path {self.data_dir_edit.text()}",
                        f"--pred_save_dir {output_dir}",
                        f"--save_overlay",
                        f"--png_save_dir {vis_dir}"
                    ]
                    
                    if self.save_nii_check.isChecked():
                        nifti_dir = os.path.join("results", "segs_nifti", "medsam2")
                        os.makedirs(nifti_dir, exist_ok=True)
                        command.append("--save_nifti")
                        command.append(f"--nifti_path {nifti_dir}")
                    
                    if self.labels_edit.text():
                        command.append(f"--label {self.labels_edit.text()}")
                    
                    commands.append(" ".join(command))
                except PermissionError:
                    QMessageBox.critical(self, "Error", f"Permission denied when creating directory: {output_dir}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error creating output directory: {str(e)}")
        
        if not commands:
            self.console_output.append("No inference options selected!")
            return
        
        self.command_queue = CommandQueue(commands)
        self.command_queue.update_signal.connect(self.update_console)
        self.command_queue.progress_signal.connect(self.update_progress)
        self.command_queue.finished_signal.connect(self.command_queue_finished)
        self.command_queue.start()
    
    def compute_all_metrics(self):
        self.console_output.clear()
        self.console_output.append("Computing metrics...")
        
        commands = []
        
        # Nur für aktivierte Inferenz-Modelle Metriken berechnen
        if self.sam2_2d_check.isChecked():
            output_dir = self.get_output_path("2D", "SAM2")
            if output_dir and os.path.exists(output_dir) and os.listdir(output_dir):
                cmd = self.build_metrics_command("2D", "SAM2")
                if cmd:
                    commands.append(cmd)
        
        if self.medsam2_2d_check.isChecked():
            output_dir = self.get_output_path("2D", "MedSAM2")
            if output_dir and os.path.exists(output_dir) and os.listdir(output_dir):
                cmd = self.build_metrics_command("2D", "MedSAM2")
                if cmd:
                    commands.append(cmd)
        
        if self.sam2_3d_check.isChecked():
            output_dir = self.get_output_path("3D", "SAM2")
            if output_dir and os.path.exists(output_dir) and os.listdir(output_dir):
                cmd = self.build_metrics_command("3D", "SAM2")
                if cmd:
                    commands.append(cmd)
        
        if self.medsam2_3d_check.isChecked():
            output_dir = self.get_output_path("3D", "MedSAM2")
            if output_dir and os.path.exists(output_dir) and os.listdir(output_dir):
                cmd = self.build_metrics_command("3D", "MedSAM2")
                if cmd:
                    commands.append(cmd)
        
        if not commands:
            self.console_output.append("No completed inference outputs found for selected models!")
            return
        
        self.command_queue = CommandQueue(commands)
        self.command_queue.update_signal.connect(self.update_console)
        self.command_queue.progress_signal.connect(self.update_progress)
        self.command_queue.finished_signal.connect(self.command_queue_finished)
        self.command_queue.start()
    
    def visualize_metrics(self):
        self.console_output.clear()
        self.console_output.append("Visualizing individual metrics...")
        
        # Verwende das eingegebene Metrics-Verzeichnis, wenn vorhanden
        metrics_dir = self.metrics_output_dir_edit.text().strip()
        if metrics_dir:
            command = f"python ./eval_metrics/visualizePlots.py --metrics_dir \"{metrics_dir}\""
        else:
            command = f"python ./eval_metrics/visualizePlots.py"
        
        self.executor = CommandExecutor(command)
        self.executor.update_signal.connect(self.update_console)
        self.executor.finished_signal.connect(self.command_finished)
        self.executor.start()
    
    def visualize_comparison(self):
        self.console_output.clear()
        self.console_output.append("Visualizing comparisons...")
        
        commands = []
        
        # Verwende das eingegebene Metrics-Verzeichnis als Basis, wenn vorhanden
        metrics_dir = self.metrics_output_dir_edit.text().strip()
        if not metrics_dir:
            metrics_dir = os.path.join(os.getcwd(), "metric_results")
        
        # Hole die Rotation aus der ComboBox
        rotation_value = self.vis_rotation_combo.currentData()
        
        for dimension in ["2D", "3D"]:
            # Erstelle die Pfade basierend auf dem Metrics-Verzeichnis
            sam2_dir = self.get_output_path(dimension, "SAM2")
            medsam2_dir = self.get_output_path(dimension, "MedSAM2")
            
            if os.path.exists(sam2_dir) and os.path.exists(medsam2_dir):
                # Erstelle das Ausgabeverzeichnis innerhalb des Metrics-Verzeichnisses
                vis_dir = os.path.join(metrics_dir, dimension, "comparison", "visualizations")
                try:
                    os.makedirs(vis_dir, exist_ok=True)
                
                    # Bereite die Slices vor
                    slices_str = self.vis_slices_edit.text().strip()
                    
                    vis_command = [
                        f"python ./eval_metrics/visualize_segmentations.py",
                        f"-g {self.data_dir_edit.text()}",
                        f"-s2 {sam2_dir}",
                        f"-ms2 {medsam2_dir}",
                        f"-o {vis_dir}",
                        f"-m {metrics_dir}",
                        f"--rotate {rotation_value}"
                    ]
                    
                    # Füge Slices hinzu, wenn sie angegeben wurden
                    if slices_str:
                        vis_command.append(f"-s {' '.join(slices_str.split())}")
                    
                    vis_command.append(f"-l 1")
                    
                    commands.append(" ".join(vis_command))
                except Exception as e:
                    self.console_output.append(f"Error creating visualization directory {vis_dir}: {str(e)}")
        
        if not commands:
            self.console_output.append("No complete model pairs found for comparison!")
            return
        
        self.command_queue = CommandQueue(commands)
        self.command_queue.update_signal.connect(self.update_console)
        self.command_queue.progress_signal.connect(self.update_progress)
        self.command_queue.finished_signal.connect(self.command_queue_finished)
        self.command_queue.start()
    
    def update_console(self, text):
        self.console_output.append(text)
        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )
    
    def update_progress(self, current, total):
        self.console_output.append(f"\nProgress: {current}/{total} commands completed\n")
    
    def command_finished(self, success, message):
        if success:
            self.console_output.append("\n" + message)
            self.console_output.append("Command completed successfully!")
        else:
            self.console_output.append("\n" + message)
            self.console_output.append("Command failed!")
    
    def command_queue_finished(self, success, message):
        if success:
            self.console_output.append("\n" + message)
            self.console_output.append("All commands completed successfully!")
            
            if "Computing metrics" in message or "metric" in message.lower():
                self.refresh_metrics_results()
        else:
            self.console_output.append("\n" + message)
            self.console_output.append("Command queue completed with errors!")
    
    def save_label_names(self):
        for label_id, label_edit in self.label_edits.items():
            self.settings_manager.settings["label_names"][label_id] = label_edit.text()
        
        if self.settings_manager.save_settings():
            QMessageBox.information(self, "Label Names Saved", "Label names saved successfully!")
        else:
            QMessageBox.warning(self, "Error", "Failed to save label names")
    
    def refresh_metrics_results(self):
        self.metrics_results_display.clear()
        
        html_content = "<style>table {border-collapse: collapse; width: 100%;} th, td {padding: 8px; text-align: left; border-bottom: 1px solid #ddd;} th {background-color: #f2f2f2;} tr:hover {background-color: #f5f5f5;}</style>"
        
        html_content += "<h1>Individual Model Results</h1>"
        
        models_to_check = []
        if self.sam2_2d_check.isChecked():
            models_to_check.append(("2D", "sam2"))
        if self.medsam2_2d_check.isChecked():
            models_to_check.append(("2D", "medsam2"))
        if self.sam2_3d_check.isChecked():
            models_to_check.append(("3D", "sam2"))
        if self.medsam2_3d_check.isChecked():
            models_to_check.append(("3D", "medsam2"))
        
        all_metrics = {}
        model_data = {}
        
        for dimension, model in models_to_check:
            metrics_dir = os.path.join("metric_results", dimension, model)
            if not os.path.exists(metrics_dir):
                continue
            
            html_content += f"<h2>{model.upper()} {dimension} Results</h2>"
            
            csv_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
            
            if not csv_files:
                html_content += f"<p>No metrics found for {model} {dimension}</p>"
                continue
            
            for csv_file in csv_files:
                metric_type = csv_file.replace('.csv', '')
                html_content += f"<h3>{metric_type}</h3>"
                
                try:
                    import pandas as pd
                    import numpy as np
                    csv_path = os.path.join(metrics_dir, csv_file)
                    
                    if metric_type in ['dsc_summary', 'nsd_summary']:
                        df = pd.read_csv(csv_path, header=0)
                        
                        if '' in df.columns or df.columns[0] == 'Unnamed: 0':
                            df = pd.read_csv(csv_path, index_col=0, header=None)
                            df = df.T
                            df.insert(0, 'metric', metric_type)
                            
                            value_cols = []
                            for col in df.columns:
                                if col != 'metric' and df[col].notna().any() and (df[col] != 0).any():
                                    value_cols.append(col)
                            
                            filtered_df = df[['metric'] + value_cols]
                            table_html = filtered_df.to_html(index=False)
                            html_content += table_html
                            continue
                    
                    df = pd.read_csv(csv_path)
                    
                    if 'case' not in df.columns:
                        df = pd.read_csv(csv_path, index_col=0)
                        df.reset_index(inplace=True)
                    
                    cols_to_keep = ['case']
                    for col in df.columns:
                        if col != 'case':
                            if df[col].notna().any() and (df[col] != 0).any():
                                cols_to_keep.append(col)
                    
                    filtered_df = df[cols_to_keep]
                    
                    model_key = f"{model}_{dimension}"
                    if metric_type not in all_metrics:
                        all_metrics[metric_type] = []
                    all_metrics[metric_type].append(model_key)
                    model_data[(model_key, metric_type)] = filtered_df
                    
                    table_html = filtered_df.to_html(index=False)
                    html_content += table_html
                except Exception as e:
                    html_content += f"<p>Error loading {csv_file}: {str(e)}</p>"
        
        html_content += "<h1>Model Comparison</h1>"
        
        for metric_type, models_list in all_metrics.items():
            if len(models_list) > 1:  
                html_content += f"<h2>{metric_type} Comparison</h2>"
                
                try:
                    import pandas as pd
                    import numpy as np
                    
                    comparison_data = {}
                    cases = set()
                    
                    for model_key in models_list:
                        if (model_key, metric_type) in model_data:
                            df = model_data[(model_key, metric_type)]
                            
                            if 'case' in df.columns:
                                cases.update(df['case'].tolist())
                                
                                for col in df.columns:
                                    if col != 'case':
                                        if df[col].notna().any() and (df[col] != 0).any():
                                            new_col = f"{model_key}_{col}"
                                            value_dict = dict(zip(df['case'], df[col]))
                                            comparison_data[new_col] = value_dict
                    
                    if cases:
                        comparison_df = pd.DataFrame({'case': sorted(list(cases))})
                        
                        for col, value_dict in comparison_data.items():
                            comparison_df[col] = comparison_df['case'].map(value_dict).fillna(np.nan)
                        
                        non_empty_columns = ['case']
                        for col in comparison_df.columns:
                            if col != 'case' and comparison_df[col].notna().sum() > 0 and (comparison_df[col] != 0).any():
                                non_empty_columns.append(col)
                        
                        comparison_df = comparison_df[non_empty_columns]
                        
                        if len(non_empty_columns) > 1:
                            table_html = comparison_df.to_html(index=False)
                            html_content += table_html
                            
                            html_content += "<h3>Average Values by Model</h3>"
                            
                            model_averages = {}
                            for col in comparison_df.columns:
                                if col != 'case':
                                    model_name = "_".join(col.split("_")[:2])
                                    if model_name not in model_averages:
                                        model_averages[model_name] = []
                                    values = comparison_df[col].replace(0, np.nan)  
                                    avg_value = values.mean(skipna=True)
                                    if not pd.isna(avg_value):
                                        model_averages[model_name].append(avg_value)
                            
                            if model_averages:
                                avg_data = {
                                    'Model': list(model_averages.keys()),
                                    f'Average {metric_type}': [
                                        sum(values)/len(values) if values else np.nan 
                                        for values in model_averages.values()
                                    ]
                                }
                                avg_df = pd.DataFrame(avg_data)
                                avg_df = avg_df.sort_values(f'Average {metric_type}', ascending=False)
                                
                                html_content += avg_df.to_html(index=False)
                        else:
                            html_content += "<p>No comparable data found between models for this metric.</p>"
                    else:
                        html_content += "<p>No cases found to compare for this metric.</p>"
                except Exception as e:
                    html_content += f"<p>Error creating comparison for {metric_type}: {str(e)}</p>"
        
        if len(all_metrics) == 0:
            html_content += "<p>No metrics found for selected models. Please run 'Compute Metrics' first.</p>"
        
        self.metrics_results_display.setHtml(html_content)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMInferenceGUI()
    window.show()
    sys.exit(app.exec_())