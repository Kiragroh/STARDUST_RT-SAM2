import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import csv
from datetime import datetime
import re

class CTBrowserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("STARDUST-Case-Selector")
        
        # Maximale Fenstergröße basierend auf Bildschirmauflösung
        window_width = min(1200, root.winfo_screenwidth() - 100)
        window_height = min(800, root.winfo_screenheight() - 100)
        self.root.geometry(f"{window_width}x{window_height}")
        
        self.root.configure(bg="#f0f0f0")
        
        # Load settings
        self.load_settings()
        
        # Initialize variables
        self.current_index = -1
        self.dataframe = None
        self.filtered_data = None
        self.current_image = None
        self.current_image_km = None
        self.current_image_mr = None
        self.image_object = None
        self.image_object_km = None
        self.image_object_mr = None
        self.total_images = 0
        self.remaining_images = 0
        self.accepted_images = set()
        self.refinement_mode = False
        self.refinement_data = None
        self.extra_label = None
        self.show_rejected = False
        self.predefined_labels = [
            "Metastase-Kopf", "Meningeom", "MetBett-Kopf", "Glomus", "Lymphom", 
            "Lunge", "Mediastinum", "NNiere", "Bauchwand", "Sonstiges", "Orbita", "Pankreas",
            "Oesophagus", "Glio", "Astrozytom", "Milz", "Knochen", "BC", "Extrem", "Axilla"
        ]
        
        # Filter variables aus gespeicherten Einstellungen laden
        self.selected_labels = set(self.settings.get("selected_labels", []))
        self.selected_labels2 = set(self.settings.get("selected_labels2", []))
        
        # Dynamische Bildgrößenanpassung - Bindung an Fenstergröße
        self.root.bind("<Configure>", self.on_window_resize)
        
        # Create UI
        self.create_ui()
        
        # Check if output file exists and load accepted images
        self.load_accepted_images()
        
        # Load Excel data and update filter options on startup
        self.load_excel_and_update_filters()
    
    def load_settings(self):
        try:
            with open("settings.json", "r") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            self.settings = {
                "input_excel": "",
                "output_csv": "accepted_images.csv",
                "images_folder": "",
                "filter_mode": "include",
                "selected_labels": [],
                "selected_labels2": []
            }
            self.save_settings()
    
    def save_settings(self):
        # Speichern der aktuellen Filter vor dem Schreiben der Einstellungen
        if hasattr(self, 'filter_mode_var'):
            self.settings["filter_mode"] = self.filter_mode_var.get()
        if hasattr(self, 'selected_labels'):
            self.settings["selected_labels"] = list(self.selected_labels)
        if hasattr(self, 'selected_labels2'):
            self.settings["selected_labels2"] = list(self.selected_labels2)
            
        with open("./STARDUST_CaseSelector/settings.json", "w") as f:
            json.dump(self.settings, f, indent=4)
    
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel - enthält Einstellungen und Filter (immer an oberster Stelle)
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, side=tk.TOP, pady=5)
        
        # Bottom panel - enthält die Bedienelemente (immer am untersten Rand sichtbar)
        bottom_panel = ttk.Frame(main_frame)
        bottom_panel.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(top_panel, text="Einstellungen")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Input Excel File
        ttk.Label(settings_frame, text="Excel-Datei:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_excel_var = tk.StringVar(value=self.settings.get("input_excel", ""))
        ttk.Entry(settings_frame, textvariable=self.input_excel_var, width=150).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Durchsuchen", command=self.browse_excel).grid(row=0, column=2, padx=5, pady=5)
        
        # Output CSV File
        ttk.Label(settings_frame, text="Ausgabe-CSV:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_csv_var = tk.StringVar(value=self.settings.get("output_csv", ""))
        ttk.Entry(settings_frame, textvariable=self.output_csv_var, width=150).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Durchsuchen", command=self.browse_csv).grid(row=1, column=2, padx=5, pady=5)
        
        # Images Folder
        ttk.Label(settings_frame, text="Bilder-Ordner:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.images_folder_var = tk.StringVar(value=self.settings.get("images_folder", ""))
        ttk.Entry(settings_frame, textvariable=self.images_folder_var, width=150).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Durchsuchen", command=self.browse_folder).grid(row=2, column=2, padx=5, pady=5)
        
        # Filter Frame
        filter_frame = ttk.LabelFrame(top_panel, text="Filter")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Filter Mode
        filter_controls_frame = ttk.Frame(filter_frame)
        filter_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_controls_frame, text="Filtermodus:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.filter_mode_var = tk.StringVar(value=self.settings.get("filter_mode", "include"))
        filter_mode_frame = ttk.Frame(filter_controls_frame)
        filter_mode_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(filter_mode_frame, text="Einschließen", variable=self.filter_mode_var, 
                       value="include").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(filter_mode_frame, text="Ausschließen", variable=self.filter_mode_var, 
                       value="exclude").pack(side=tk.LEFT, padx=5)
        
        # Label Filter
        ttk.Label(filter_controls_frame, text="Label:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(filter_controls_frame, textvariable=self.label_var, width=20)
        self.label_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Label2 Filter
        ttk.Label(filter_controls_frame, text="Label2:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        self.label2_var = tk.StringVar()
        self.label2_combo = ttk.Combobox(filter_controls_frame, textvariable=self.label2_var, width=20)
        self.label2_combo.grid(row=0, column=5, padx=5, pady=5)
        
        # Add Filter Button
        ttk.Button(filter_controls_frame, text="Filter hinzufügen", command=self.add_filter).grid(row=0, column=6, padx=5, pady=5)
        ttk.Button(filter_controls_frame, text="Filter zurücksetzen", command=self.reset_filters).grid(row=0, column=7, padx=5, pady=5)
        
        # Start Button und CSV zu Excel Button in einer Zeile
        ttk.Button(filter_controls_frame, text="Start", command=self.start_browsing).grid(row=0, column=8, padx=5, pady=5)
        ttk.Button(filter_controls_frame, text="Refinement Start", command=self.start_refinement).grid(row=0, column=9, padx=5, pady=5)
        
        # Checkbox für das Anzeigen von abgelehnten Bildern im Refinement-Modus
        self.show_rejected_var = tk.BooleanVar(value=self.show_rejected)
        ttk.Checkbutton(filter_controls_frame, text="Abgelehnte im Refinement anzeigen", variable=self.show_rejected_var, 
                      command=self.toggle_show_rejected).grid(row=0, column=12, padx=5, pady=5)
        
        ttk.Button(filter_controls_frame, text="CSV zu Excel", command=self.convert_csv_to_excel).grid(row=0, column=10, padx=5, pady=5)
        ttk.Button(filter_controls_frame, text="Refine CSV zu Excel", command=self.convert_refine_csv_to_excel).grid(row=0, column=11, padx=5, pady=5)
        
        # Selected Labels Display (horizontal layout)
        selected_filters_frame = ttk.Frame(filter_frame)
        selected_filters_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(selected_filters_frame, text="Ausgewählte Filter:").pack(side=tk.LEFT, padx=5, pady=5)
        
        # Frame für horizontales Scrollen der Filter
        filter_scroll_frame = ttk.Frame(selected_filters_frame)
        filter_scroll_frame.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        
        # Scrollbare Fläche für Filter
        self.selected_labels_frame = ttk.Frame(filter_scroll_frame)
        self.selected_labels_frame.pack(fill=tk.X, expand=True)
        
        # Image Display Frame - mit dynamischer Größe
        middle_panel = ttk.Frame(main_frame)
        middle_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_frame = ttk.LabelFrame(middle_panel, text="CT-Slice")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Container für mehrere Bilder nebeneinander
        self.images_container = ttk.Frame(self.image_frame)
        self.images_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame für jedes der drei möglichen Bilder
        self.display_frame = ttk.Frame(self.images_container)
        self.display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.display_frame_km = ttk.Frame(self.images_container)
        self.display_frame_km.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.display_frame_mr = ttk.Frame(self.images_container)
        self.display_frame_mr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        # Image Labels für alle drei Bilder
        self.image_label = ttk.Label(self.display_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        self.image_label_km = ttk.Label(self.display_frame_km)
        self.image_label_km.pack(fill=tk.BOTH, expand=True)
        
        self.image_label_mr = ttk.Label(self.display_frame_mr)
        self.image_label_mr.pack(fill=tk.BOTH, expand=True)
        
        # Info Frame
        info_frame = ttk.Frame(middle_panel)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Image Info
        self.info_label = ttk.Label(info_frame, text="Keine Bilder geladen")
        self.info_label.pack(side=tk.LEFT, padx=5)
        
        # Counter Label
        self.counter_label = ttk.Label(info_frame, text="0/0 Bilder")
        self.counter_label.pack(side=tk.RIGHT, padx=5)
        
        # Comments Frame
        comments_frame = ttk.LabelFrame(bottom_panel, text="Kommentar")
        comments_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Comment Entry
        self.comment_var = tk.StringVar()
        ttk.Entry(comments_frame, textvariable=self.comment_var, width=100).pack(fill=tk.X, padx=5, pady=5)
        
        # Navigation Button Frame - immer am unteren Rand des Fensters
        button_frame = ttk.Frame(bottom_panel)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Navigation Buttons
        ttk.Button(button_frame, text="Zurück", command=self.prev_image).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Accept/Reject Buttons
        ttk.Button(button_frame, text="✔️ Akzeptieren", command=self.accept_image,
                  style="Accept.TButton").pack(side=tk.RIGHT, padx=10, pady=5)
        ttk.Button(button_frame, text="❌ Ablehnen", command=self.reject_image,
                  style="Reject.TButton").pack(side=tk.RIGHT, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Weiter", command=self.next_image).pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Label Buttons Frame - unterhalb der Navigationsbuttons
        label_buttons_frame = ttk.LabelFrame(bottom_panel, text="Refinement-Labels")
        label_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Mehrere Reihen von Label-Buttons
        labels_per_row = 10
        for i, label in enumerate(self.predefined_labels):
            row_num = i // labels_per_row
            col_num = i % labels_per_row
            ttk.Button(
                label_buttons_frame, 
                text=label, 
                command=lambda l=label: self.accept_with_label(l)
            ).grid(row=row_num, column=col_num, padx=5, pady=3, sticky="w")
        
        # Set up styles
        self.setup_styles()
    
    def setup_styles(self):
        style = ttk.Style()
        style.configure("Accept.TButton", background="#4CAF50", foreground="green")
        style.configure("Reject.TButton", background="#F44336", foreground="red")
    
    def load_excel_and_update_filters(self):
        """Automatically load Excel file and update filter options on startup"""
        if os.path.exists(self.settings.get("input_excel", "")):
            try:
                self.dataframe = pd.read_excel(self.settings.get("input_excel", ""))
                
                # Update Label combobox
                if 'Label' in self.dataframe.columns:
                    unique_labels = self.dataframe['Label'].dropna().unique().tolist()
                    self.label_combo['values'] = [""] + unique_labels
                
                # Update Label2 combobox
                if 'Label2' in self.dataframe.columns:
                    unique_labels2 = self.dataframe['Label2'].dropna().unique().tolist()
                    self.label2_combo['values'] = [""] + unique_labels2
                
                print(f"Excel-Datei geladen: {len(self.dataframe)} Einträge gefunden.")
                
                # Gespeicherte Filter-Auswahl anzeigen
                self.update_selected_filters_display()
                
            except Exception as e:
                print(f"Fehler beim Laden der Excel-Datei: {str(e)}")
    
    def add_filter(self):
        """Add the selected label/label2 to the filter list"""
        label = self.label_var.get()
        label2 = self.label2_var.get()
        
        # Add to selected sets
        if label:
            self.selected_labels.add(label)
        if label2:
            self.selected_labels2.add(label2)
        
        # Update display
        self.update_selected_filters_display()
        
        # Clear selections
        self.label_var.set("")
        self.label2_var.set("")
    
    def reset_filters(self):
        """Reset all selected filters"""
        self.selected_labels.clear()
        self.selected_labels2.clear()
        self.update_selected_filters_display()
    
    def update_selected_filters_display(self):
        """Update the display of selected filters"""
        # Clear current display
        for widget in self.selected_labels_frame.winfo_children():
            widget.destroy()
        
        # Horizontales Layout für Filter
        filter_container = ttk.Frame(self.selected_labels_frame)
        filter_container.pack(fill=tk.X, expand=True)
        
        column = 0
        # Add Label filters
        for label in self.selected_labels:
            filter_frame = ttk.Frame(filter_container)
            filter_frame.grid(row=0, column=column, sticky=tk.W, padx=2, pady=2)
            
            ttk.Label(filter_frame, text=f"Label: {label}").pack(side=tk.LEFT, padx=2)
            ttk.Button(filter_frame, text="X", width=2, 
                      command=lambda l=label: self.remove_filter("label", l)).pack(side=tk.LEFT, padx=2)
            column += 1
        
        # Add Label2 filters
        for label2 in self.selected_labels2:
            filter_frame = ttk.Frame(filter_container)
            filter_frame.grid(row=0, column=column, sticky=tk.W, padx=2, pady=2)
            
            ttk.Label(filter_frame, text=f"Label2: {label2}").pack(side=tk.LEFT, padx=2)
            ttk.Button(filter_frame, text="X", width=2, 
                      command=lambda l=label2: self.remove_filter("label2", l)).pack(side=tk.LEFT, padx=2)
            column += 1
    
    def remove_filter(self, filter_type, value):
        """Remove a filter from the selected set"""
        if filter_type == "label":
            self.selected_labels.discard(value)
        elif filter_type == "label2":
            self.selected_labels2.discard(value)
        
        self.update_selected_filters_display()
    
    def browse_excel(self):
        filename = filedialog.askopenfilename(
            title="Excel-Datei auswählen",
            filetypes=[("Excel files", "*.xlsx;*.xls")],
            initialdir=os.getcwd()
        )
        if filename:
            self.input_excel_var.set(filename)
            self.settings["input_excel"] = filename
            self.save_settings()
            self.update_filter_options()
    
    def browse_csv(self):
        filename = filedialog.asksaveasfilename(
            title="CSV-Datei auswählen",
            filetypes=[("CSV files", "*.csv")],
            initialdir=os.getcwd(),
            defaultextension=".csv"
        )
        if filename:
            self.output_csv_var.set(filename)
            self.settings["output_csv"] = filename
            self.save_settings()
    
    def browse_folder(self):
        folder = filedialog.askdirectory(
            title="Bilder-Ordner auswählen",
            initialdir=os.getcwd()
        )
        if folder:
            self.images_folder_var.set(folder)
            self.settings["images_folder"] = folder
            self.save_settings()
    
    def update_filter_options(self):
        try:
            self.dataframe = pd.read_excel(self.input_excel_var.get())
            
            # Update Label combobox
            if 'Label' in self.dataframe.columns:
                unique_labels = self.dataframe['Label'].dropna().unique().tolist()
                self.label_combo['values'] = [""] + unique_labels
            
            # Update Label2 combobox
            if 'Label2' in self.dataframe.columns:
                unique_labels2 = self.dataframe['Label2'].dropna().unique().tolist()
                self.label2_combo['values'] = [""] + unique_labels2
            
            messagebox.showinfo("Info", f"Excel-Datei geladen: {len(self.dataframe)} Einträge gefunden.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden der Excel-Datei: {str(e)}")
    
    def load_accepted_images(self):
        output_csv = self.output_csv_var.get()
        if os.path.exists(output_csv):
            try:
                with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';')
                    next(reader)  # Skip header row
                    for row in reader:
                        if len(row) > 0:  # Check if row has data
                            # Extract Patient-ID and PlanID as unique identifier
                            self.accepted_images.add(row[0] + "_" + row[5])
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden der akzeptierten Bilder: {str(e)}")
    
    def start_browsing(self):
        # Save the settings
        self.settings["input_excel"] = self.input_excel_var.get()
        self.settings["output_csv"] = self.output_csv_var.get()
        self.settings["images_folder"] = self.images_folder_var.get()
        self.save_settings()
        
        # Check if input files exist
        if not os.path.exists(self.settings["input_excel"]):
            messagebox.showerror("Fehler", "Excel-Datei nicht gefunden.")
            return
        
        if not os.path.exists(self.settings["images_folder"]):
            messagebox.showerror("Fehler", "Bilder-Ordner nicht gefunden.")
            return
        
        # Load Excel file if not already loaded
        if self.dataframe is None:
            try:
                self.dataframe = pd.read_excel(self.settings["input_excel"])
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden der Excel-Datei: {str(e)}")
                return
        
        # Apply filters
        self.apply_filters()
        
        # Load accepted images if not already loaded
        if not self.accepted_images:
            self.load_accepted_images()
        
        # Reset index and display first image
        self.current_index = -1
        self.next_image()
    
    def apply_filters(self):
        # Start with the full dataframe
        self.filtered_data = self.dataframe.copy()
        
        filter_mode = self.filter_mode_var.get()
        
        # Apply Label filters based on mode
        if self.selected_labels:
            if filter_mode == "include":
                # Include mode - keep only matching rows
                self.filtered_data = self.filtered_data[self.filtered_data['Label'].isin(self.selected_labels)]
            else:
                # Exclude mode - filter out matching rows
                self.filtered_data = self.filtered_data[~self.filtered_data['Label'].isin(self.selected_labels)]
        
        # Apply Label2 filters based on mode
        if self.selected_labels2:
            if filter_mode == "include":
                # Include mode - keep only matching rows
                self.filtered_data = self.filtered_data[self.filtered_data['Label2'].isin(self.selected_labels2)]
            else:
                # Exclude mode - filter out matching rows
                self.filtered_data = self.filtered_data[~self.filtered_data['Label2'].isin(self.selected_labels2)]
        
        # Filter out already accepted images
        if self.accepted_images:
            self.filtered_data = self.filtered_data[~self.filtered_data.apply(
                lambda row: f"{row['Patient-ID']}_{row['PlanID']}" in self.accepted_images, axis=1)]
        
        # Reset index
        self.filtered_data = self.filtered_data.reset_index(drop=True)
        
        # Update counters
        self.total_images = len(self.filtered_data)
        self.remaining_images = self.total_images
        
        # Update counter label
        self.update_counter_label()
        
        # Speichern der aktuellen Filter für den nächsten Start
        self.save_settings()
        
        messagebox.showinfo("Filter angewendet", f"{self.total_images} Bilder nach Filtern verfügbar.")
    
    def update_counter_label(self):
        if self.current_index >= 0 and self.total_images > 0:
            self.counter_label.config(text=f"{self.current_index + 1}/{self.total_images} Bilder")
        else:
            self.counter_label.config(text="0/0 Bilder")
    
    def update_info_label(self):
        if self.current_index >= 0 and self.total_images > 0 and self.current_index < self.total_images:
            row = self.filtered_data.iloc[self.current_index]
            info_text = f"Patient: {row['Patient-ID']} - {row['Nachname']}, {row['Vorname']} | Plan: {row['PlanID']} | GTV: {row['GTV']} | Volume: {row['GTV-Volume']:.2f} | Label: {row['Label']} - {row['Label2']}"
            self.info_label.config(text=info_text)
            
            # Im Refinement-Modus lade den vorhandenen Kommentar in das Eingabefeld
            if self.refinement_mode and 'Comment' in row and pd.notna(row['Comment']):
                self.comment_var.set(row['Comment'])
                if 'casekind' in row and pd.notna(row['casekind']):
                    self.extra_label = row['casekind']
        else:
            self.info_label.config(text="Keine Bilder verfügbar")
    
    def load_image(self):
        if self.current_index >= 0 and self.current_index < self.total_images:
            row = self.filtered_data.iloc[self.current_index]
            image_path = row['PNG_Path']
            
            # Erst alle Bilder und Referenzen vollständig zurücksetzen
            self.image_object = None
            self.image_object_km = None
            self.image_object_mr = None
            
            # Auch die PhotoImage-Objekte zurücksetzen
            self.current_image = None
            self.current_image_km = None
            self.current_image_mr = None
            
            # Alle Labels zurücksetzen
            self.image_label.config(image=None)
            self.image_label_km.config(image=None)
            self.image_label_mr.config(image=None)
            
            # Erstellen Sie alternative Pfade für die KM- und MR-Bilder
            base_path, ext = os.path.splitext(image_path)
            if base_path.endswith('_KM') or base_path.endswith('_MR'):
                # Entferne vorhandene Suffixe, um den Basisnamen zu erhalten
                base_path = base_path.replace('_KM', '').replace('_MR', '')
            
            km_path = f"{base_path}_KM{ext}"
            mr_path = f"{base_path}_MR{ext}"
            
            # Lade das Hauptbild
            self.image_object = self.load_image_from_path(image_path)
            if not self.image_object:
                return False
                
            # Versuche, die KM- und MR-Bilder zu laden
            self.image_object_km = self.load_image_from_path(km_path)
            self.image_object_mr = self.load_image_from_path(mr_path)
            
            # Passe die Bilder an und zeige sie an
            self.resize_images()
            
            # Update info label
            self.update_info_label()
            return True
        else:
            # Vollständiges Zurücksetzen, wenn keine Bilder geladen werden
            self.image_object = None
            self.image_object_km = None
            self.image_object_mr = None
            self.current_image = None
            self.current_image_km = None
            self.current_image_mr = None
            self.image_label.config(image=None)
            self.image_label_km.config(image=None)
            self.image_label_mr.config(image=None)
            self.info_label.config(text="Keine Bilder verfügbar")
            return False
    
    def load_image_from_path(self, image_path):
        """Hilfsfunktion zum Laden eines Bildes von einem Pfad"""
        # Handle network path or local path
        if image_path.startswith('\\\\'):
            # Network path - try to use directly or convert to local path
            try:
                image_path_fixed = image_path.replace('\\', '/')
                return Image.open(image_path_fixed)
            except:
                # Try using the local path
                file_name = os.path.basename(image_path)
                local_path = os.path.join(self.settings["images_folder"], file_name)
                if os.path.exists(local_path):
                    return Image.open(local_path)
                else:
                    # Kein Fehler für KM und MR Bilder anzeigen, nur für Hauptbild
                    if not (image_path.endswith('_KM.png') or image_path.endswith('_MR.png')):
                        messagebox.showerror("Fehler", f"Bild nicht gefunden: {image_path}\nVersucht auch: {local_path}")
                    return None
        else:
            # Local path
            try:
                return Image.open(image_path)
            except:
                # Try using just the filename in the images folder
                file_name = os.path.basename(image_path)
                local_path = os.path.join(self.settings["images_folder"], file_name)
                if os.path.exists(local_path):
                    return Image.open(local_path)
                else:
                    # Kein Fehler für KM und MR Bilder anzeigen, nur für Hauptbild
                    if not (image_path.endswith('_KM.png') or image_path.endswith('_MR.png')):
                        messagebox.showerror("Fehler", f"Bild nicht gefunden: {image_path}\nVersucht auch: {local_path}")
                    return None
            
    def resize_images(self):
        """Skaliert alle vorhandenen Bilder und zeigt sie an"""
        # Bestimme verfügbare Größe in jedem Display-Frame
        self.images_container.update_idletasks()  # Aktualisiere Layout
        
        # Ermittle die Anzahl der anzuzeigenden Bilder
        num_images = sum(1 for img in [self.image_object, self.image_object_km, self.image_object_mr] if img is not None)
        
        # Passe die Breite jedes Bildes entsprechend an
        container_width = self.images_container.winfo_width()
        container_height = self.images_container.winfo_height()
        
        # Mindestbreite und -höhe für die Bildanzeige
        min_width = 200
        min_height = 150
        
        # Explizit alle Bilder zurücksetzen, die nicht vorhanden sind
        if self.image_object is None:
            self.current_image = None
            self.image_label.config(image=None)
            
        if self.image_object_km is None:
            self.current_image_km = None
            self.image_label_km.config(image=None)
            
        if self.image_object_mr is None:
            self.current_image_mr = None
            self.image_label_mr.config(image=None)
        
        if num_images > 0:
            # Berechne die verfügbare Breite pro Bild
            max_width_per_image = max(min_width, (container_width - 20) // num_images)  # Platz für Ränder
            max_height = max(min_height, container_height - 20)  # Platz für Ränder
            
            # Skaliere und zeige das Hauptbild an
            if self.image_object:
                self.current_image = self.scale_and_display_image(
                    self.image_object, max_width_per_image, max_height, self.image_label)
            
            # Skaliere und zeige das KM-Bild an, wenn vorhanden
            if self.image_object_km:
                self.current_image_km = self.scale_and_display_image(
                    self.image_object_km, max_width_per_image, max_height, self.image_label_km)
            
            # Skaliere und zeige das MR-Bild an, wenn vorhanden
            if self.image_object_mr:
                self.current_image_mr = self.scale_and_display_image(
                    self.image_object_mr, max_width_per_image, max_height, self.image_label_mr)
    
    def scale_and_display_image(self, image_object, max_width, max_height, label):
        """Skaliert ein Bild und zeigt es im angegebenen Label an"""
        # Original-Bildgröße
        width, height = image_object.size
        
        # Berechne neue Abmessungen unter Beibehaltung des Seitenverhältnisses
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Bild mit hoher Qualität skalieren
        resized_image = image_object.resize((new_width, new_height), Image.LANCZOS)
        
        # Erstelle ein leeres Bild mit der maximal verfügbaren Größe
        blank_image = Image.new("RGB", (max_width, max_height), (240, 240, 240))
        
        # Berechne Position zum Einfügen des skalierten Bildes zentriert im leeren Bild
        position = ((max_width - new_width) // 2, (max_height - new_height) // 2)
        
        # Füge das skalierte Bild in die leere Leinwand ein
        blank_image.paste(resized_image, position)
        
        # Konvertiere zu PhotoImage und zeige an
        photo_image = ImageTk.PhotoImage(blank_image)
        label.config(image=photo_image)
        
        return photo_image
        
    def on_window_resize(self, event):
        """Wird aufgerufen, wenn das Fenster seine Größe ändert"""
        # Nur auf Größenänderungen des Hauptfensters reagieren
        if event.widget == self.root:
            # Warte einen Moment, damit alle Widgets ihre Größe aktualisieren können
            self.root.after(100, self.resize_images)
    
    def next_image(self):
        if self.total_images <= 0:
            messagebox.showinfo("Info", "Keine Bilder verfügbar. Starten Sie die Anwendung mit Filtern.")
            return
        
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            if not self.load_image():
                # If image loading failed, try next one
                self.next_image()
            else:
                self.update_counter_label()
        else:
            messagebox.showinfo("Ende", "Sie haben alle verfügbaren Bilder angesehen.")
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            if not self.load_image():
                # If image loading failed, try previous one
                self.prev_image()
            else:
                self.update_counter_label()
        else:
            messagebox.showinfo("Anfang", "Sie sind beim ersten Bild.")
    
    def accept_image(self):
        if self.current_index >= 0 and self.current_index < self.total_images:
            row = self.filtered_data.iloc[self.current_index]
            
            # Handle refinement mode differently
            if self.refinement_mode:
                self.save_refinement(row)
                self.next_image()
                return
                
            # Create unique identifier for the image
            identifier = f"{row['Patient-ID']}_{row['PlanID']}"
            
            # Check if already accepted
            if identifier in self.accepted_images:
                messagebox.showinfo("Info", "Dieses Bild wurde bereits akzeptiert.")
                self.next_image()
                return
            
            # Mark as accepted
            self.accepted_images.add(identifier)
            
            # Save to CSV
            self.save_to_csv(row)
            
            # Go to next image
            self.next_image()
        else:
            messagebox.showinfo("Info", "Kein Bild zum Akzeptieren verfügbar.")
    
    def reject_image(self):
        if self.current_index >= 0 and self.current_index < self.total_images:
            row = self.filtered_data.iloc[self.current_index]
            
            # Handle refinement mode differently
            if self.refinement_mode:
                # In refinement mode, rejection means removing from refinement output
                self.next_image()
                return
                
            # Create unique identifier for the image
            identifier = f"{row['Patient-ID']}_{row['PlanID']}"
            
            # Check if already accepted
            if identifier in self.accepted_images:
                messagebox.showinfo("Info", "Dieses Bild wurde bereits akzeptiert.")
                self.next_image()
                return
            
            # Mark as accepted (to avoid showing it again)
            self.accepted_images.add(identifier)
            
            # Save to CSV with "abgelehnt" comment
            original_comment = self.comment_var.get()
            self.comment_var.set("abgelehnt" if not original_comment else f"abgelehnt - {original_comment}")
            self.save_to_csv(row)
            self.comment_var.set("")  # Reset comment field
            
            # Go to next image
            self.next_image()
        else:
            messagebox.showinfo("Info", "Kein Bild zum Ablehnen verfügbar.")
    
    def save_to_csv(self, row):
        output_csv = self.output_csv_var.get()
        comment = self.comment_var.get()
        
        # Create file with header if it doesn't exist
        file_exists = os.path.isfile(output_csv)
        
        try:
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                
                # Write header if file is new
                if not file_exists:
                    header = list(self.dataframe.columns) + ['Comment', 'AcceptedDate']
                    writer.writerow(header)
                
                # Write row data
                row_data = list(row)
                row_data.append(comment)
                row_data.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                writer.writerow(row_data)
            
            # Clear comment field
            self.comment_var.set("")
            
            # Update remaining count
            self.remaining_images -= 1
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern in CSV: {str(e)}")
    
    def convert_csv_to_excel(self):
        csv_file = self.output_csv_var.get()
        
        if not os.path.exists(csv_file):
            messagebox.showerror("Fehler", "CSV-Datei nicht gefunden.")
            return
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
            
            # Create Excel filename (same name but with xlsx extension)
            excel_file = os.path.splitext(csv_file)[0] + '.xlsx'
            
            # Save to Excel
            df.to_excel(excel_file, index=False)
            
            messagebox.showinfo("Erfolg", f"CSV in Excel konvertiert: {excel_file}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Konvertierung: {str(e)}")

    def convert_refine_csv_to_excel(self):
        if not hasattr(self, 'refinement_data') or self.refinement_data is None:
            messagebox.showinfo("Info", "Keine Refinement-Daten vorhanden!")
            return
        
        # Convert refinement data to Excel
        output_file = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Speichern als Excel"
        )
        if not output_file:
            return
        
        try:
            self.refinement_data.to_excel(output_file, index=False)
            messagebox.showinfo("Erfolg", f"Refinement-Daten erfolgreich gespeichert als {output_file}")
            
            # Display Excel data in new window
            self.display_excel_data(self.refinement_data)
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern der Excel-Datei: {str(e)}")

    def get_refinement_csv_path(self):
        """Erstellt den Pfad für die Refinement-CSV-Datei"""
        base_path = os.path.splitext(self.output_csv_var.get())[0]
        return f"{base_path}_refine.csv"

    def start_refinement(self):
        """Startet den Refinement-Modus zum Überarbeiten der akzeptierten Bilder"""
        # Setze den Refinement-Modus
        self.refinement_mode = True
        
        # Überprüfe, ob die Ausgabedatei existiert
        output_csv = self.output_csv_var.get()
        if not os.path.exists(output_csv):
            messagebox.showerror("Fehler", "Keine CSV-Datei zum Verfeinern gefunden.")
            self.refinement_mode = False
            return
        
        try:
            # Lade die CSV-Datei als DataFrame
            self.refinement_data = pd.read_csv(output_csv, delimiter=';', encoding='utf-8')
            
            # Stelle sicher, dass die Dateispalten existieren
            required_cols = ['Patient-ID', 'PlanID', 'PNG_Path', 'Comment']
            for col in required_cols:
                if col not in self.refinement_data.columns:
                    messagebox.showerror("Fehler", f"Spalte '{col}' fehlt in der CSV-Datei.")
                    self.refinement_mode = False
                    return
            
            # Filtere abgelehnte Bilder heraus, wenn die Option nicht aktiviert ist
            if not self.show_rejected_var.get():
                # Filtere Bilder mit "abgelehnt" im Kommentar heraus
                if 'Comment' in self.refinement_data.columns:
                    # Erstelle eine Maske für Zeilen, die nicht "abgelehnt" im Kommentar enthalten
                    mask = ~self.refinement_data['Comment'].str.contains('abgelehnt', case=False, na=False)
                    self.refinement_data = self.refinement_data[mask]
            
            # Setze den Index auf -1, um mit dem ersten Bild zu beginnen
            self.current_index = -1
            
            # Definiere die gefilterten Daten als die Refinement-Daten
            self.filtered_data = self.refinement_data.copy()
            
            # Zurücksetzen der akzeptierten Bilder (für den Refinement-Modus nicht relevant)
            self.accepted_images = set()
            
            # Aktualisiere Zähler
            self.total_images = len(self.filtered_data)
            self.remaining_images = self.total_images
            
            # Zeige das erste Bild
            self.next_image()
            
            # Informiere den Benutzer, wie viele Bilder im Refinement-Modus verfügbar sind
            reject_status = "einschließlich abgelehnter Bilder" if self.show_rejected_var.get() else "ohne abgelehnte Bilder"
            messagebox.showinfo("Refinement gestartet", f"{self.total_images} Bilder zum Verfeinern geladen ({reject_status}).")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Starten des Refinements: {str(e)}")
            self.refinement_mode = False

    def accept_with_label(self, label):
        if self.current_index >= 0 and self.current_index < self.total_images:
            row = self.filtered_data.iloc[self.current_index]
            
            # Handle refinement mode differently
            if self.refinement_mode:
                self.save_refinement(row, label)
                self.next_image()
                return
            
            # Create unique identifier for the image
            identifier = f"{row['Patient-ID']}_{row['PlanID']}"
            
            # Check if already accepted
            if identifier in self.accepted_images:
                messagebox.showinfo("Info", "Dieses Bild wurde bereits akzeptiert.")
                self.next_image()
                return
            
            # Mark as accepted
            self.accepted_images.add(identifier)
            
            # Save to CSV with label
            self.save_to_csv_with_label(row, label)
            
            # Go to next image
            self.next_image()
        else:
            messagebox.showinfo("Info", "Kein Bild zum Akzeptieren verfügbar.")

    def save_to_csv_with_label(self, row, label):
        output_csv = self.output_csv_var.get()
        comment = self.comment_var.get()
        
        # Create file with header if it doesn't exist
        file_exists = os.path.isfile(output_csv)
        
        try:
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                
                # Write header if file is new
                if not file_exists:
                    header = list(self.dataframe.columns) + ['Comment', 'AcceptedDate', 'casekind']
                    writer.writerow(header)
                
                # Write row data
                row_data = list(row)
                row_data.append(comment)
                row_data.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                row_data.append(label)
                writer.writerow(row_data)
            
            # Clear comment field
            self.comment_var.set("")
            
            # Update remaining count
            self.remaining_images -= 1
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern in CSV: {str(e)}")

    def save_refinement(self, row, label=None):
        refinement_csv = self.get_refinement_csv_path()
        comment = self.comment_var.get()
        
        # Create file with header if it doesn't exist
        file_exists = os.path.isfile(refinement_csv)
        
        try:
            with open(refinement_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                
                # Write header if file is new
                if not file_exists:
                    # Add casekind column if needed
                    header = list(self.refinement_data.columns)
                    if 'casekind' not in header:
                        header.append('casekind')
                    if 'RefinementDate' not in header:
                        header.append('RefinementDate')
                    writer.writerow(header)
                
                # Write row data
                row_data = list(row)
                
                # Update comment if needed
                if 'Comment' in self.refinement_data.columns:
                    comment_index = list(self.refinement_data.columns).index('Comment')
                    if comment_index < len(row_data):
                        row_data[comment_index] = comment
                
                # Add or update casekind if provided
                if label is not None:
                    if 'casekind' in self.refinement_data.columns:
                        casekind_index = list(self.refinement_data.columns).index('casekind')
                        if casekind_index < len(row_data):
                            row_data[casekind_index] = label
                        else:
                            row_data.append(label)
                    else:
                        row_data.append(label)
                elif 'casekind' not in self.refinement_data.columns:
                    row_data.append("")  # Empty label if none provided
                
                # Add refinement date
                row_data.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                writer.writerow(row_data)
            
            # Clear comment field
            self.comment_var.set("")
            
            # Update remaining count
            self.remaining_images -= 1
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern in Refinement-CSV: {str(e)}")

    def toggle_show_rejected(self):
        self.show_rejected = self.show_rejected_var.get()
        if self.refinement_mode:
            self.start_refinement()

    def display_excel_data(self, data):
        # Create a new window to display the Excel data
        excel_window = tk.Toplevel(self.root)
        excel_window.title("Excel Daten")

        # Create a text box to display the data
        text_box = tk.Text(excel_window)
        text_box.pack(fill=tk.BOTH, expand=True)

        # Insert the data into the text box
        text_box.insert(tk.END, str(data))

if __name__ == "__main__":
    root = tk.Tk()
    app = CTBrowserApp(root)
    root.mainloop()
