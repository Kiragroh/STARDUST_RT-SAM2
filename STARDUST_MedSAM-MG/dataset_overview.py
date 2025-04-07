#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from dicom_converter import visualize_slices
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PSEUDO_FILE_CT = os.path.join(BASE_DIR, "data", "DICOM-All", "pseudo_mapping_CT.csv")
PSEUDO_FILE_MR = os.path.join(BASE_DIR, "data", "DICOM-All", "pseudo_mapping_MR.csv")
REFINE_ALL_FILE = os.path.join(BASE_DIR, "data", "DICOM-All", "Refine_ALL.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_overview")
EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "patient_overview.xlsx")

DATASET_DIRS = [
    ("npz_custom_Kopf", "CT"),
    ("npz_custom_Lunge", "CT"),
    ("npz_custom_Rest", "CT"),
    ("npz_custom_Mix", "CT"),
    ("npz_custom_Mix2", "CT"),
    ("npz_files_MR", "MR"),
    ("npz_files_MRglio", "MR"),
    ("npz_files_MixMix", "MR+CT")
]

MR_MAPPING = {
    'npz_files_MR': 'MR-Head',
    'npz_files_MRglio': 'MR-Glio',
    'npz_files_MixMix': 'MR+CT'
}
def plot_dataset_structure_with_split(
    dataset_sizes,
    train_test_counts,
    output_path="./dataset_overview/plots/dataset_structure_schema.png" # Neuer Dateiname V4
    ):
    """
    Zeichnet ein strukturiertes Diagramm der Datensatz-Zusammensetzung
    mit matplotlib und networkx, inklusive Train/Test-Split-Visualisierung
    als Kreissektoren. Stellt sicher, dass Kreise rund sind, platziert
    Text im unteren Bereich und minimiert Leerraum durch präzisere Limits.

    Args:
        dataset_sizes (dict): Mapping von Datensatznamen zu Gesamtfallzahlen (n).
        train_test_counts (dict): Mapping von Datensatznamen zu Tupeln (train_count, test_count).
        output_path (str): Pfad zum Speichern des generierten Plots.
    """
    # Konstanten, Layout, Graph-Erstellung etc. (wie in V3)
    NODE_SIZE_MULTIPLIER = 4000
    edges = [
        ("Head", "Mix"), ("Lung", "Mix"), ("Others", "Mix"),
        ("Lung", "Mix (no Head)"), ("Others", "Mix (no Head)"),
        ("MR-Glio", "MR-Head"),
        ("Mix (no Head)", "MR+CT"), ("MR-Head", "MR+CT"),
    ]
    G = nx.DiGraph()
    # ... (Knoten hinzufügen und Konsistenz prüfen - Code von V3 übernehmen) ...
    all_nodes_in_edges = set(item for edge in edges for item in edge)
    all_graph_nodes = all_nodes_in_edges.union(dataset_sizes.keys())
    missing_nodes_warnung = []
    missing_split_warnung = []
    for node in all_graph_nodes:
        total_size = dataset_sizes.get(node)
        split_data = train_test_counts.get(node)
        if total_size is None:
            if node in all_nodes_in_edges:
                missing_nodes_warnung.append(node)
                total_size = 0
            else: continue
        if split_data is None:
             missing_split_warnung.append(node)
             train_count, test_count = total_size, 0
        elif len(split_data) == 2:
            train_count, test_count = split_data
            if abs((train_count + test_count) - total_size) > 1e-6:
                 warnings.warn(f"Warnung: Split-Summe != Gesamtgröße für '{node}'. Verwende Split-Summe.")
                 total_size = train_count + test_count
                 dataset_sizes[node] = total_size
        else: raise ValueError(f"Ungültiges Format train_test_counts bei '{node}': {split_data}")
        G.add_node(node, size=total_size, train=train_count, test=test_count)
    if missing_nodes_warnung: warnings.warn(f"Warnung: Knoten {missing_nodes_warnung} ohne Größe. Größe=0.")
    if missing_split_warnung: warnings.warn(f"Warnung: Knoten {missing_split_warnung} ohne Split. Annahme: Test=0.")
    G.add_edges_from(edges)

    pos = { # Layout von V3 beibehalten
        "Head": (0, 3.1), 
        "Lung": (1.2, 3.1), 
        "Others": (2.4, 3.1),
        "Mix": (0.4, 1.8), 
        "Mix (no Head)": (1.9, 1.8),
        "MR-Glio": (3.4, 3.1), 
        "MR-Head": (3.1, 1.8), 
        "MR+CT": (1.9, 0.3),
    }
    nodes_to_draw = [n for n in G.nodes() if n in pos]
    if len(nodes_to_draw) != len(G.nodes()):
         missing_pos = [n for n in G.nodes() if n not in pos]
         warnings.warn(f"Warnung: Für Knoten {missing_pos} keine Position. Nicht gezeichnet.")

    # Radienberechnung (wie V3)
    node_radii = {n: np.sqrt(max(G.nodes[n]["size"], 1)) for n in nodes_to_draw}
    max_node_val = max(G.nodes[n]["size"] for n in nodes_to_draw) if nodes_to_draw else 1
    RADIUS_SCALE_FACTOR = 0.25
    scaled_radii = {
        n: np.sqrt(max(G.nodes[n]["size"], 1) / max_node_val) * RADIUS_SCALE_FACTOR * np.sqrt(len(nodes_to_draw))
        for n in nodes_to_draw
    }
    min_radius_display = 0.05
    for n in scaled_radii: scaled_radii[n] = max(scaled_radii[n], min_radius_display)

    # Farben (wie V3)
    color_map = {
         'primary_ct': '#a6cee3', 'primary_mr': '#fdbf6f',
         'mixed_intermediate': '#cab2d6', 'final_mix': '#b2df8a',
         'test_split': '#aaaaaa', 'default': '#cccccc'
    }
    node_base_colors = {}
    for node in nodes_to_draw:
        if node in ["Head", "Lung", "Others"]: node_base_colors[node] = color_map['primary_ct']
        elif node == "MR-Glio": node_base_colors[node] = color_map['primary_mr']
        elif node in ["Mix", "Mix (no Head)", "MR-Head"]: node_base_colors[node] = color_map['mixed_intermediate']
        elif node == "MR+CT": node_base_colors[node] = color_map['final_mix']
        else: node_base_colors[node] = color_map['default']

    # Plot-Setup - Wähle figsize, das ungefähr zum erwarteten Layout passt
    # Unser Layout ist breiter als hoch (ca. 3.6 Einheiten breit, ca. 2.6 hoch)
    layout_width = max(p[0] for p in pos.values()) - min(p[0] for p in pos.values())
    layout_height = max(p[1] for p in pos.values()) - min(p[1] for p in pos.values())
    aspect_ratio = layout_height / layout_width if layout_width > 0 else 1
    base_width = 8 # Kleinere Basisbreite versuchen
    fig_width = base_width
    fig_height = base_width * aspect_ratio * 1.2 # Kleiner Puffer für Radien/Höhe

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 1. Kanten zeichnen (wie V3)
    nx.draw_networkx_edges( G, pos, ax=ax, edge_color="#888888", arrows=True, arrowstyle='-|>', arrowsize=15, node_size=0)

    # 2. Knoten zeichnen (wie V3)
    for node in nodes_to_draw:
        x, y = pos[node]
        total_size = G.nodes[node]['size']
        test_count = G.nodes[node]['test']
        radius = scaled_radii[node]
        base_color = node_base_colors[node]
        test_color = color_map['test_split']
        if total_size <= 0: continue
        test_proportion = test_count / total_size
        test_angle = test_proportion * 360
        circle = patches.Circle((x, y), radius, facecolor=base_color, edgecolor='black', linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        if test_angle > 0.1:
            theta1, theta2 = 90 - test_angle, 90
            wedge = patches.Wedge(center=(x, y), r=radius, theta1=theta1, theta2=theta2, facecolor=test_color, edgecolor='black', linewidth=1.5, zorder=3)
            ax.add_patch(wedge)

    # 3. Labels zeichnen (Textposition wie V3)
    for node in nodes_to_draw:
        x, y = pos[node]
        radius = scaled_radii[node]
        label_text = f"{node}\nn={G.nodes[node]['size']}"
        text_y_offset = radius * 0.14 # Beibehalten von V3
        text_x = x
        text_y = y - text_y_offset
        ax.text(text_x, text_y, label_text, ha='center', va='top', fontsize=10, fontweight="normal", color="black",
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=4)

    # --- ACHSENLIMITS UND ASPECT RATIO ANPASSEN ---
    # 1. Finde die exakten Grenzen der gezeichneten Objekte (inkl. Radien)
    all_x_centers = [pos[n][0] for n in nodes_to_draw]
    all_y_centers = [pos[n][1] for n in nodes_to_draw]
    max_radius_val = max(scaled_radii.values()) if scaled_radii else 0

    # Berechne die minimalen/maximalen Koordinaten unter Berücksichtigung der Radien
    x_min_data = min(all_x_centers) - max_radius_val*0.6
    x_max_data = max(all_x_centers) + max_radius_val*0.6
    y_min_data = min(all_y_centers) - max_radius_val*1.1
    y_max_data = max(all_y_centers) + max_radius_val*0.6

    # 2. Setze die Achsenlimits auf diese exakten Grenzen
    ax.set_xlim(x_min_data, x_max_data)
    ax.set_ylim(y_min_data, y_max_data)

    # 3. Erzwinge das gleiche Seitenverhältnis
    #    'adjustable='box'' passt die Boxgröße an, um das Seitenverhältnis bei festen Limits zu erreichen.
    ax.set_aspect('equal', adjustable='box')

    # 4. Achsen ausblenden
    ax.axis('off')

    # Kein Titel

    # Speichern mit bbox_inches='tight'. Dies sollte jetzt effektiver sein,
    # da die Achsenlimits bereits eng an den Daten liegen.
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05) # Minimaler Rand explizit
    plt.close(fig)

    print(f"Plot gespeichert unter: {output_path}")
    return output_path

    """
    Zeichnet ein strukturiertes Diagramm der Datensatz-Zusammensetzung
    mit matplotlib und networkx.

    Args:
        dataset_sizes (dict): Mapping von Datensatznamen zu Fallzahlen (n).
        output_path (str): Pfad zum Speichern des generierten Plots.
    """
    # Konstante für die Skalierung der Knotengröße
    # Erhöht, damit der Text besser passt (vorher * 10)
    NODE_SIZE_MULTIPLIER = 45 # Deutlich größer für besseren Textplatz

    # Struktur der Beziehungen (Kanten des Graphen)
    edges = [
        ("Head", "Mix"),
        ("Lung", "Mix"),
        ("Others", "Mix"),
        ("Lung", "Mix (no Head)"),
        ("Others", "Mix (no Head)"),
        ("MR-Glio", "MR-Head"),
        ("Mix (no Head)", "MR+CT"),
        ("MR-Head", "MR+CT"),
    ]

    # Erstelle einen gerichteten Graphen
    G = nx.DiGraph()

    # Stelle sicher, dass alle Knoten aus den Kanten auch in dataset_sizes sind oder eine Größe von 0 haben
    all_nodes_in_edges = set(item for edge in edges for item in edge)
    all_graph_nodes = all_nodes_in_edges.union(dataset_sizes.keys())

    missing_nodes_warnung = []
    for node in all_graph_nodes:
        size = dataset_sizes.get(node)
        if size is None:
            # Wenn ein Knoten in den Kanten vorkommt, aber keine Größe hat
            if node in all_nodes_in_edges:
                missing_nodes_warnung.append(node)
                size = 0 # Weise Größe 0 zu, damit der Knoten gezeichnet wird
            else:
                # Wenn ein Knoten nur in dataset_sizes ist, aber nicht in Kanten (isoliert)
                # Füge ihn trotzdem hinzu, er wird aber ggf. nicht gezeichnet, wenn layout fehlt
                size = 0 # Oder dataset_sizes[node], falls vorhanden
                # Da wir ein festes Layout haben, ignorieren wir Knoten ohne Position
                continue # Diesen Knoten erstmal ignorieren, wenn er keine Kanten hat UND keine Position

        # Füge Knoten mit seiner Größe hinzu
        G.add_node(node, size=size)

    if missing_nodes_warnung:
        warnings.warn(f"Warnung: Für Knoten {missing_nodes_warnung} wurde keine Größe in 'dataset_sizes' gefunden. Größe wurde auf 0 gesetzt.")

    # Füge die Kanten hinzu
    G.add_edges_from(edges)

    # Layout mit festen Ebenen (Positionen der Knoten)
    # Leicht angepasst für bessere Lesbarkeit evtl.
    pos = {
        "Head": (0, 3),
        "Lung": (1.2, 3), # Etwas mehr Abstand
        "Others": (2.4, 3), # Etwas mehr Abstand
        "Mix": (0.6, 2),  # Angepasst an Head/Lung/Others
        "Mix (no Head)": (1.8, 2), # Angepasst an Lung/Others
        "MR-Glio": (3.6, 3), # Weiter rechts
        "MR-Head": (3.0, 2), # Angepasst an MR-Glio / Mixes
        "MR+CT": (1.8, 1), # Zentrierter unter den Mixes
    }

    # Überprüfe, ob alle Knoten im Graphen auch eine Position haben
    # Wenn nicht, werden sie von nx.draw ignoriert oder an (0,0) platziert
    nodes_to_draw = [n for n in G.nodes() if n in pos]
    if len(nodes_to_draw) != len(G.nodes()):
         missing_pos = [n for n in G.nodes() if n not in pos]
         warnings.warn(f"Warnung: Für Knoten {missing_pos} wurde keine Position im 'pos'-Layout definiert. Sie werden nicht gezeichnet.")
         # Filtere G, um nur Knoten mit Position zu zeichnen (optional, nx.draw macht das meistens)
         # G = G.subgraph(nodes_to_draw) # Besser nicht G ändern, nx.draw filtern lassen

    # Berechne Knotengrößen für das Plotten
    # Verwende max(size, 1) * multiplier, um sehr kleine/null Knoten sichtbar zu machen
    # Oder filtere Knoten mit Größe 0? Lassen wir sie klein.
    node_sizes = [max(G.nodes[n]["size"], 1) * NODE_SIZE_MULTIPLIER for n in nodes_to_draw]

    # Definiere ein neues Farbschema (Beispiel: Verwendung von Pastellfarben oder thematisch)
    # color_palette = plt.cm.Pastel2 # Beispiel Matplotlib Colormap
    # colors = [mcolors.to_hex(color_palette(i)) for i in range(len(nodes_to_draw))] # Einfach durchnummeriert

    # Thematisches Farbschema (verbessert)
    color_map = {
         'primary_ct': '#a6cee3', # Hellblau (für Head, Lung, Others)
         'primary_mr': '#fdbf6f', # Hellorange (für MR-Glio)
         'mixed_intermediate': '#cab2d6', # Lavendel (für Mix, Mix (no Head), MR-Head)
         'final_mix': '#b2df8a', # Hellgrün (für MR+CT)
         'default': '#cccccc' # Grau für nicht kategorisierte (sollte nicht vorkommen)
    }
    node_colors = []
    for node in nodes_to_draw:
        if node in ["Head", "Lung", "Others"]:
            node_colors.append(color_map['primary_ct'])
        elif node == "MR-Glio":
            node_colors.append(color_map['primary_mr'])
        elif node in ["Mix", "Mix (no Head)", "MR-Head"]:
            node_colors.append(color_map['mixed_intermediate'])
        elif node == "MR+CT":
             node_colors.append(color_map['final_mix'])
        else:
             node_colors.append(color_map['default'])


    # Erstelle die Labels mit zwei Zeilen: Name und n=Anzahl
    labels = {node: f"{node}\nn={G.nodes[node]['size']}" for node in nodes_to_draw}

    # Plot-Setup
    plt.figure(figsize=(12, 7)) # Etwas größer für bessere Lesbarkeit

    # Zeichne den Graphen
    nx.draw(
        G,
        pos=pos,                 # Verwende das definierte Layout
        nodelist=nodes_to_draw,  # Nur Knoten mit Position zeichnen
        labels=labels,           # Verwende die benutzerdefinierten Labels (2 Zeilen)
        node_size=node_sizes,    # Verwende die berechneten Größen
        node_color=node_colors,  # Verwende die definierten Farben
        arrows=True,             # Zeichne Pfeile für Kantenrichtung
        arrowstyle='-|>',        # Stil der Pfeilspitze
        arrowsize=15,            # Größe der Pfeilspitze
        edge_color="#888888",    # Etwas helleres Grau für Kanten
        font_size=9,             # Schriftgröße (ggf. anpassen)
        font_weight="normal",    # Normale Schriftstärke (fett kann bei kleiner Schrift schwer lesbar sein)
        font_color="black",      # Schriftfarbe
        linewidths=1.0,          # Randbreite der Knoten
        edgecolors="black"       # Randfarbe der Knoten
    )

    # Berechne die Summe der initialen Quellen für den Titel
    initial_sources = ["Head", "Lung", "Others", "MR-Glio"]
    total_initial_cases = sum(dataset_sizes.get(src, 0) for src in initial_sources)
    plt.title(f"Zusammensetzung des Datensatzes (Initiale Fälle: {total_initial_cases})", fontsize=16, pad=20)

    # Layout anpassen und speichern
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight') # bbox_inches für besseren Randabstand
    plt.close() # Schließt die Figur, gibt Speicher frei

    print(f"Plot gespeichert unter: {output_path}")
    return output_path


def save_middle_gtv_image(imgs, gts, case_name, out_dir):
    if 1 not in np.unique(gts):
        return
    gtv_slices = np.where(np.any(gts == 1, axis=(1, 2)))[0]
    if len(gtv_slices) == 0:
        return
    mid_slice = gtv_slices[len(gtv_slices) // 2]
    ct_slice = imgs[mid_slice]
    gtv_mask = (gts[mid_slice] == 1).astype(np.uint8)

    coords = cv2.findNonZero(gtv_mask)
    x, y, w, h = cv2.boundingRect(coords)
    center_x, center_y = x + w // 2, y + h // 2

    H, W = ct_slice.shape
    zoom_factor = 0.6  # 40% vom Rand entfernen
    new_w = int(W * zoom_factor)
    new_h = int(H * zoom_factor)

    start_x = max(0, center_x - new_w // 2)
    end_x = min(W, start_x + new_w)
    start_y = max(0, center_y - new_h // 2)
    end_y = min(H, start_y + new_h)

    ct_cropped = ct_slice[start_y:end_y, start_x:end_x]
    mask_cropped = gtv_mask[start_y:end_y, start_x:end_x]

    ct_resized = cv2.resize(ct_cropped, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_cropped, (W, H), interpolation=cv2.INTER_NEAREST)

    rgb_img = cv2.cvtColor(ct_resized, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb_img, contours, -1, (0, 0, 255), 2)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{case_name}_gtv.png")
    cv2.imwrite(out_path, rgb_img)

def create_dataset_histogram(overview_df):
    print("Erstelle Histogramm der Datensatzverteilung...")

    dataset_counts = overview_df.groupby(['Dataset', 'Split']).size().unstack(fill_value=0)
    if 'train' not in dataset_counts.columns:
        dataset_counts['train'] = 0
    if 'test' not in dataset_counts.columns:
        dataset_counts['test'] = 0

    dataset_counts = dataset_counts[['train', 'test']]

    dataset_mapping = {
        'npz_custom_Kopf': 'Head',
        'npz_custom_Lunge': 'Lung',
        'npz_custom_Rest': 'Others',
        'npz_custom_Mix': 'Mix',
        'npz_custom_Mix2': 'Mix\n(no Head)',
        'npz_files_MR': 'MR-Head',
        'npz_files_MRglio': 'MR-Glio',        
        'npz_files_MixMix': 'MR+CT'
    }
    dataset_order = ['Head', 'Lung', 'Others', 'Mix', 'Mix\n(no Head)', 'MR-Head', 'MR-Glio', 'MR+CT']

    dataset_counts.index = [dataset_mapping.get(dataset, dataset) for dataset in dataset_counts.index]
    dataset_counts = dataset_counts.reindex(dataset_order).dropna(how='all')

    fig, ax = plt.subplots(figsize=(14, 5))  # Erzeugt explizit ein Figure-Objekt mit Achse
    sns.set(style="whitegrid")
    dataset_counts.plot(kind='bar', width=0.7, ax=ax, zorder=2)  # Übergibt die Achse, die die Größe hat
    ax.grid(True, zorder=1)  # Grid unten

    for i, (index, row) in enumerate(dataset_counts.iterrows()):
        for j, value in enumerate(row):
            if pd.notna(value):
                ax.text(i + (j - 0.5) * 0.35, value + 1, str(int(value)), 
                        ha='center', va='bottom', fontsize=10, color='black')

    #plt.title('Number of Cases per Dataset and Split', fontsize=16)
    #plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(['Train', 'Test'], fontsize=12)

    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "dataset_distribution.png"), dpi=300, bbox_inches='tight')
    #plt.savefig(os.path.join(plots_dir, "dataset_distribution.pdf"), bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    print(f"Histogramm gespeichert in {plots_dir}")

def create_year_distribution_plot(df):
    print("Erstelle Jahresverteilung der einzigartigen Patient-Fälle...")
    if 'Patient_ID' not in df.columns or 'Plan ID' not in df.columns or 'LastTreatDate' not in df.columns:
        print("Notwendige Spalten fehlen für Jahresverteilung.")
        return

    df_unique = df.drop_duplicates(subset=["Patient_ID", "Plan ID"])
    df_unique = df_unique[df_unique['LastTreatDate'].notna()]
    df_unique['Year'] = df_unique['LastTreatDate'].str[:4]

    year_counts = df_unique['Year'].value_counts().sort_index()
    total_unique = len(df_unique)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax)

    # Zahlen über den Balken anzeigen
    for i, value in enumerate(year_counts.values):
        ax.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

    # Titel und Achsen
    #plt.title(f'Number of Unique Patient Cases per Year (n = {total_unique})', fontsize=16)
    print(f'Number of Unique Patient Cases per Year (n = {total_unique})')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Unique Cases', fontsize=14)
    plt.xticks(rotation=0)

    # Speichern
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "year_distribution.png"), dpi=300, bbox_inches='tight')
    #plt.savefig(os.path.join(plots_dir, "year_distribution.pdf"), bbox_inches='tight')
    plt.close()
    print("Jahresverteilung gespeichert.")

def create_summary_sheet(df):
    print("Erstelle Zusammenfassungs-Analyseblatt...")
    df_unique = df.drop_duplicates(subset=["Patient_ID", "Plan ID"])
    total = len(df_unique)
    train = len(df_unique[df_unique["Split"] == "train"])
    test = len(df_unique[df_unique["Split"] == "test"])

    summary = pd.DataFrame({
        "Metric": ["Total Unique Cases", "Train Unique Cases", "Test Unique Cases"],
        "Count": [total, train, test]
    })
    return summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    refine_df = pd.read_excel(REFINE_ALL_FILE, dtype=str)
    all_results = []

    for dataset_folder, modality in DATASET_DIRS:
        print(f"Processing {dataset_folder} ({modality})...")
        pseudo_file = PSEUDO_FILE_CT if modality == "CT" else PSEUDO_FILE_MR
        pseudo_df = pd.read_csv(pseudo_file, dtype=str)

        for split in ["train", "test"]:
            npz_dir = os.path.join(BASE_DIR, "data", dataset_folder, split)
            if not os.path.exists(npz_dir):
                continue

            npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
            for npz_path in tqdm(npz_files, desc=f"{dataset_folder} {split}"):
                filename = os.path.basename(npz_path)
                entry = {
                    "Case_Name": filename,
                    "Dataset": dataset_folder,
                    "Split": split,
                    "Modality": modality
                }

                match = pseudo_df[pseudo_df['NPZ_Filename'] == filename]
                if not match.empty:
                    entry["Patient_ID"] = match.iloc[0].get("Patient_ID", "")
                    entry["Case_Number"] = match.iloc[0].get("Case_Number", "")
                else:
                    entry["Patient_ID"] = "Unbekannt"
                    entry["Case_Number"] = "Unbekannt"

                refine_match = refine_df[refine_df['Patient ID'] == entry['Patient_ID']]
                if not refine_match.empty:
                    for col in refine_df.columns:
                        if col != "Patient ID":
                            entry[col] = refine_match.iloc[0].get(col, "")

                try:
                    if modality == "done_CT" or modality == "done_MR" or modality == "done_MR+CT":
                        data = np.load(npz_path, allow_pickle=True)
                        imgs = data['imgs']
                        gts = data['gts']
                        labels = data['labels'].item() if isinstance(data['labels'], np.ndarray) else data['labels']
                        case_name = os.path.splitext(filename)[0]

                        gtv_out = os.path.join(OUTPUT_DIR, "gtv_only", dataset_folder, split)
                        save_middle_gtv_image(imgs, gts, case_name, gtv_out)

                        if "xxxxx" in dataset_folder:
                            debug_out = os.path.join(OUTPUT_DIR, "debug_viz", dataset_folder, split)
                            visualize_slices(imgs, gts, labels, debug_out, case_name, num_slices=5)
                            
                except Exception as e:
                    print(f"Fehler bei Visualisierung von {filename}: {e}")

                all_results.append(entry)

    df = pd.DataFrame(all_results)
    summary_df = create_summary_sheet(df)
    with pd.ExcelWriter(EXCEL_OUTPUT) as writer:
        df.to_excel(writer, index=False, sheet_name="Overview")
        summary_df.to_excel(writer, index=False, sheet_name="Analysis")
    print(f"Excel gespeichert unter: {EXCEL_OUTPUT}")
    create_dataset_histogram(df)
    create_year_distribution_plot(df)    
    # DataSet-Größen (Train+Test)
    dataset_sizes = {
        "Head": 69, "Lung": 93, "Others": 92, "Mix": 255,
        "Mix (no Head)": 186, "MR-Glio": 63,
        "MR-Head": 83, "MR+CT": 269
    }

    # Plot generieren
    #plot_path = plot_dataset_structure(dataset_sizes)
    dataset_sizes_neu = {
        "Head": 69, "Lung": 93, "Others": 92, "Mix": 255,
        "Mix (no Head)": 186, "MR-Glio": 63,
        "MR-Head": 83, "MR+CT": 269
    }

    # Train/Test Zahlen aus dem Balkendiagramm extrahiert: {Node: (Train, Test)}
    train_test_daten = {
        "Head": (59, 10),
        "Lung": (83, 10),
        "Others": (80, 12),
        "Mix": (223, 32),
        "Mix (no Head)": (164, 22),
        "MR-Head": (67, 16),   # Aus dem Diagramm, nicht MR-Glio
        "MR-Glio": (49, 14),   # Aus dem Diagramm
        "MR+CT": (231, 38)
    }


    # Plot generieren mit optimiertem Code und Split-Visualisierung
    plot_path = plot_dataset_structure_with_split(dataset_sizes_neu, train_test_daten)
    
if __name__ == "__main__":
    main()

