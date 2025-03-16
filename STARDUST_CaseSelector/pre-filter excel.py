import pandas as pd
import numpy as np
import os

# Datei einlesen (ersetze durch deinen Dateinamen)
input_file = ".\STARDUST_CaseSelector\PlansWithGTV.xlsx"
output_file = ".\STARDUST_CaseSelector\PlansWithGTV_filtered.xlsx"

# Excel-Datei einlesen
df = pd.read_excel(input_file)

# **Spalten sicher als numerisch casten (nicht konvertierbare Werte werden NaN)**
df["TargetVolume"] = pd.to_numeric(df["TargetVolume"], errors="coerce")
df["GTV-Volume"] = pd.to_numeric(df["GTV-Volume"], errors="coerce")

# **Zeilen entfernen, wo eine der beiden Spalten NaN ist (keine gültige Zahl)**
df = df.dropna(subset=["TargetVolume", "GTV-Volume"])

# **Label-Logik anwenden**
def assign_label(row):
    labels = []

    # Prüfe, ob "average" in "Series" vorkommt (Groß-/Kleinschreibung ignorieren)
    if isinstance(row["Series"], str) and "average" in row["Series"].lower():
        labels.append("A")

    # Prüfe auf Einträge in "KM_Series_Comment"
    if pd.notna(row["KM_Series_Comment"]) and row["KM_Series_Comment"].strip():
        labels.append("KMct")

    # Prüfe auf Einträge in "MR_Series_Comment"
    if pd.notna(row["MR_Series_Comment"]) and row["MR_Series_Comment"].strip():
        if "km" in row["MR_Series_Comment"].lower():
            labels.append("KMmr")
        else:
            labels.append("MR")

    return "+".join(labels) if labels else "None"

# Neue Spalte `Label` erzeugen
df["Label"] = df.apply(assign_label, axis=1)

# **Neue Funktion für `Label2` (Körperregion)**
def assign_body_region(series):
    if not isinstance(series, str):
        return "Sonstiges"  # Falls `Series` leer oder kein String ist

    series_lower = series.lower()

    # **Hals + Thorax zuerst prüfen, damit "Hals" nicht vorher als "Thorax" erkannt wird**
    if "hals" in series_lower and "thorax" in series_lower:
        return "Hals"
    if "thorax" in series_lower:
        return "Thorax"
    if "hals" in series_lower:
        return "Hals"
    if "abd" in series_lower:
        return "Abdomen"
    if "becken" in series_lower:
        return "Becken"
    if "bc" in series_lower:
        return "BC"
    if "mediastinum" in series_lower:
        return "Mediastinum"
    if "resp" in series_lower:
        return "Respiratorisch"
    if "kopf" in series_lower:
        return "Kopf"
    if "bws" in series_lower or "bws/lws" in series_lower or "bws lws" in series_lower:
        return "BWS"
    if "lws" in series_lower:
        return "LWS"
    if "hws" in series_lower:
        return "HWS"
    if "lunge" in series_lower:
        return "Lunge"
    if "leber" in series_lower:
        return "Leber"
    if "mamma" in series_lower:
        return "Mamma"
    if "extrem" in series_lower:
        return "Extremität"

    return "Sonstiges"

# Neue Spalte `Label2` für Körperregion
df["Label2"] = df["Series"].apply(assign_body_region)

# Add a new column for the percentage difference
df['PTV_Percentage_Larger'] = ((df['TargetVolume'] - df['GTV-Volume']) / df['GTV-Volume']) * 100
df['PTV_Percentage_Larger'] = df['PTV_Percentage_Larger'].round(2)
df['PTV_Radius'] = 10*(3 * df['TargetVolume'] / (4 * np.pi)) ** (1/3)
df['GTV_Radius'] = 10*(3 * df['GTV-Volume'] / (4 * np.pi)) ** (1/3)
df['Radius_Difference_mm'] = (df['PTV_Radius'] - df['GTV_Radius']).round(2)
# **Filter anwenden**
#df = df[(df["PTV_Percentage_Larger"] <= 1000) & 
 #       (df["PTV_Percentage_Larger"] >= 0) & 
  #      (df["Radius_Difference_mm"] <= 14)]
df = df[
        (df["PTV_Percentage_Larger"] >= 0)]
df = df[~df["GTV"].astype(str).str.contains("prä|op", case=False, na=False)]
df = df[df["GTV_Radius"] > 20]
df = df[(df["Fx"] > 4)&(df["Fx"] < 25)]
df = df[~df["TargetID"].astype(str).str.contains("sbl|sba|prost|sb|loge|hals|hno|neurocranium|wbrt|ganzhirn|anal|vulva|rectum|rectum|anus", case=False, na=False)]
df = df[~df["GTV"].astype(str).str.contains("sbl|sba|prost|sb|loge|hals|hno|neurocranium|wbrt|ganzhirn|anal|vulva|rectum|rectum|anus", case=False, na=False)]
df = df[~df["PlanID"].astype(str).str.contains("sbl|sba|prost|sb|loge|hals|hno|neurocranium|wbrt|ganzhirn|anal|vulva|rectum|rectum|anus", case=False, na=False)]
#df = df[df["GTV#>10GyAndInPTV"] == 1]
df = df[df["Patient-ID"].astype(str).str.match(r"^\d")]

# **Berechnung von TotalVolume**
df["TotalVolume"] = df["TargetVolume"] + df["GTV-Volume"]

# **Filterung: Behalte nur Zeilen mit minimalem PTV + GTV Volume pro SeriesUID**
df_filtered = df.loc[df.groupby("SeriesUID")["TotalVolume"].idxmin()]

# Unnötige Hilfsspalte entfernen
df_filtered.drop(columns=["TotalVolume"], inplace=True)

# **Ergebnis speichern**
df_filtered.to_excel(output_file, index=False)

print(f"Postprocessing abgeschlossen. Gefilterte Datei gespeichert als: {output_file}")

image_folder = os.path.dirname(df_filtered["PNG_Path"].dropna().iloc[0]) if not df_filtered["PNG_Path"].dropna().empty else None
# **🔹 Bild-Löschroutine**
if os.path.exists(image_folder):
    print(f"Prüfe überflüssige Bilder in: {image_folder}...")

    # **Liste aller gültigen Bildpfade aus der gefilterten Tabelle**
    valid_image_paths = set(df_filtered["PNG_Path"].dropna().astype(str))

    # **Durchsuche den Bilderordner und lösche unnötige Bilder**
    deleted_files = []
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)

        # Nur PNG-Dateien prüfen
        if file_path.endswith(".png") and file_path not in valid_image_paths:
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except Exception as e:
                print(f"Fehler beim Löschen von {file_path}: {e}")

    print(f"{len(deleted_files)} überflüssige Bilder gelöscht.")

else:
    print(f"Ordner '{image_folder}' existiert nicht. Kein Bild gelöscht.")