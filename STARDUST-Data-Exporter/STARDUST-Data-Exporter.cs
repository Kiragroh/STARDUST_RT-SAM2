using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using VMS.TPS.Common.Model.API;
using System.IO;
using VMS.TPS.Common.Model.Types;
using System.Windows;
using Application = VMS.TPS.Common.Model.API.Application;
using OfficeOpenXml;
using System.Diagnostics;
using System.Reflection;

[assembly: ESAPIScript(IsWriteable = true)]
[assembly: AssemblyVersion("1.0.1.2")]

namespace STARDUST_DataExporter
{
    class Program
    {
        private static bool EXPORT_DICOM;
        private static bool EXPORT_MR;
        private static int MAX_PATIENTS;

        public const string AET = @"DCMTK";
        public const string AEC = @"VMSDBD1";
        public const string AEM = @"VMSFD1";
        public const string IP_PORT = @" ip port";
        public const string CMD_FILE_FMT = @"move-DICOMRT-{0}({1})-{2}.cmd";
        public const string ESAPIimportPath = @"path to export from DICOMdeamon\";
        private const string CT_TARGET_PATH = @"E:\STARDUSTmining\CT_Export\";
        private const string KM_TARGET_PATH = @"E:\STARDUSTmining\KM_Export\";
        private const string MR_TARGET_PATH = @"E:\STARDUSTmining\MR_Export\";
        private static string excelFilePath = @"PlansWithGTV_filtered_Mine.xlsx";
        private static string outputLogFile = @"Output_Log.csv";
        private static string cmdFolderPath = @"path to temp\cmd\";
        private static int processedPatients = 0;  // **Zähler für Patienten**

        [STAThread]
        static void Main(string[] args)
        {
            // **Einstellungen laden**
            LoadSettings();

            try
            {
                using (Application app = Application.CreateApplication())
                {
                    Execute(app);
                }
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e.ToString());
            }
            Console.WriteLine("Execution finished. Press any key to exit.");
            Console.ReadKey();
        }

        // **Lese Einstellungen aus "settings.ini" im Build-Ordner**
        private static void LoadSettings()
        {
            string settingsFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "settings.ini");

            if (!File.Exists(settingsFile))
            {
                Console.WriteLine("Einstellungsdatei nicht gefunden! Standardwerte werden verwendet.");
                EXPORT_DICOM = true;
                EXPORT_MR = true;
                MAX_PATIENTS = 5555555;
                return;
            }

            Dictionary<string, string> settings = new Dictionary<string, string>();

            foreach (string line in File.ReadAllLines(settingsFile))
            {
                if (!string.IsNullOrWhiteSpace(line) && line.Contains("="))
                {
                    string[] parts = line.Split('=');
                    if (parts.Length == 2)
                    {
                        settings[parts[0].Trim()] = parts[1].Trim().ToLower();
                    }
                }
            }

            // **Werte aus Datei setzen, falls vorhanden**
            EXPORT_DICOM = settings.ContainsKey("EXPORT_DICOM") ? settings["EXPORT_DICOM"] == "true" : true;
            EXPORT_MR = settings.ContainsKey("EXPORT_MR") ? settings["EXPORT_MR"] == "true" : true;

            // **Zählerwert aus Datei lesen**
            if (settings.ContainsKey("COUNTER") && int.TryParse(settings["COUNTER"], out int counterValue))
            {
                MAX_PATIENTS = counterValue;
            }
            else
            {
                MAX_PATIENTS = 5555555;
            }

            Console.WriteLine($"Lade Einstellungen: EXPORT_DICOM={EXPORT_DICOM}, EXPORT_MR={EXPORT_MR}, MAX_PATIENTS={MAX_PATIENTS}");
        }


        static void Execute(Application app)
        {
            Console.WriteLine($"Einstellungen geladen: EXPORT_DICOM={EXPORT_DICOM}, EXPORT_MR={EXPORT_MR}, MAX_PATIENTS={MAX_PATIENTS}");

            if (!File.Exists(excelFilePath))
            {
                Console.WriteLine("Excel-Datei nicht gefunden.");
                return;
            }

            File.WriteAllText(outputLogFile, "Patient-ID;SeriesUID;StructureSetID;Status\n");
            
            using (var package = new ExcelPackage(new FileInfo(excelFilePath)))
            {
                var worksheet = package.Workbook.Worksheets.First();
                int rowCount = worksheet.Dimension.Rows;

                for (int row = 2; row <= rowCount; row++)
                {
                    string patientID = worksheet.Cells[row, 1].Text;
                    string sourceSeriesUID = worksheet.Cells[row, 5].Text;
                    string sourceStructureSetID = worksheet.Cells[row, 10].Text;
                    string targetID = worksheet.Cells[row, 12].Text;
                    string gtvID = worksheet.Cells[row, 15].Text;
                    string kmSeriesUID = worksheet.Cells[row, 21].Text;
                    string mrSeriesUID = worksheet.Cells[row, 23].Text;
                    string mrRegUID = worksheet.Cells[row, 26].Text;

                    if (string.IsNullOrEmpty(patientID) || string.IsNullOrEmpty(sourceSeriesUID) || string.IsNullOrEmpty(sourceStructureSetID))
                        continue;
                    if (processedPatients >= MAX_PATIENTS)
                        break;
                    Patient patient = app.OpenPatientById(patientID);
                    if (patient == null)
                    {
                        LogEntry(patientID, sourceSeriesUID, sourceStructureSetID, "Patient nicht gefunden");
                        continue;
                    }

                    try
                    {
                        StructureSet sourceSS = patient.StructureSets
                            .FirstOrDefault(ss => ss.Image.Series.UID == sourceSeriesUID && ss.Id == sourceStructureSetID);

                        if (sourceSS == null)
                        {
                            LogEntry(patientID, sourceSeriesUID, sourceStructureSetID, "StructureSet nicht gefunden");
                            app.ClosePatient();
                            continue;
                        }

                        Image targetImage = sourceSS.Image;


                        // Überprüfen, ob das StructureSet mit der gewünschten ID bereits existiert
                        StructureSet newSS = patient.StructureSets.FirstOrDefault(ss => ss.Id == "zCopyFromPlan");

                        bool structureSetAlreadyExists = newSS != null;

                        if (!structureSetAlreadyExists)
                        {
                            // Falls nicht vorhanden, erstelle ein neues StructureSet
                            patient.BeginModifications();
                            newSS = sourceSS.Image.CreateNewStructureSet();
                            newSS.Id = "zCopyFromPlan";

                            foreach (var structure in sourceSS.Structures.Where(s =>
                                !s.IsEmpty &&
                                !s.Id.ToLower().StartsWith("z") &&
                                !s.Id.ToLower().Contains("prv") &&
                                !s.Id.ToLower().Contains("ring") &&
                                !s.Id.ToLower().Contains("body") &&
                                !s.Id.ToLower().Contains("haut") &&
                                !s.Id.ToLower().Contains("saum") &&
                                !s.Id.ToLower().Contains("couch") &&
                                !s.Id.ToLower().Contains("help")))
                            {
                                try
                                {
                                    string newName = structure.Id.Replace(" OAR", "");
                                    if (structure.Id == targetID) newName = "target+";
                                    if (structure.Id == gtvID) newName = "gtv+";

                                    Structure newStruct = newSS.AddStructure(structure.DicomType, newName);
                                    newStruct.Color = structure.Color;
                                    newStruct.SegmentVolume = structure.SegmentVolume;
                                }
                                catch { }
                            }

                            app.SaveModifications();
                            LogEntry(patient.Id, sourceSeriesUID, sourceStructureSetID, "StructureSet neu erstellt");
                        }
                        else
                        {
                            LogEntry(patient.Id, sourceSeriesUID, sourceStructureSetID, "StructureSet existiert bereits, Erstellung übersprungen");
                        }

                        // **Der Export läuft in jedem Fall**
                        if (EXPORT_DICOM)
                        {
                            ExportDicom(patient, newSS.Image, newSS);
                            MoveFilesToTarget(patient, newSS.Image, newSS, CT_TARGET_PATH);
                        }

                        StructureSet kmSS = null;
                        if (!string.IsNullOrEmpty(kmSeriesUID))
                        {
                            Image kmImage = patient.Studies.SelectMany(study => study.Images3D)
                                                           .FirstOrDefault(img => img.Series.UID == kmSeriesUID);

                            if (kmImage != null)
                            {
                                kmSS = patient.StructureSets.FirstOrDefault(ss => ss.Id == "zCopyFromPlan_KM");

                                if (kmSS == null)
                                {
                                    patient.BeginModifications();
                                    kmSS = kmImage.CreateNewStructureSet();
                                    kmSS.Id = "zCopyFromPlan_KM";

                                    foreach (var structure in newSS.Structures.Where(s => !s.IsEmpty))
                                    {
                                        try
                                        {
                                            Structure newStruct = kmSS.AddStructure(structure.DicomType, structure.Id);
                                            newStruct.Color = structure.Color;
                                            newStruct.SegmentVolume = structure.SegmentVolume;
                                        }
                                        catch { }
                                    }

                                    app.SaveModifications();
                                }

                                ExportDicom(patient, kmSS.Image, kmSS);
                                MoveFilesToTarget(patient, kmSS.Image, kmSS, KM_TARGET_PATH);
                            }
                        }

                        // MR-Export
                        Image mrImage = patient.Studies.SelectMany(study => study.Images3D)
                                                       .FirstOrDefault(img => img.Series.UID == mrSeriesUID);

                        if (mrImage != null && EXPORT_MR)
                        {
                            ExportDicomMR(patient, mrImage, mrRegUID);
                            MoveFilesToTarget(patient, mrImage, kmSS, MR_TARGET_PATH);
                        }

                        LogEntry(patientID, sourceSeriesUID, sourceStructureSetID, "Export erfolgreich");

                        processedPatients++;

                        if (processedPatients % 5 == 0)
                        {
                            DeleteEmptyFolders(ESAPIimportPath);
                            DeleteCmdFiles(cmdFolderPath);
                        }

                    }
                    catch (Exception ex)
                    {
                        LogEntry(patientID, sourceSeriesUID, sourceStructureSetID, "Fehler: " + ex.Message);
                    }

                    app.ClosePatient();
                }
            }
        }

        private static void DeleteEmptyFolders(string path)
        {
            foreach (var directory in Directory.GetDirectories(path))
            {
                DeleteEmptyFolders(directory);
                if (Directory.GetFiles(directory).Length == 0 && Directory.GetDirectories(directory).Length == 0)
                {
                    Directory.Delete(directory, false);
                }
            }
        }

        private static void DeleteCmdFiles(string path)
        {
            try
            {
                foreach (string file in Directory.GetFiles(path, "*.cmd"))
                {
                    File.Delete(file);
                }
                Console.WriteLine("Alle CMD-Dateien wurden gelöscht.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Fehler beim Löschen der CMD-Dateien: " + ex.Message);
            }
        }

        private static void ExportDicom(Patient patient, Image image, StructureSet structureSet)
        {
            DateTime dt = DateTime.Now;
            string datetext = dt.ToString("yyyyMMddHHmmss");
            string cmdFile = Path.Combine(cmdFolderPath,
                                          string.Format(CMD_FILE_FMT, patient.LastName, patient.Id, datetext));

            using (StreamWriter sw = new StreamWriter(cmdFile, false, Encoding.Default))
            {
                string DCMTK_BIN_PATH = Directory.Exists(@"D:\DCMTK\dcmtk-3.6.5-win64-dynamic\bin")
                    ? @"D:\DCMTK\dcmtk-3.6.5-win64-dynamic\bin"
                    : @"C:\DCMTK\dcmtk-3.6.5-win64-dynamic\bin";

                sw.WriteLine(@"@set PATH=%PATH%;" + DCMTK_BIN_PATH);

                // **CT-Bild exportieren**
                sw.WriteLine($"movescu -v -aet {AET} -aec {AEC} -aem {AEM} -S -k \"0008,0052=SERIES\" -k \"0020,000E={image.Series.UID}\" {IP_PORT}");

                // **StructureSet exportieren**
                sw.WriteLine($"movescu -v -aet {AET} -aec {AEC} -aem {AEM} -S -k \"0008,0052=IMAGE\" -k \"0008,0018={structureSet.UID}\" {IP_PORT}");
            }

            // **CMD-Datei ausführen**
            using (Process process = new Process())
            {
                process.StartInfo.FileName = "PowerShell.exe";
                process.StartInfo.Arguments = $"&'{cmdFile}'";
                process.StartInfo.UseShellExecute = false;
                process.Start();
                process.WaitForExit();
                process.Close();
            }
        }
        private static void ExportDicomMR(Patient patient, Image image, string regUID)
        {
            DateTime dt = DateTime.Now;
            string datetext = dt.ToString("yyyyMMddHHmmss");
            string cmdFile = Path.Combine(cmdFolderPath,
                                          string.Format(CMD_FILE_FMT, patient.LastName, patient.Id, datetext));

            using (StreamWriter sw = new StreamWriter(cmdFile, false, Encoding.Default))
            {
                string DCMTK_BIN_PATH = Directory.Exists(@"D:\DCMTK\dcmtk-3.6.5-win64-dynamic\bin")
                    ? @"D:\DCMTK\dcmtk-3.6.5-win64-dynamic\bin"
                    : @"C:\DCMTK\dcmtk-3.6.5-win64-dynamic\bin";

                sw.WriteLine(@"@set PATH=%PATH%;" + DCMTK_BIN_PATH);

                // **CT-Bild exportieren**
                sw.WriteLine($"movescu -v -aet {AET} -aec {AEC} -aem {AEM} -S -k \"0008,0052=SERIES\" -k \"0020,000E={image.Series.UID}\" {IP_PORT}");

                // **StructureSet exportieren**
                sw.WriteLine($"movescu -v -aet {AET} -aec {AEC} -aem {AEM} -S -k \"0008,0052=IMAGE\" -k \"0008,0018={regUID}\" {IP_PORT}");
            }

            // **CMD-Datei ausführen**
            using (Process process = new Process())
            {
                process.StartInfo.FileName = "PowerShell.exe";
                process.StartInfo.Arguments = $"&'{cmdFile}'";
                process.StartInfo.UseShellExecute = false;
                process.Start();
                process.WaitForExit();
                process.Close();
            }
        }

        private static void MoveFilesToTarget(Patient patient, Image image, StructureSet structureSet, string targetPath)
        {
            string patientFolder = Path.Combine(targetPath, patient.Id);
            string studyFolder = Path.Combine(patientFolder, image.Series.UID);

            if (!Directory.Exists(studyFolder))
                Directory.CreateDirectory(studyFolder);

            string sourcePath = Path.Combine(ESAPIimportPath, patient.Id);
            if (Directory.Exists(sourcePath))
            {
                foreach (string file in Directory.GetFiles(sourcePath))
                {                    
                    string destFile = Path.Combine(studyFolder, Path.GetFileName(file));
                    if (File.Exists(destFile)) File.Delete(destFile);
                    File.Move(file, destFile);                    
                }
            }
        }


        private static void LogEntry(string patientID, string seriesUID, string structureSetID, string status)
        {
            string logEntry = $"{patientID};{seriesUID};{structureSetID};{status}\n";
            Console.WriteLine(logEntry);
            File.AppendAllText(outputLogFile, logEntry);
        }
    }
}


