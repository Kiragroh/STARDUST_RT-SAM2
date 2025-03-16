using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using VMS.TPS.Common.Model.API;
using System.IO;
using VMS.TPS.Common.Model.Types;
using System.Collections;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows;
using Application = VMS.TPS.Common.Model.API.Application;
using OfficeOpenXml;

namespace STARDUSTtpsMining
{
    class Program
    {
        private static string baseImagePath = @"\\path\to\STARDUST";
        private static string excelFilePath = @"PlansWithGTV_filtered_Mine.xlsx";
        private static string outputLogFile = @"Output_Log.csv";

        [STAThread]
        static void Main(string[] args)
        {
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

        static void Execute(Application app)
        {
            if (!File.Exists(excelFilePath))
            {
                Console.WriteLine("Excel-Datei nicht gefunden.");
                return;
            }
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string outputFile = Path.Combine(baseImagePath, $"PlansWithGTV_{timestamp}.csv");

            string header = "Patient-ID;Nachname;Vorname;Series;SeriesUID;PlanID;Fx;GD;ED;StructureSetID;LastTreated;TargetID;TargetVolume;" +
                    "GTV#>10GyAndInPTV;GTV;GTV-Volume;HUavg;HU StdDev;RelVarHU;PNG_Path;KM_Series_UID;KM_Series_Comment;" +
                    "MR_Series_UID;MR_Slices;MR_Series_Comment;MRregUID;OtherGTV\n"; 
            File.WriteAllText(outputFile, header);

            File.WriteAllText(outputLogFile, "Patient-ID;SeriesUID;StructureSetID;Status\n");
            int counter = 0;
            using (var package = new ExcelPackage(new FileInfo(excelFilePath)))
            {
                var worksheet = package.Workbook.Worksheets.First();
                int rowCount = worksheet.Dimension.Rows;

                for (int row = 2; row <= rowCount; row++)
                {
                    string patientID = worksheet.Cells[row, 1].Text;
                    string PlanID = worksheet.Cells[row, 6].Text;

                    if (string.IsNullOrEmpty(patientID))
                        continue;
                    if (counter > 5555555)
                        break;
                    Patient p = app.OpenPatientById(patientID);
                    if (p == null)
                    {
                        LogEntry(patientID, "Patient nicht gefunden");
                        continue;
                    }

                    try
                    {
                        foreach (Course c in p.Courses.Where(x => !x.Patient.Id.ToLower().StartsWith("z")))
                        {
                            PlanSetup ps = c.PlanSetups.Where(x => x.Id == PlanID).FirstOrDefault();
                            if (ps == null)
                                continue;
                            
                            try
                            {
                                StructureSet ss = ps.StructureSet;
                                string gtvstructures = "";
                                double gtvVolume = 0;
                                string gtvId = "";
                                double avgHU = 0, stdHU = 0, relVarHU = 0;
                                string lastTreat = "";

                                try
                                {
                                    lastTreat = ps.TreatmentSessions
                                        .Where(x => x.Status == TreatmentSessionStatus.Completed)
                                        .Max(t => t.HistoryDateTime)
                                        .ToString("yyyyMMdd");
                                }
                                catch
                                {
                                    lastTreat = "Error";
                                }





                                if (ss.Structures.Any(x => !x.IsEmpty && x.Id.ToLower().StartsWith("gtv")) && ps.TargetVolumeID != "")
                                {
                                    Structure target = ss.Structures.FirstOrDefault(x => x.Id == ps.TargetVolumeID);
                                    Structure gtv = target;
                                    if (target == null) continue;



                                    int GTVcount = 0;
                                    foreach (Structure s in ss.Structures.Where(x => !x.IsEmpty && x.Id.ToLower().StartsWith("gtv") && target.IsPointInsideSegment(x.CenterPoint)))
                                    {
                                        DVHData dvhStat = ps.GetDVHCumulativeData(s, DoseValuePresentation.Absolute, VolumePresentation.AbsoluteCm3, 0.1);
                                        if (dvhStat.MeanDose.Dose < 5)
                                            continue;

                                        GTVcount++;
                                        gtvstructures += s.Id.Replace(";", "") + ", ";
                                        if (s.Volume > gtvVolume)
                                        {
                                            gtvVolume = Math.Round(s.Volume, 4);
                                            gtvId = s.Id;
                                        }
                                    }

                                    int zTarget = GetMiddleSlice(ss.Image, gtv);
                                    if (zTarget == -1) continue;

                                    gtv = ss.Structures.FirstOrDefault(x => x.Id == gtvId);
                                    // Berechne HU-Statistiken
                                    GetSliceHUStats(ss.Image, zTarget, gtv, out avgHU, out stdHU, out relVarHU);

                                    if (GTVcount == 0 || GTVcount > 5 || relVarHU.ToString().Contains("a"))
                                        continue;

                                    // Erstelle PNG-Pfad
                                    string patientFolder = Path.Combine(baseImagePath, "middleSlices");
                                    if (!Directory.Exists(patientFolder))
                                    {
                                        Directory.CreateDirectory(patientFolder);
                                    }

                                    string pngFilename = MakeFilenameValid($"{p.Id}_{ps.Id}_{ps.TargetVolumeID}_{gtvId}.png");
                                    string pngPath = Path.Combine(patientFolder, pngFilename);



                                    // **Neue Spalten für KM-Serie**
                                    string kmSeriesUID = "";
                                    string kmSeriesComment = "";

                                    // **Prüfe auf Serie mit gleichem FOR und "KM" im Kommentar**
                                    var matchingKMSeries = p.Studies.SelectMany(study => study.Series)
                                                                    .FirstOrDefault(s => s.FOR == ss.Image.Series.FOR && s.Comment.Contains("KM"));

                                    if (matchingKMSeries != null)
                                    {
                                        kmSeriesUID = matchingKMSeries.UID;
                                        kmSeriesComment = matchingKMSeries.Comment;
                                        SaveContourAsPNG(matchingKMSeries.Images.Where(x => string.IsNullOrEmpty(x.UID)).FirstOrDefault(), zTarget, ss, pngPath.Replace(".png", "_KM.png"));
                                    }
                                    // **Finde registriertes MR mit den meisten Slices**
                                    string mrSeriesUID = "", mrSeriesComment = "";
                                    int mrMaxSlices = 0;
                                    var registeredMR = FindRegisteredMRWithMostSlices(p, ss.Image.Series.FOR);
                                    string MRregUID = "";

                                    // Speichere das PNG
                                    SaveContourAsPNG(ss.Image, zTarget, ss, pngPath);

                                    if (registeredMR.series != null)
                                    {
                                        // Schritt 1: Berechne die relative Position der z-Slice im CT
                                        double zMinCT = ss.Image.Origin.z;
                                        double zMaxCT = ss.Image.Origin.z + (ss.Image.ZSize - 1) * ss.Image.ZRes;
                                        double relativePosition = (zTarget * ss.Image.ZRes + ss.Image.Origin.z - zMinCT) / (zMaxCT - zMinCT);

                                        Image mrImage = registeredMR.series.Images.Where(x => string.IsNullOrEmpty(x.UID)).FirstOrDefault();

                                        mrSeriesUID = registeredMR.series.UID;
                                        mrSeriesComment = registeredMR.series.Comment;
                                        mrMaxSlices = registeredMR.series.Images.Count();
                                        MRregUID = registeredMR.regUID;
                                        int mrSlice = GetMatchingMRSlice(p.Registrations.FirstOrDefault(r=>r.UID==MRregUID), ss.Image, mrImage, gtv);
                                        SaveContourAsPNG(mrImage, mrSlice, ss, pngPath.Replace(".png", "_MR.png"));

                                    }

                                    counter++;
                                    string message = $"{p.Id};{p.LastName.Replace(",", "")};{p.FirstName.Replace(",", "")};" +
                                                $"{ss.Image.Series.Comment};{ss.Image.Series.UID};{ps.Id};{ps.NumberOfFractions};" +
                                                $"{Math.Round(ps.TotalDose.Dose, 2)};{Math.Round(ps.DosePerFraction.Dose, 2)};{ss.Id};{lastTreat};" +
                                                $"{ps.TargetVolumeID};{Math.Round(target.Volume, 4)};{GTVcount};{gtvId};{gtvVolume};" +
                                                $"{avgHU:F2};{stdHU:F2};{relVarHU:P2};{pngPath};{kmSeriesUID};{kmSeriesComment};" +
                                                $"{mrSeriesUID};{mrMaxSlices};{mrSeriesComment};{MRregUID};{gtvstructures.Replace(";", ".")}\n";


                                    Console.WriteLine(message);
                                    File.AppendAllText(outputFile, message);
                                }
                            }
                            catch (Exception e) { Console.WriteLine("small_" + e.ToString() + "\n"); LogEntry(patientID, "Fehler: " + e.Message); }
                            
                        }
                    }
                    catch (Exception e) { Console.WriteLine("big_" + e.ToString() + "\n"); LogEntry(patientID, "Fehler: " + e.Message); }
                    app.ClosePatient();
                }
                Console.WriteLine("DONE collecting StructureInfos, file saved here {0}.", outputFile);

                // **Hier CSV zu Excel umwandeln**
                ConvertCsvToExcel(outputFile);
            }
        }
        private static void ConvertCsvToExcel(string csvPath)
        {
            string excelPath = Path.ChangeExtension(csvPath, ".xlsx");

            //ExcelPackage.LicenseContext = LicenseContext.NonCommercial; // **WICHTIG für EPPlus 5+**

            using (var package = new ExcelPackage(new FileInfo(excelPath)))
            {
                var worksheet = package.Workbook.Worksheets.Add("Data");

                var lines = File.ReadAllLines(csvPath);
                for (int i = 0; i < lines.Length; i++)
                {
                    var columns = lines[i].Split(';'); // Falls Trennzeichen anders ist, anpassen
                    for (int j = 0; j < columns.Length; j++)
                    {
                        worksheet.Cells[i + 1, j + 1].Value = columns[j];
                    }
                }

                package.Save();
            }

            Console.WriteLine($"Excel file created: {excelPath}");
        }
        private static int GetMiddleSlice(Image image, Structure structure)
        {
            var contourSlices = Enumerable.Range(0, image.ZSize)
                                          .Where(z => structure.GetContoursOnImagePlane(z).Any())
                                          .ToList();

            if (!contourSlices.Any()) return -1;

            return contourSlices[contourSlices.Count / 2];
        }

        private static void GetSliceHUStats(Image image, int sliceZ, Structure structure, out double avgHU, out double stdHU, out double relVarHU)
        {
            var contour = structure.GetContoursOnImagePlane(sliceZ);
            if (!contour.Any())
            {
                avgHU = double.NaN;
                stdHU = double.NaN;
                relVarHU = double.NaN;
                return;
            }

            int[,] buffer = new int[image.XSize, image.YSize];
            image.GetVoxels(sliceZ, buffer);

            List<double> huValues = new List<double>();

            foreach (var segment in contour)
            {
                foreach (var point in segment)
                {
                    int x = (int)Math.Round((point.x - image.Origin.x) / image.XRes);
                    int y = (int)Math.Round((point.y - image.Origin.y) / image.YRes);

                    if (x >= 0 && x < image.XSize && y >= 0 && y < image.YSize)
                    {
                        huValues.Add(image.VoxelToDisplayValue(buffer[x, y]));
                    }
                }
            }

            if (huValues.Count == 0)
            {
                avgHU = stdHU = relVarHU = double.NaN;
                return;
            }

            double mean = huValues.Average();
            double variance = huValues.Select(v => Math.Pow(v - mean, 2)).Average();
            double stdDev = Math.Sqrt(variance);
            double relativeVariance = mean != 0 ? stdDev / Math.Abs(mean) : double.NaN;

            avgHU = mean;
            stdHU = stdDev;
            relVarHU = relativeVariance;
        }

        private static void SaveContourAsPNG(Image image, int sliceZ, StructureSet structureSet, string filename)
        {
            int width = image.XSize;
            int height = image.YSize;
            bool isMR = false;
            isMR = filename.Contains("_MR.png");
            Console.WriteLine(isMR.ToString() +"_"+filename);
            // Finde die "Body"-Struktur nur für CT-Bilder
            Structure body = !isMR && structureSet != null ? structureSet.Structures.FirstOrDefault(s => s.Id.ToLower().StartsWith("body") || s.Id.ToLower().StartsWith("körper") || s.Id.ToLower().StartsWith("outer")) : null;

            // Berechne Bounding-Box nur für CT-Bilder
            Rect? bodyBounds = (body != null) ? (Rect?)GetBoundingBox(body, image, sliceZ) : (Rect?)null;


            // Bestimme die aktuelle Größe
            double currentWidth = bodyBounds.HasValue ? bodyBounds.Value.Width : width;
            double currentHeight = bodyBounds.HasValue ? bodyBounds.Value.Height : height;
            double maxDim = Math.Max(currentWidth, currentHeight);

            // Setze die gewünschte Mindestgröße, z. B. 512 Pixel
            double desiredSize = 888;
            double scale = desiredSize / maxDim;
            if (scale < 1.0) scale = 1.0; // Nicht verkleinern, nur vergrößern

            // Berechne die Rendergröße
            int renderWidth = (int)(currentWidth * scale);
            int renderHeight = (int)(currentHeight * scale);

            DrawingVisual drawingVisual = new DrawingVisual();
            using (DrawingContext dc = drawingVisual.RenderOpen())
            {
                // Skaliere den DrawingContext
                dc.PushTransform(new ScaleTransform(scale, scale));

                // Zeichne das Bild als Graustufen-Hintergrund
                if (bodyBounds.HasValue & !isMR)
                {
                    Console.WriteLine("if: " + isMR.ToString() + "_" + filename);
                    DrawCTSlice(dc, image, sliceZ, (int)bodyBounds.Value.X, (int)bodyBounds.Value.Y,
                                (int)bodyBounds.Value.Width, (int)bodyBounds.Value.Height);
                }
                else if (isMR)
                {
                    Console.WriteLine("else if: "+isMR.ToString() + "_" + filename);
                    DrawMRSlice(dc, image, sliceZ, 0, 0, width, height);
                }
                else
                {
                    Console.WriteLine("else: " + isMR.ToString() + "_" + filename);
                    DrawCTSlice(dc, image, sliceZ, 0, 0, width, height);
                }

                // Zeichne die Konturen nur für CT-Bilder
                if (structureSet != null && !isMR)
                {
                    foreach (var structure in structureSet.Structures)
                    {
                        if (!structure.Id.ToUpper().StartsWith("Z") && !structure.Id.ToLower().Contains("-oar") && !structure.Id.ToLower().Contains("couch") &&
                            !structure.Id.ToLower().Contains("haut") && !structure.Id.ToUpper().StartsWith("HK") &&
                            structure.GetContoursOnImagePlane(sliceZ).Any())
                        {
                            System.Windows.Media.Color structureColor = System.Windows.Media.Color.FromRgb(
                                structure.Color.R, structure.Color.G, structure.Color.B);
                            // Passe die Stiftbreite an die Skalierung an
                            Pen pen = new Pen(new SolidColorBrush(structureColor), 1 / scale);

                            var contour = structure.GetContoursOnImagePlane(sliceZ);
                            foreach (var segment in contour)
                            {
                                if (segment.Length > 1)
                                {
                                    StreamGeometry geometry = new StreamGeometry();
                                    using (StreamGeometryContext sgc = geometry.Open())
                                    {
                                        sgc.BeginFigure(ConvertToZoomedPoint(segment[0], image, bodyBounds), false, false);
                                        sgc.PolyLineTo(
                                            segment.Skip(1).Select(p => ConvertToZoomedPoint(p, image, bodyBounds)).ToArray(),
                                            true,
                                            false
                                        );
                                    }
                                    dc.DrawGeometry(null, pen, geometry);
                                }
                            }
                        }
                    }
                }
                dc.Pop(); // Transformierung zurücksetzen
            }

            // Render das Bild in ein RenderTargetBitmap
            RenderTargetBitmap bmp = new RenderTargetBitmap(renderWidth, renderHeight, 96, 96, PixelFormats.Pbgra32);
            bmp.Render(drawingVisual);

            // Speichern als PNG
            using (var fileStream = new System.IO.FileStream(filename, System.IO.FileMode.Create))
            {
                PngBitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bmp));
                encoder.Save(fileStream);
            }
        }

        private static void SaveContourAsPNGwithDose(Image image, int sliceZ, StructureSet structureSet, PlanSetup ps, string filename)
        {
            int width = image.XSize;
            int height = image.YSize;
            bool isMR = filename.Contains("_MR.png");
            Console.WriteLine(isMR.ToString() + "_" + filename);

            // Finde die "Body"-Struktur nur für CT-Bilder
            Structure body = !isMR && structureSet != null ? structureSet.Structures.FirstOrDefault(s => s.Id.ToLower().StartsWith("body") || s.Id.ToLower().StartsWith("körper") || s.Id.ToLower().StartsWith("outer")) : null;

            // Berechne Bounding-Box nur für CT-Bilder
            Rect? bodyBounds = (body != null) ? (Rect?)GetBoundingBox(body, image, sliceZ) : null;

            // Schritt 2: Dosisdaten holen
            double[,] doseData = GetDoseData(ps, sliceZ);

            // Bestimme die aktuelle Größe
            double currentWidth = bodyBounds.HasValue ? bodyBounds.Value.Width : width;
            double currentHeight = bodyBounds.HasValue ? bodyBounds.Value.Height : height;
            double maxDim = Math.Max(currentWidth, currentHeight);

            // Setze die gewünschte Mindestgröße, z. B. 888 Pixel
            double desiredSize = 888;
            double scale = desiredSize / maxDim;
            if (scale < 1.0) scale = 1.0; // Nicht verkleinern, nur vergrößern

            // Berechne die Rendergröße
            int renderWidth = (int)(currentWidth * scale);
            int renderHeight = (int)(currentHeight * scale);

            // Hintergrund rendern (Bild + Konturen)
            DrawingVisual backgroundVisual = new DrawingVisual();
            using (DrawingContext dc = backgroundVisual.RenderOpen())
            {
                dc.PushTransform(new ScaleTransform(scale, scale));

                // Zeichne das Bild als Graustufen-Hintergrund
                if (bodyBounds.HasValue && !isMR)
                {
                    Console.WriteLine("if: " + isMR.ToString() + "_" + filename);
                    DrawCTSlice(dc, image, sliceZ, (int)bodyBounds.Value.X, (int)bodyBounds.Value.Y,
                                (int)bodyBounds.Value.Width, (int)bodyBounds.Value.Height);
                }
                else if (isMR)
                {
                    Console.WriteLine("else if: " + isMR.ToString() + "_" + filename);
                    DrawMRSlice(dc, image, sliceZ, 0, 0, width, height);
                }
                else
                {
                    Console.WriteLine("else: " + isMR.ToString() + "_" + filename);
                    DrawCTSlice(dc, image, sliceZ, 0, 0, width, height);
                }

                // Zeichne die Konturen nur für CT-Bilder
                if (structureSet != null && !isMR)
                {
                    foreach (var structure in structureSet.Structures)
                    {
                        if (!structure.Id.ToUpper().StartsWith("Z") && !structure.Id.ToLower().Contains("-oar") &&
                            !structure.Id.ToLower().Contains("couch") && !structure.Id.ToLower().Contains("haut") &&
                            !structure.Id.ToUpper().StartsWith("HK") && structure.GetContoursOnImagePlane(sliceZ).Any() &&
                            structure.Volume < 400)
                        {
                            System.Windows.Media.Color structureColor = System.Windows.Media.Color.FromRgb(
                                structure.Color.R, structure.Color.G, structure.Color.B);
                            Pen pen = new Pen(new SolidColorBrush(structureColor), 1 / scale);

                            var contour = structure.GetContoursOnImagePlane(sliceZ);
                            foreach (var segment in contour)
                            {
                                if (segment.Length > 1)
                                {
                                    StreamGeometry geometry = new StreamGeometry();
                                    using (StreamGeometryContext sgc = geometry.Open())
                                    {
                                        sgc.BeginFigure(ConvertToZoomedPoint(segment[0], image, bodyBounds), false, false);
                                        sgc.PolyLineTo(segment.Skip(1).Select(p => ConvertToZoomedPoint(p, image, bodyBounds)).ToArray(),
                                                       true, false);
                                    }
                                    dc.DrawGeometry(null, pen, geometry);
                                }
                            }
                        }
                    }
                }
                dc.Pop(); // Transformierung zurücksetzen
            }

            // RenderTargetBitmap für den Hintergrund erstellen
            RenderTargetBitmap backgroundBmp = new RenderTargetBitmap(renderWidth, renderHeight, 96, 96, PixelFormats.Pbgra32);
            backgroundBmp.Render(backgroundVisual);

            // Dosis halbtransparent rendern
            DrawingVisual doseVisual = new DrawingVisual();
            using (DrawingContext dc = doseVisual.RenderOpen())
            {
                dc.PushTransform(new ScaleTransform(scale, scale));

                // Hole die Ursprünge vom Dosisraster und vom Bild
                VVector doseOrigin = ps.Dose.Origin;
                VVector imageOrigin = image.Origin;

                // Auflösungen
                double doseResX = ps.Dose.XRes;
                double doseResY = ps.Dose.YRes;
                double imageResX = image.XRes;
                double imageResY = image.YRes;

                // Schwellenwert für die Dosisanzeige (z.B. 10% der maximalen Dosis)
                double doseThreshold = ps.TotalDose.Dose * 0.14; // 10% der Maximaldosis
                                                                 // Du kannst auch einen absoluten Wert verwenden, z.B. 5 Gy
                                                                 // double doseThreshold = 5.0; 
                double doseOpacity = 0.4;

                for (int i = 0; i < doseData.GetLength(0); i++)
                {
                    for (int j = 0; j < doseData.GetLength(1); j++)
                    {
                        double doseValue = doseData[i, j];
                        if (doseValue > doseThreshold) // Nur Dosis > Schwellenwert anzeigen
                        {
                            System.Windows.Media.Color color = GetColorFromDose(doseValue, ps.TotalDose.Dose);
                            color.A = (byte)(255 * doseOpacity); // Halbtransparent (50% Opacity)
                            Brush brush = new SolidColorBrush(color);

                            // Konvertiere von Dosisrasterindizes zu Patientenkoordinaten
                            double patientX = doseOrigin.x + (i * doseResX);
                            double patientY = doseOrigin.y + (j * doseResY);

                            // Konvertiere von Patientenkoordinaten zu Bildpixelkoordinaten
                            double x = (patientX - imageOrigin.x) / imageResX;
                            double y = (patientY - imageOrigin.y) / imageResY;

                            // Voxelgröße im Bildkoordinatensystem
                            double voxelWidth = doseResX / imageResX;
                            double voxelHeight = doseResY / imageResY;

                            // Anpassung für bodyBounds-Cropping falls erforderlich
                            if (bodyBounds.HasValue)
                            {
                                x -= bodyBounds.Value.X;
                                y -= bodyBounds.Value.Y;
                            }

                            dc.DrawRectangle(brush, null, new Rect(x, y, voxelWidth, voxelHeight));
                        }
                    }
                }
                dc.Pop();
            }

            // RenderTargetBitmap für die Dosis erstellen
            RenderTargetBitmap doseBmp = new RenderTargetBitmap(renderWidth, renderHeight, 96, 96, PixelFormats.Pbgra32);
            doseBmp.Render(doseVisual);

            // Create a DrawingVisual to combine the bitmaps
            DrawingVisual combinedVisual = new DrawingVisual();
            using (DrawingContext dc = combinedVisual.RenderOpen())
            {
                // Draw the background bitmap
                dc.DrawImage(backgroundBmp, new Rect(0, 0, renderWidth, renderHeight));
                // Draw the dose bitmap on top (assuming it has transparency)
                dc.DrawImage(doseBmp, new Rect(0, 0, renderWidth, renderHeight));
            }

            // Create the final bitmap and render the combined visual onto it
            RenderTargetBitmap finalBmp = new RenderTargetBitmap(renderWidth, renderHeight, 96, 96, PixelFormats.Pbgra32);
            finalBmp.Render(combinedVisual);

            // Save to PNG (assuming this is your next step)
            using (var fileStream = new FileStream(filename, FileMode.Create))
            {
                PngBitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(finalBmp));
                encoder.Save(fileStream);
            }
        }
        private static System.Windows.Media.Color GetColorFromDose(double dose, double maxDose)
        {
            double normalized = dose / maxDose;
            if (normalized < 0) normalized = 0;
            if (normalized > 1) normalized = 1;

            byte r = (byte)(255 * normalized); // Rotanteil steigt mit Dosis
            byte b = (byte)(255 * (1 - normalized)); // Blauanteil sinkt mit Dosis
            return System.Windows.Media.Color.FromRgb(r, 0, b);
        }


        private static double[,] GetDoseData(PlanSetup plan, int planeIndex)
        {
            plan.DoseValuePresentation = DoseValuePresentation.Absolute;
            var dose = plan.Dose;
            if (dose == null) throw new Exception("Keine Dosisdaten im Plan verfügbar.");

            var data = new int[dose.XSize, dose.YSize];
            dose.GetVoxels(planeIndex, data);

            var doseMatrix = new double[dose.XSize, dose.YSize];
            for (int i = 0; i < dose.XSize; i++)
                for (int j = 0; j < dose.YSize; j++)
                    doseMatrix[i, j] = dose.VoxelToDoseValue(data[i, j]).Dose; // Dosis in Gy
            return doseMatrix;
        }

        private static Rect GetBoundingBox(Structure body, Image image, int sliceZ)
        {
            var contours = body.GetContoursOnImagePlane(sliceZ);
            if (!contours.Any()) return new Rect(0, 0, 0, 0);

            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            foreach (var segment in contours)
            {
                foreach (var point in segment)
                {
                    double x = (point.x - image.Origin.x) / image.XRes;
                    double y = (point.y - image.Origin.y) / image.YRes;

                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
            // Erweitere die Bounding Box um 5 Einheiten in alle Richtungen
            double padding = 5;
            minX -= padding;
            maxX += padding;
            minY -= padding;
            maxY += padding;

            return new Rect(minX, minY, maxX - minX, maxY - minY);
        }
        private static void DrawMRSlice(DrawingContext dc, Image image, int sliceZ, int startX, int startY, int width, int height)
        {
            int[,] buffer = new int[image.XSize, image.YSize];
            image.GetVoxels(sliceZ, buffer);

            byte[] pixels = new byte[width * height];

            // **Automatisches Windowing für MR**
            int minIntensity = buffer.Cast<int>().Min();
            int maxIntensity = buffer.Cast<int>().Max();

            // Sicherheitscheck, um Division durch Null zu vermeiden
            if (minIntensity == maxIntensity) maxIntensity = minIntensity + 1;

            for (int x = startX; x < startX + width; x++)
            {
                for (int y = startY; y < startY + height; y++)
                {
                    int voxel = buffer[x, y];
                    double intensity = (voxel - minIntensity) / (double)(maxIntensity - minIntensity) * 255.0;

                    byte pixelValue = (byte)Math.Max(0, Math.Min(255, intensity));

                    int localX = x - startX;
                    int localY = y - startY;
                    pixels[localY * width + localX] = pixelValue;
                }
            }

            BitmapSource bitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, pixels, width);
            RenderOptions.SetBitmapScalingMode(bitmap, BitmapScalingMode.HighQuality);
            dc.DrawImage(bitmap, new System.Windows.Rect(0, 0, width, height));
        }

        private static void DrawCTSlice(DrawingContext dc, Image image, int sliceZ, int startX, int startY, int width, int height)
        {
            int[,] buffer = new int[image.XSize, image.YSize];
            image.GetVoxels(sliceZ, buffer);

            byte[] pixels = new byte[width * height];
            // **Windowing je nach Kopf-Scan anpassen**
            string seriesComment = image.Series.Comment?.ToLower() ?? "";
            double windowCenter = seriesComment.Contains("kopf") ? 80 : 0;
            double windowWidth = seriesComment.Contains("kopf") ? 350 : 900;            

            for (int x = startX; x < startX + width; x++)
            {
                for (int y = startY; y < startY + height; y++)
                {
                    int voxel = buffer[x, y];
                    double hu = image.VoxelToDisplayValue(voxel);
                    // **Windowing-Anpassung für bessere Kontraste**
                    double intensity = ((hu - (windowCenter - windowWidth / 2)) / windowWidth) * 255.0;
                    byte pixelValue = (byte)Math.Max(0, Math.Min(255, intensity));
                    int localX = x - startX;
                    int localY = y - startY;
                    pixels[localY * width + localX] = pixelValue;
                }
            }

            BitmapSource bitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, pixels, width);
            // Verbessere die Skalierungsqualität
            RenderOptions.SetBitmapScalingMode(bitmap, BitmapScalingMode.HighQuality);
            dc.DrawImage(bitmap, new System.Windows.Rect(0, 0, width, height));
        }

        private static System.Windows.Point ConvertToZoomedPoint(VVector point, Image image, Rect? bodyBounds)
        {
            double x = (point.x - image.Origin.x) / image.XRes;
            double y = (point.y - image.Origin.y) / image.YRes;

            if (bodyBounds.HasValue)
            {
                x -= bodyBounds.Value.X;
                y -= bodyBounds.Value.Y;
            }

            return new System.Windows.Point(x, y);
        }

        private static string MakeFilenameValid(string s)
        {
            char[] invalidChars = System.IO.Path.GetInvalidFileNameChars();
            foreach (char ch in invalidChars)
            {
                s = s.Replace(ch, '_');
            }
            return s;
        }
        private static (Series series, string regUID) FindRegisteredMRWithMostSlices(Patient patient, string referenceFOR)
        {
            Series bestMR = null;
            Series fallbackMR = null;
            int maxSlicesPreferred = 0;
            int maxSlicesFallback = 0;
            string regUIDPreferred = "";
            string regUIDFallback = "";

            foreach (var reg in patient.Registrations)
            {
                // **Registrierung muss gültig sein (kein "ONLINE" oder "EXACTRAC")**
                if (string.IsNullOrEmpty(reg.Id) || reg.Id.StartsWith("ONLINE") || reg.Id.StartsWith("EXACTRAC"))
                    continue;

                // **Prüfe, ob das FOR übereinstimmt**
                bool isRegisteredToReference =
                    (reg.RegisteredFOR == referenceFOR || reg.SourceFOR == referenceFOR);

                if (!isRegisteredToReference)
                    continue;

                // **Durchsuche alle MR-Serien**
                foreach (var series in patient.Studies.SelectMany(study => study.Series)
                                                      .Where(s => s.Modality.ToString() == "MR"))
                {
                    string comment = series.Comment?.ToLower() ?? "";

                    // **Bevorzuge Serien mit "ax" oder "tra" im Kommentar**
                    if (comment.Contains("ax") || comment.Contains("tra"))
                    {
                        if (series.Images.Count() > maxSlicesPreferred)
                        {
                            maxSlicesPreferred = series.Images.Count();
                            bestMR = series;
                            regUIDPreferred = reg.UID;
                        }
                    }
                    else
                    {
                        // **Fallback auf die Serie mit den meisten Slices, falls keine "ax" oder "tra" gefunden wird**
                        if (series.Images.Count() > maxSlicesFallback)
                        {
                            maxSlicesFallback = series.Images.Count();
                            fallbackMR = series;
                            regUIDFallback = reg.UID;
                        }
                    }
                }
            }

            // **Falls eine passende "ax" oder "tra" MR gefunden wurde, verwende sie**
            if (bestMR != null)
            {
                return (bestMR, regUIDPreferred);
            }

            // **Andernfalls verwende einfach die MR-Serie mit den meisten Slices**
            return (fallbackMR, regUIDFallback);
        }
        private static void LogEntry(string patientID, string status)
        {
            string logEntry = $"{patientID};{status}\n";
            Console.WriteLine(logEntry);
            File.AppendAllText(outputLogFile, logEntry);
        }
        private static int GetMatchingMRSlice(Registration registration, Image ctImage, Image mrImage, Structure gtv)
        {
            if (registration == null || ctImage == null || mrImage == null)
                return mrImage.ZSize / 2; // Fallback: Mitte des MR-Volumes, falls keine Registrierung

            
            VVector transformedPoint = gtv.CenterPoint;
            // **Transformiere den Punkt ins MR-Koordinatensystem**
            if (registration.SourceFOR == ctImage.Series.FOR)
                transformedPoint = registration.TransformPoint(gtv.CenterPoint);
            if (registration.RegisteredFOR == ctImage.Series.FOR)
                transformedPoint = registration.InverseTransformPoint(gtv.CenterPoint);

            // **Finde den nächsten passenden MR-Slice**
            int bestSlice = 0;
            double minDistance = double.MaxValue;

            for (int z = 0; z < mrImage.ZSize; z++)
            {
                double zMR = mrImage.Origin.z + z * mrImage.ZRes;
                double distance = Math.Abs(zMR - transformedPoint.z);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    bestSlice = z;
                }
            }

            return bestSlice;
        }
    }
}
