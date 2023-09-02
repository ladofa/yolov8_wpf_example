using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace wpf_yolov8
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        

        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {
            //SessionOptions gpuOption = new SessionOptions();
            //gpuOption.AppendExecutionProvider_DML(0);
            Yolov8.LoadOnnxSession("yolov8n (5).onnx");
        }

        OpenCvSharp.Mat mainMat;

        private void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.ShowDialog();

            mainMat = OpenCvSharp.Cv2.ImRead(openFileDialog.FileName);
            var source = OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(mainMat);
            MainImage.Source = source;
        }

        private void ProcessModelButton_Click(object sender, RoutedEventArgs e)
        {
            List<float[]> results = Yolov8.RunOnnxSesion(mainMat);

            Point relativeLocation = MainImage.TranslatePoint(new Point(0, 0), MainGrid);
            double marginWidth = relativeLocation.X;
            double marginHeight = relativeLocation.Y;

            var W = MainImage.ActualWidth;
            var H = MainImage.ActualHeight;

            ResultGrid.Children.Clear();

            foreach (float[] box in results)
            {
                var x1 = box[0] / 640 * W;
                var y1 = box[1] / 640 * H;
                var x2 = box[2] / 640 * W;
                var y2 = box[3] / 640 * H;
                var w = x2 - x1;
                var h = y2 - y1;

                var rect = new Rectangle()
                {
                    Width = w,
                    Height = h,
                    Margin = new Thickness(x1 + marginWidth, y1 + marginHeight, 0, 0),
                    HorizontalAlignment = HorizontalAlignment.Left,
                    VerticalAlignment = VerticalAlignment.Top,
                    Stroke = Brushes.Red,
                    StrokeThickness = 3,
                };

                ResultGrid.Children.Add(rect);
            }
        }
    }
}
