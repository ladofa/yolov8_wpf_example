using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Text;
using System.Windows;

namespace wpf_yolov8
{
    public static class Yolov8
    {
        static InferenceSession session;
        public static void LoadOnnxSession(string path)
        {
            session = new InferenceSession(path);
            MessageBox.Show("Model Loaded : " + session.ToString());
        }

        public static List<float[]> RunOnnxSesion(Mat input)
        {
            var mat = CvDnn.BlobFromImage(input, 1.0 / 255.0, new OpenCvSharp.Size(640, 640), swapRB: true, crop: false);
            float[] mem = new float[3 * 640 * 640];
            System.Runtime.InteropServices.Marshal.Copy(mat.DataStart, mem, 0, 640 * 640 * 3);
            var tensor = new DenseTensor<float>(mem, new[] { 1, 3, 640, 640 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>("images", tensor)
            };

            using (var output = session.Run(inputs))
            {
                var result = output.ElementAt(0).Value as DenseTensor<float>;

                if (result == null)
                {
                    return new List<float[]>();
                }
                var arr = result.ToArray();
                return postProcessing(arr);
            }
        }

        static List<float[]> postProcessing(float[] arr, int cat = 80, float thres = 0.3f)
        {
            int n = 8400;
            float[] maxProb = new float[n];
            float[] maxIndex = new float[n];

            List<float[]> boxes = new List<float[]>();

            for (int i = 0; i < n; i++)
            {
                float M = 0;
                float index = 0;
                for (int j = 0; j < cat; j++)
                {
                    var prob = arr[(j + 4) * n + i];
                    if (prob > M)
                    {
                        M = prob;
                        index = j;
                    }
                }

                if (M > 0.5)
                {
                    float cx = arr[0 * n + i];
                    float cy = arr[1 * n + i];
                    float w2 = arr[2 * n + i] / 2;
                    float h2 = arr[3 * n + i] / 2;
                    float x1 = cx - w2;
                    float y1 = cy - h2;
                    float x2 = cx + w2;
                    float y2 = cy + h2;
                    float[] box = new float[6]
                    {
                        x1, y1, x2, y2, M, index
                    };
                    boxes.Add(box);
                }
            }
            var sorted = boxes.OrderByDescending(x => x[5]);

            float iou(float[] box, float[] BOX)
            {
                float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
                float X1 = BOX[0], Y1 = BOX[1], X2 = BOX[2], Y2 = BOX[3];
                float ix1 = Math.Max(x1, X1);
                float ix2 = Math.Min(x2, X2);
                float iy1 = Math.Max(y1, Y1);
                float iy2 = Math.Min(y2, Y2);
                float iw = ix2 - ix1;
                float ih = iy2 - iy1;
                if (iw < 0 || ih < 0)
                    return 0;
                float inter = (ix2 - ix1) * (iy2 - iy1);

                float area = (x2 - x1) * (y2 - y1);
                float AREA = (X2 - X1) * (Y2 - Y1);
                float union = area + AREA - inter;
                return inter / union;
            };

            List<float[]> result = new List<float[]>();

            foreach (var candi in sorted)
            {
                bool found = false;
                foreach(var r in result)
                {
                    if (iou(r, candi) > 0.7)
                    {
                        found = true;
                        break;
                    }
                }
                
                if (!found)
                {
                    result.Add(candi);
                }
            }

            return result;
        }
    }
}
