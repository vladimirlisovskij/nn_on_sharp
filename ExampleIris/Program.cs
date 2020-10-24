using System;
using System.Collections.Generic;
using System.IO;
using activation;
using loss;
using neuron_list;
using Wraper;
using optimizer;

namespace ExampleIris
{
    class BodyWraper : AWraper
    {
        public BodyWraper(ILayer layer, AWraper next) : base(layer, next, null) {}

        public override object Feed(object rawX, object rawY)
        {
            return Layer.Back(Next.Feed(Layer.Predict(rawX), rawY));
        }
        
        public override object Predict(object rawX)
        {
            return Next.Predict(Layer.Predict(rawX));
        }
    }
    
    class HeadWraper : AWraper
    {
        public HeadWraper(ILayer layer, ILoss loss) : base(layer, null, loss) {}

        public override object Feed (object rawX, object rawY)
        {
            int[] y = rawY as int[];
            double[,] res = Layer.Predict(rawX) as double[,];
            double[] loss = new double[res.GetLength(1)];

            for (int i = 0; i < res.GetLength(1); ++i) loss[i] = 0;
            
            for (int batch = 0; batch < res.GetLength(0); ++batch)
            {
                double[] temp = new double[res.GetLength(1)];
                for (int tempNeur = 0; tempNeur < res.GetLength(1); tempNeur++)
                {
                    temp[tempNeur] = res[batch, tempNeur];
                }
                Console.WriteLine(this.Loss.Value(temp, y[batch]));
                temp = this.Loss.Derivative(temp, y[batch]) as double[];
                for (int tempNeur = 0; tempNeur < res.GetLength(1); tempNeur++)
                {
                    loss[tempNeur] += temp[tempNeur];
                }
            }

            for (int i = 0; i < res.GetLength(1); ++i) loss[i] /= res.GetLength(0);
            
            return Layer.Back(loss);
        }

        public override object Predict(object rawX)
        {
            return this.Layer.Predict(rawX);
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            HeadWraper head = new HeadWraper(new SoftMax(3), new CrossEntopy());
            BodyWraper body3 = new BodyWraper(new Layer(3, 3, new Sigm(), new Stohastic(0.001)), head);
            BodyWraper body2 = new BodyWraper(new Layer(5, 3, new Sigm(), new Stohastic(0.001)), body3);
            BodyWraper body1 = new BodyWraper(new Layer(4, 5, new Sigm(), new Stohastic(0.001)), body2);

            List<double[]> LX = new List<double[]>();
            List<int> LY = new List<int>();

            using (var reader = new StreamReader(@"./iris.data"))
            {
                while (!reader.EndOfStream)
                {
                    String line = reader.ReadLine();
                    if (String.IsNullOrEmpty(line)) continue;
                    
                    var values = line.Split(',');

                    double[] temp = new double[4];
                    for (int i = 0; i < 4; ++i)
                    {
                        String num = values[i];
                        num = num.Replace(".", ",");
                        temp[i] = double.Parse(num);
                    }
                    LX.Add(temp);

                    switch (values[4])
                    {
                        case "Iris-setosa":
                            LY.Add(0);
                            break;
                        case "Iris-versicolor":
                            LY.Add(1);
                            break;
                        case "Iris-virginica":
                            LY.Add(2);
                            break;
                    }

                }
            }

            double[][] x = LX.ToArray();
            int[] y = LY.ToArray();
            
            Random rnd = new Random();
            for  (int z = 0; z < 30; ++z)
            for (int i = 1; i < 150; ++i)
            {
                double[,] tempX = new double[1, 4];
                for (int j = 0; j < 4; ++j) tempX[0, j] = x[i][j];
                int[] tempY = new int[1] {y[i]};
                body1.Feed(tempX, tempY);
            }
            
            for (int i = 0; i < 10; ++i)
            {
                int num = rnd.Next(0, 149);
                double[,] tempX = new double[1, 4];
                for (int j = 0; j < 4; ++j) tempX[0, j] = x[num][j];
                double[,] res =  body1.Predict(tempX) as double[,];
                Console.WriteLine("\n\nRES");
                for (int j = 0; j < 3; ++j) Console.WriteLine(j + " : " + res[0, j]);
                Console.WriteLine(y[num]);
            }
            // double[,] res =  body1.Predict(new double[1, 2] {{7.7,2.8}}) as double[,];
            // Console.WriteLine("\n\nRES");
            // for (int j = 0; j < 3; ++j) Console.WriteLine(j + " : " + res[0, j]);
            
        }
    }
}