﻿using System;
using activation;
using optimizer;
using matrix;

namespace neuron_list
{
    public interface ILayer
    {
        public object Predict(object rawData);
        public object Back(object rawData);
    }

    public class SoftMax : ILayer
    {
        private readonly int _inputs;
        private double[,] _lastRes;

        public SoftMax(int inputs)
        {
            this._inputs = inputs;
        }

        public object Predict(object rawData)
        {
            double[,] data = rawData as double[,];

            for (int batch = 0; batch < data.GetLength(0); ++batch)
            {
                double sum = 0;
                for (int neur = 0; neur < this._inputs; ++neur)
                {
                    data[batch, neur] = Math.Exp(data[batch, neur]);
                    sum += data[batch, neur];
                }
                for (int neur = 0; neur < this._inputs; ++neur)
                {
                    data[batch, neur] /= sum;
                }
            }

            this._lastRes = data;
            return data;
        }

        public object Back(object rawData)
        {
            double[] prewLoss = rawData as double[];
            
            double[] sums = new double[this._inputs];
            for (int neur = 0; neur < this._inputs; ++neur)
            {
                for (int batch = 0; batch < this._lastRes.GetLength(0); ++batch)
                {
                    sums[neur] += this._lastRes[batch, neur];
                }

                sums[neur] /= this._lastRes.GetLength(0);
            }

            for (int neurI = 0; neurI < this._inputs; ++neurI)
            {
                double tempSum = 0;
                for (int neurJ = 0; neurJ < this._inputs; ++neurJ)
                {
                    int sig = Convert.ToInt32(neurI == neurJ);
                    tempSum += sums[neurJ] * (1 - sums[neurI]);
                }

                prewLoss[neurI] *= tempSum;
            }
            
            return prewLoss;
        }
    }
    
    public class Layer : ILayer
    {
        private readonly IActivation _fun;
        private readonly IOptimizer _opt;
        private readonly double[,] _w;
        private readonly double[] _bias;
        private double[] _meanLastInput;
        private double[] _meanLastSum;
        public Layer (int nIns, int nOuts, IActivation fun, IOptimizer opt)
        {
            //  заполняем веса случайными значениями
            Random rnd = new Random();
            this._w = new double[nIns, nOuts];
            for (int y = 0; y < this._w.GetLength(0); y++)
            {
                for (int x = 0; x < nOuts; x++)
                {
                    _w[y, x] = rnd.Next(-1000, 1000) / 1000.0;
                }
            }
            this._bias = new double[nOuts];
            for (int neur = 0; neur < nOuts; ++neur) this._bias[neur] = rnd.Next(-100, 100) / 100.0;

            this._fun = fun;
            this._opt = opt;
        }

        public object Predict (object rawData)
        {
            double[,] data = rawData as double[,];
            if (data == null) throw new Exception("Wrong input");

            //  сохраним среднее значение каждого входа
            this._meanLastInput = new double[data.GetLength(1)];
            for (int input = 0; input < data.GetLength(1); ++input)
            {
                for (int batch = 0; batch < data.GetLength(0); ++batch)
                {
                    this._meanLastInput[input] += data[batch, input];
                }
                this._meanLastInput[input] /= data.GetLength(0);
            }


            //  значения сумматора - матричное умножение входа на весы
            double[,] res = Matrix.Mult(in data, in this._w);

            //  сохраним средние значения сумматоров каждого нейрона
            this._meanLastSum = new double[res.GetLength(1)];
            for (int neur = 0; neur < res.GetLength(1); ++neur)
            {
                this._meanLastSum[neur] = 0;
                for (int s = 0; s < res.GetLength(0); ++s) this._meanLastSum[neur] += res[s, neur] + this._bias[neur];
                this._meanLastSum[neur] /= res.GetLength(0);
            }

            //  предсказание - значение функции активации от значения сумматора
            for (int y = 0; y < res.GetLength(0); y++)
            {
                for (int x = 0; x < res.GetLength(1); x++)
                {
                    res[y, x] = this._fun.Value(res[y, x] + this._bias[x]);
                }
            }

            return res;
        }

        public object Back(object rawPrewLoss)
        {
            double[] prewLoss = rawPrewLoss as double[];
            if (prewLoss == null) throw new Exception("Wrong input");

            //  результат - усредненная ошибка каждого входа
            double[] res = new double[this._w.GetLength(0)];

            for (int neur = 0; neur < this._w.GetLength(1); neur++)
            {
                //  ошибка на одном из нейронов * производную функции активации
                double delta = prewLoss[neur] * this._fun.Derivative(this._meanLastSum[neur]);

                //  обновляем веса и составляем ответ
                for (int weight = 0; weight < this._w.GetLength(0); weight++)
                {
                    res[weight] += delta * this._w[weight, neur];
                    this._w[weight, neur] = this._opt.Optimize(this._w[weight, neur], delta * this._meanLastInput[weight]);
                }

                //  фиктивный вход в ответ не вносим, но вес обновляем
                this._bias[neur] = this._opt.Optimize(this._bias[neur], delta);
            }

            return res;
        }
    }
}
