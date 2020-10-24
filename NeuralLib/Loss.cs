﻿using System;


namespace loss
{
    public interface ILoss
    {
        public object Value(object predicted, object real);
        public object Derivative(object predicted, object real);
    }

    public class MSE : ILoss
    {
        public object Value(object predicted, object real)
        {
            return Math.Pow((double)predicted - (double)real, 2) / 2.0;
        }

        public object Derivative(object predicted, object real)
        {
            return (double)predicted - (double)real;
        }
    }

    public class CrossEntopy : ILoss
    {
        public object Value(object rawPredicted, object real)
        {
            double[] predicted = rawPredicted as double[];
            return -Math.Log(predicted[(int)real]);
        }

        public object Derivative(object rawPredicted, object real)
        {
            double[] predicted = rawPredicted as double[];
            double[] res = new double[predicted.GetLength(0)];
            for (int i = 0; i < res.GetLength(0); ++i)
            {
                if (i == (int) real)
                {
                    res[i] = -1 / predicted[i];
                }
                else
                {
                    res[i] = 0;
                }
            }
            return res;
        }
    }
}

