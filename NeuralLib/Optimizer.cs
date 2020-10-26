﻿namespace optimizer
{
    public interface IOptimizer
    {
        double Optimize(double val, double dev);
    }

    public class SGD : IOptimizer
    {
        private readonly double _k;
        public SGD(double k)
        {
            this._k = k;
        }

        public double Optimize(double val, double dev)
        {
            return val - _k * dev;
        }
    }
}
