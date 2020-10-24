﻿using System;

namespace activation
{
    public interface IActivation
    {
        public double Value(double x);
        public double Derivative(double x);
    }

    public class Sigm : IActivation
    {
        public double Value(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double Derivative(double x)
        {
            x = Value(x);
            return (1 - x) * x;
        }
    }

    public class Trivial : IActivation
    {
        public double Value(double x) 
        { 
            return x; 
        }

        public double Derivative(double x)
        {
            return 1;
        }
    }

    public class Tanh : IActivation
    {
        public double Value(double x)
        {
            return Math.Tanh(x);
        }

        public double Derivative(double x)
        {
            return 1/Math.Pow(Math.Cosh(x),2);
        }
    }

    public class ReLU : IActivation
    {
        public double Value(double x)
        {
            return Math.Max(0,x);
        }

        public double Derivative(double x)
        {
            return Convert.ToDouble(x >= 0);
        }
    }
}