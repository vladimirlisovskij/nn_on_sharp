﻿using loss;
using neuron_list;

namespace Wraper
{
    public abstract class AWraper
    {
        public readonly AWraper Next;
        public readonly ILayer Layer;
        public readonly ILoss Loss;

        public AWraper(ILayer layer, AWraper next, ILoss loss)
        {
            this.Layer = layer;
            this.Next = next;
            this.Loss = loss;
        }

        public abstract object Feed (object rawX, object rawY);
        public abstract object Predict (object rawX);
    }
}
