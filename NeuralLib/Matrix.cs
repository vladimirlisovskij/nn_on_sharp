﻿using System;
using System.Collections.Generic;

namespace matrix
{
    public static class Matrix
    {
        public static double[,] Mult (in double[,] first, in double[,] second)
        {
            int fN = first.GetLength(0), fM = first.GetLength(1);
            int sN = second.GetLength(0), sM = second.GetLength(1);
            if (fM != sN) throw new Exception("Wrong dimentions");
            double[,] res = new double[fN, sM];
            for (int y = 0; y < sM; ++y)
            {
                for (int x = 0; x < fN; ++x)
                {
                    res[x, y] = 0;
                    for (int i = 0; i < fM; ++i)
                    {

                        res[x, y] += 
                            first[x, i] * 
                            second[i, y];
                    }
                }
            }
            return res;
        }
    }
}
