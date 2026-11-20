namespace neiro.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_Layer) :
            base(non, nopn, nt, nm_Layer)
        { }

        // прямой проход 
        public override void Recognize(Network net, Layer nextlayer)
        {
            double[] hidden_out = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
                hidden_out[i] = neurons[i].Output;
            nextlayer.Data = hidden_out;                     // передача выходного сигнала на вход след слоя
        }

        // обратный проход
        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];

            // цикл вычисления градиентной суммы j-го нейрона текущего слоя
            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                // Бежим по нейронам текущего слоя (их количество gr_sums.Length, либо numofneurons, но безопаснее использовать gr_sums.Length или min)
                int nextNeuronsCount = gr_sums.Length;
                for (int k = 0; k < nextNeuronsCount; k++)
                {
                    // Защита от выхода за границы, если в текущем слое нейронов больше, чем в следующем
                    if (k < neurons.Length)
                    {
                        sum += neurons[k].Weights[j + 1] * neurons[k].Derivative * gr_sums[k];
                    }
                }
                gr_sum[j] = sum;
            }

            // цикл коррекции синаптических весов
            for (int i = 0; i < numofneurons; i++)
            {
                // Если индекс нейрона i выходит за пределы размера gr_sums, пропускаем коррекцию весов для этого нейрона,
                // чтобы избежать IndexOutOfRangeException
                if (i >= gr_sums.Length) continue;

                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0) // если порог
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * neurons[i].Derivative * gr_sums[i];
                    else
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * neurons[i].Derivative * gr_sums[i];

                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw;
                }
            }

            return gr_sum;
        }
    }
}