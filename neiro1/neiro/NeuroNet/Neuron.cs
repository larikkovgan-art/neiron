using static System.Math;

namespace neiro.NeuroNet
{
    class Neuron
    {
        // Поля
        private NeuronType type; // Тип нейрона
        private double[] weights; // Его веса
        private double[] inputs; // Его входы
        private double output; // Его выход
        private double derivative; // Производная

        // Свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output => output;
        public double Derivative => derivative;

        // Конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }

        // Метод активации нейрона
        public void Activator(double[] i)
        {
            inputs = i; // Передача вектора входного сигнала в массив входных данных нейрона

            // Защита от System.IndexOutOfRange: если размер массива весов меньше, 
            // чем количество входов + 1 (для смещения weights[0]), автоматически расширяем массив.
            if (weights == null || weights.Length < inputs.Length + 1)
            {
                System.Array.Resize(ref weights, inputs.Length + 1);
            }

            double sum = weights[0]; // Аффинное преобразование через смещение (нулевой вес - порог)
            for (int j = 0; j < inputs.Length; j++) // Цикл вычисления индуцированного поля нейрона
            {
                sum += inputs[j] * weights[j + 1]; // Линейные преобразования входных сигналов
            }

            switch (type)
            {
                case NeuronType.Hidden:  // Для нейронов скрытого слоя
                case NeuronType.Output: // Для нейронов выходного слоя
                    output = Logistic(sum);
                    derivative = LogisticDerivative(output);
                    break;
            }
        }

        // Логистическая функция активации (сигмоид)
        private double Logistic(double sum)
        {
            return 1.0 / (1.0 + Exp(-sum));
        }

        // Производная логистической функции (принимает уже посчитанный выход)
        private double LogisticDerivative(double outVal)
        {
            return outVal * (1.0 - outVal); // Исправлен синтаксис (добавлен знак умножения)
        }
    }
}