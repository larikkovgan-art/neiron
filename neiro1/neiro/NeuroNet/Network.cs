namespace neiro.NeuroNet
{
    class Network
    {
        //все слои сети
        private InputLayer input_layer = null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(71, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(30, 70, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 30, NeuronType.Output, nameof(output_layer));

        private double[] fact = new double[10]; //массив фактического входа сети
        private double[] e_error_avr; //среднее значение энергии ошибки эпохи обучения

        //свойство
        public double[] Fact { get => fact; } //массив фактического схода сети

        //среднее значение энергии ошибки эпохи обучения
        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }

        //конструктор
        public Network() { }

        //прямой проход сети
        public void ForwardPass(double[] netInput)
        {
            hidden_layer1.Data = netInput;
            hidden_layer1.Recognize(null, hidden_layer2);
            hidden_layer2.Recognize(null, output_layer);
            output_layer.Recognize(this, null);
        }

        //непосредственно обучение
        public void Train(Network net)
        {
            input_layer = new InputLayer(NetworkMode.Train); //инициализация входного слоя для формирования множества обучающих образов

            int epoches = 8; //количество эпох обучения
            double tmpSumError; //временная переменная суммы ошщибок
            double[] errors; //вектор (массив) сигнала ошибки выходного слоя
            double[] temp_gsums1; //вектор градиента 1-го скрытого слоя
            double[] temp_gsums2; //вектор градиента 2-го скрытого слоя

            e_error_avr = new double[epoches];
            for(int k = 0; k < epoches; k++) //перебор эпох обучения
            {
                e_error_avr[k] = 0; //вначале каждой эпохи обучения значение средней энергии ошибки эпохи обнуляется
                input_layer.Shuffling_Array_Rows(input_layer.Trainset); //перетасовка обучающей выборки методом
                for(int i = 0; i<input_layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15]; //обучающий образ
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = input_layer.Trainset[i, j + 1];

                    //прямой проход
                    ForwardPass(tmpTrain); //прямой проход обучающего образа

                    //вычисление ошибки по итерации
                    tmpSumError = 0; //для каждого обучающего образа среднее значение ошибки этого образа обнуляется
                    errors = new double[fact.Length]; //переопределение массива сигнала ошибки выходного слоя
                    for(int x = 0; x < errors.Length; x++)
                    {
                        if (x == input_layer.Trainset[i, 0]) //если номер выходного нейрона совпадает с желаемым
                            errors[x] = 1.0 - fact[x];
                        else
                            errors[x] = -fact[x]; //errors[x] = 0.0 - net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; //суммарное значение энергии ошибки k-ой эпохи

                    // обратный проход и коррекция весов!!!
                    temp_gsums2 = output_layer.BackwardPass(errors);
                    temp_gsums1 = hidden_layer2.BackwardPass(temp_gsums2);
                    hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= input_layer.Trainset.GetLength(0); //среднее значение энергии ошибки одной эпохи
            }

            input_layer = null; //обнуление (уборка) входного слоя

            //запись скорректированных весов в "память"
            hidden_layer1.WeightInitialize(MemoryMode.SET, nameof(hidden_layer1) + "_memory.csv");
            hidden_layer2.WeightInitialize(MemoryMode.SET, nameof(hidden_layer2) + "_memory.csv");
            output_layer.WeightInitialize(MemoryMode.SET, nameof(output_layer) + "_memory.csv");
        }
    }
}
