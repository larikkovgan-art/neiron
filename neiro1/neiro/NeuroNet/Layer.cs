using System;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using neiro.NeuroNet;

namespace neiro.NeuroNet
{
    abstract class Layer
    {
        //поля
        protected string name_Layer; //наименование слоя, использующееся для связи с одноимённым файлом памяти
        string pathDirWeights; //путь к каталогу, где находится файл синаптич. весов для нейрона
        string pathFileWeights; //путь к файлу синаптич. весов  для нейросети
        protected int numofneurons; //число нейронов текущего слоя
        protected int numofprevneurons; //число нейронов предыдущего слоя
        protected const double learningrate = 0.05d; //скорость обучения
        protected const double momentum = 0.050d; //момент инерции
        protected double[,] lastdeltaweights; //веса предыдущей итерации обучения
        protected Neuron[] neurons; //массив нейронов текущего слоя

        //свойства
        public double[] Data //передача входных данных на нейроны слоя и активация нейрона
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                {
                    neurons[i].Activator(value);
                }
            }
        }

        //Конструктор
        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer)
        {
            //nm_Layer для связи с одноимённям полю слоя файлом памяти (для метода WeightInitialize)
            numofneurons = non; //количество нейронов текущего слоя
            numofprevneurons = nopn; //количество нейронов предыдущего слоя
            neurons = new Neuron[non]; //определение массива нейронов
            name_Layer = nm_Layer; //наименование слоя, использующееся для связи с одноимённым файлом памяти
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\"; //путь к каталогу, где находится файл синаптич. весов для нейрона
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv"; //путь к файлу синаптич. весов  для нейросети

            lastdeltaweights = new double[non, nopn + 1];
            double[,] Weights; //временный массив синаптических весов текущего слоя

            if (File.Exists(pathFileWeights))
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights);
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (int i = 0; i < non; i++)
            {
                double[] tmp_weigths = new double[nopn + 1];
                for (int j = 0; j < nopn + 1; j++)
                {
                    tmp_weigths[j] = Weights[i, j];
                }
                neurons[i] = new Neuron(tmp_weigths, nt); //заполнение массива нейронами
            }
        }

        //метод работы с массивом синаптич. весов слоя
        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' }; //разделитель слоя ; и пробел
            string tmpStr; //временная строка для чтения
            string[] tmpStrWeights; //временный массив для строк
            double[,] weights = new double[numofneurons, numofprevneurons + 1]; //массив весов
            switch (mm)
            {
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path); //считывание строк текстового файла
                    string[] memory_element;

                    for (int i = 0; i < numofneurons; i++)
                    {

                        memory_element = tmpStrWeights[i].Split(delim); //разбивает строку по делителю

                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_element[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                case MemoryMode.SET:
                    path = pathFileWeights;
                    tmpStrWeights = new string[numofneurons]; //построение массива строк
                    if (!File.Exists(path)) //Если нет файла синапич. весов
                    {
                        MessageBox.Show("!!! В Н И М А Н И Е !!! \nФайл " + name_Layer + "_memory.csv синапрических весов" +
                            "\nНЕ НАЙДЕН!!! \nПосле нажатия кнопки OK автометически отсутствующий файл, и все значения весов будут сохранены",
                            "Внимание", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                    }
                    for (int i = 0; i < numofneurons; i++)
                    {
                        tmpStr = neurons[i].Weights[0].ToString();
                        for (int j = 1; j < numofprevneurons + 1; j++)
                        {
                            tmpStr += delim[0] + neurons[i].Weights[j].ToString();
                        }
                        tmpStrWeights[i] = tmpStr;
                    }
                    File.WriteAllLines(path, tmpStrWeights); //создаёт новый файл, записывает в него указанный массив и закрывает файл
                    //если файл существует - он переопределяется

                    break;

                case MemoryMode.INIT:
                    MessageBox.Show("!!! В Н И М А Н И Е !!! \nФайл " + name_Layer + "_memory.csv синапрических весов" +
                        "\nНЕ НАЙДЕН!!! \nПосле нажатия кнопки OK автометически проинициализируются веса и создатстся отсутствующий файл " +
                        "(произведённое обучение обнулится)",
                        "Внимание", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);

                    Random random = new Random();
                    double tmpRatio;
                    double tmpShift;
                    double[] tmpArr = new double[numofprevneurons + 1];
                    tmpStrWeights = new string[numofneurons];

                    for (int i = 0; i < numofneurons; i++)
                    {
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            tmpArr[j] = random.NextDouble();
                        }
                        tmpShift = tmpArr.Average();

                        double[] tmpArr2 = new double[numofprevneurons + 1];
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            tmpArr2[j] = Math.Pow(tmpArr[j], 2);
                        }

                        tmpRatio = Math.Sqrt(tmpArr2.Average() - Math.Pow(tmpShift, 2)); //станд отклонение
                        //double scale = 1.0 / Math.Sqrt(numofprevneurons);
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            tmpArr[j] = (tmpArr[j] - tmpShift) / tmpRatio /** scale*/;
                            weights[i, j] = tmpArr[j];
                        }
                    }

                    tmpStr = "";
                    for (int i = 0; i < numofneurons; i++)
                    {
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            tmpStr += weights[i, j].ToString() + delim[0];
                        }
                        tmpStr += "\n";
                    }
                    File.AppendAllText(path, tmpStr);
                    break;

                default:
                    break;
            }
            return weights;
        }

        abstract public void Recognize(Network net, Layer nextLayer); //для прямых обходов
        abstract public double[] BackwardPass(double[] stuff); //для обратных обходов
    }
}
