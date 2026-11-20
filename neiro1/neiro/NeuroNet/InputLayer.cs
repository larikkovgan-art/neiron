using System;
using System.IO;



namespace neiro.NeuroNet
{
    class InputLayer
    {
        //поля
        private double[,] trainset; //100 изображений в обучающей выборке
        private double[,] testset; //10 изображений в тестовой выборке

        //свойства
        public double[,] Trainset { get => trainset; }
        public double[,] Testset { get => testset; }

        //конструктор
        public InputLayer(NetworkMode nm)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory; //дирректорий
            string[] tmpArrStr; //временный массив строк
            string[] tmpStr; //временный вспомогательный массив элементов в строке

            switch (nm)
            {
                case NetworkMode.Train:
                    tmpArrStr = File.ReadAllLines(path + "train.txt"); //считывание из файла
                    trainset = new double[tmpArrStr.Length, 16]; //определение массива

                    for (int i = 0; i < tmpArrStr.Length; i++) //цикл перебора строк
                    {
                        tmpStr = tmpArrStr[i].Split(' '); //разбиение i-ой строки на массив
                        for (int j = 0; j < 16; j++)  //цикл заполнения i-ой строки обучения
                            trainset[i, j] = double.Parse(tmpStr[j]); //сроковое значение
                    }
                    break;
                case NetworkMode.Test:
                    tmpArrStr = File.ReadAllLines(path + "test.txt"); //считывание из файла тестовой выборки
                    testset = new double[tmpArrStr.Length, 16]; //определение массива тестовой выборки

                    for (int i = 0; i < tmpArrStr.Length; i++) //цикл перебора строк тестовой выборки
                    {
                        tmpStr = tmpArrStr[i].Split(' '); //разбиение i-ой строки на массив отдельныъ символов
                        for (int j = 0; j < 16; j++)  //цикл заполнения i-ой строки тестовой выборки
                            testset[i, j] = double.Parse(tmpStr[j]); //сроковое значение числа преобразуется в само число двойной точности
                    }
                    break;
                
            }
        }

        //перетасовка строк массива методом Фишера-Йетса. fisher-yates method
        public void Shuffling_Array_Rows(double[,] arr)
        {
            Random rnd = new Random();
            int rows = arr.GetLength(0);
            int cols = arr.GetLength(1);

            for (int i = rows - 1; i > 0; i--)
            {
                int j = rnd.Next(i + 1); // случайный индекс от 0 до i
                                         // обмен строк i и j
                for (int k = 0; k < cols; k++)
                {
                    double tmp = arr[i, k];
                    arr[i, k] = arr[j, k];
                    arr[j, k] = tmp;
                }
            }
        }
    }
}
    