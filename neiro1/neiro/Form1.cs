using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using neiro.NeuroNet;

namespace neiro
{
    public partial class Form1 : Form
    {
        private double[] inputPixels; //массив входных данных
        private Network network;   //обьяснение нейросети
        private object chart_Eavr;

        public Form1()
        {
            InitializeComponent();

            inputPixels = new double[15];
            network = new Network();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (((Button)sender).BackColor == Color.White)
            {
                ((Button)sender).BackColor = Color.Black;
                inputPixels[((Button)sender).TabIndex] = 1d;
            }
            else
            {
                ((Button)sender).BackColor = Color.White;
                inputPixels[((Button)sender).TabIndex] = 0d;
            }
        }

        private void buttonSaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for(int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr);
        }

        private void buttonSaveTestSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "test.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr);
        }

        private void button_recognize_Click(object sender, EventArgs e)
        {
            network.ForwardPass(inputPixels);
            labelOutput.Text = network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            labelPropobiloty.Text = (100 * network.Fact.Max()).ToString("0.00") + " %";
        }

        private void buttonTraining_Click(object sender, EventArgs e)
        {
            network.Train(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart1.Series[0].Points.AddY(network.E_error_avr[i]);
            }
            MessageBox.Show("Обучение успешно завершено.", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}       
