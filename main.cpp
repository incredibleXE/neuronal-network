#include <iostream>
#include <vector>
#include <math.h>
#include <stdexcept>
// für shared_ptr
#include <memory>

using namespace std;

//typedef vector<vector<double>> Matrix;

class Matrix2d {
public:
    vector<vector<double>> v_mat;

    Matrix2d() {
        v_mat = {{0}};
    }

    Matrix2d(vector<vector<double>> mat) {
        v_mat = mat;
    }

    void resize_rows(unsigned long i) {
        v_mat.resize(i);
    }

    void set(vector<double> list) {
        v_mat = {{0}};
        resize_rows(1);
        resize_cols(list.size());
        for (unsigned long i = 0; i < list.size(); ++i) {
            v_mat[0][i] = list[i];
        }
    }

    void resize_cols(unsigned long i) {
        for (int j = 0; j < v_mat.size(); ++j) {
            v_mat[j].resize(i);
        }
    }

    void resize(unsigned long rows, unsigned long cols) {
        resize_rows(rows);
        resize_cols(cols);
    }

    /**
    * Zählt die Reihen einer Matrize
    *
    * @return Anzahl der Reihen der Matrize
    */
    unsigned long count_rows() {
        return v_mat.size();
    }

    /**
     * Zählt die Spalten einer Matrize
     *
     * @param mat zu untersuchende Matrize
     * @return Anzahl der Spalten der Matrize
     */
    unsigned long count_cols() {
        if(v_mat.size() == 0) {
            return 0;
        }
        return v_mat[0].size();
    }

    // warum ist friend hier wichtig?
    friend ostream& operator <<(ostream& os, Matrix2d& dt)
    {
        os << typeid(dt).name() << " " << dt.count_rows() << "x" << dt.count_cols() << endl;
        for (int i = 0; i < dt.count_rows(); ++i) {
            for (int j = 0; j < dt.count_cols(); ++j) {
                os << dt.v_mat[i][j] << "|";
            }
            os << endl;
        }

        return os;
    }

    /**
     * Transkription einer Matrix Matrize
     *
     * @param input_list
     * @return input_list.T
     */
    shared_ptr<Matrix2d> T() {

        Matrix2d return_mat;

        // return_mat zusammenbauen
        return_mat.resize(count_cols(),count_rows());

        for(int i=0; i < count_rows(); ++i) {
            for (int j = 0; j < count_cols(); ++j) {
                return_mat.v_mat[j][i] = v_mat[i][j];
            }
        }

        return std::make_shared<Matrix2d>(return_mat);
    }

    /**
    * Subtrahiert Matrize b von double a
    * (baut Matrize für a auf und ruft subt() mit zwei Matrizen auf)
    *
    * @param a Minuend double wert
     * @param b Subtrahend Matrize
    * @return Subtraktionsmatrize
    */
    shared_ptr<Matrix2d>operator-(float b) {
        Matrix2d a = *this;
        unsigned long cols = a.count_cols();
        unsigned long rows = a.count_rows();

        Matrix2d tmp_return_mat;

        tmp_return_mat.resize(rows,cols);
        for(int i=0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                tmp_return_mat.v_mat[i][j] = b;
            }
        }

        return a-tmp_return_mat;
    }

    shared_ptr<Matrix2d>operator-(Matrix2d& b) {
        Matrix2d a = *this;
        unsigned long cols_a = a.count_cols(), cols_b = b.count_cols();
        unsigned long rows_a = a.count_rows(), rows_b = b.count_rows();

        Matrix2d return_mat;

        if(cols_a != cols_b || rows_b != rows_a) {
            // falls eine der Matrizen 1x1 ist, wird wert genommen und subtrahiert
            if(cols_a == 1 && rows_a == 1) {
                return b - a.v_mat[0][0];
            }

            if(cols_b == 1 && rows_b == 1) {
                return a - b.v_mat[0][0];
            }
            throw std::invalid_argument("Matrix a und b sind nicht vom gleichen Typ");
        }

        return_mat.resize(rows_a,cols_a);
        for(int i=0; i < rows_a; ++i) {
            for (int j = 0; j < cols_a; ++j) {
                return_mat.v_mat[i][j] = a.v_mat[i][j] - b.v_mat[i][j];
            }
        }

        return std::make_shared<Matrix2d>(return_mat);
    }

    shared_ptr<Matrix2d>operator*(Matrix2d& b) {
        Matrix2d a = *this;
        Matrix2d return_vector;

        // falls ein vector von beiden leer ist => gib einen leeren vector zurück
        if(a.count_rows() == 0 or b.count_rows() == 0) {
            throw std::invalid_argument("Mindestens eine Matrize ist leer.");
            return std::make_shared<Matrix2d>(return_vector);
        }

        if(a.count_cols() != b.count_rows()) {
            // Abbruch, weil Spaltenanzahl von a != Zeilenanzahl von b (// https://de.wikipedia.org/wiki/Matrizenmultiplikation) => hinreichende Bedingung
            cout << "A:" << endl << a << endl << "B:" << endl << b;
            throw std::invalid_argument("Spaltenanzahl von a != Zeilenanzahl b. Mathematisch problematisch.");
        }

        // https://de.wikipedia.org/wiki/Matrizenmultiplikation
        unsigned long rows = a.count_rows();
        unsigned long cols = b.count_cols();

        //cout << "A:" << endl << a << endl << "B:" << endl << b;
        // Berechnung: pro Feld: Summe(a(ij)*b(jk))
        return_vector.resize(rows,cols);
        //cout << "Reihen: " << rows << endl << "Spalten: "<< cols << "\n";

        for (int j = 0; j < return_vector.count_rows(); ++j) {
            for (int k = 0; k < return_vector.count_cols(); ++k) {
                for (int i = 0; i < a.count_cols(); ++i) {
                    double a_val = a.v_mat[j][i];
                    double b_val = b.v_mat[i][k];

                    //cout << return_vector.v_mat[j][k] << " + (" << a_val << " * " << b_val << ")";

                    return_vector.v_mat[j][k] += (a_val * b_val);

                    //cout << " = "<< return_vector.v_mat[j][k] << "\n";
                }
                //cout << "---\n";
            }
        }
        //cout << "Return: " << return_vector;

        return std::make_shared<Matrix2d>(return_vector);
    }

    shared_ptr<Matrix2d>operator*(float b) {
        Matrix2d a = *this;
        unsigned long cols = a.count_cols();
        unsigned long rows = a.count_rows();

        Matrix2d return_mat;

        return_mat.resize(rows,cols);
        for(int i=0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                return_mat.v_mat[i][j] = v_mat[i][j] * b;
                //cout << i << "x" << j << ": " << v_mat[i][j] << " * " << b << " = " << return_mat.v_mat[i][j] << endl;
            }
        }

        return make_shared<Matrix2d>(return_mat);
    }
};

/**
 * Neuronales Netz
 */
class NeuronalNetwork {
public:
    /**
     * b_debug wird vom Konstruktor gesteuert
     *  wenn true => train-function gibt JEDEN Matrix auf der Konsole aus.
     * deep_debug ist nur zum testen gedacht:
     *  wenn true => die Berechnungsfunktionen geben die Eingabe- und Ausgabematrizen auf der Konsole aus
     */
    bool b_debug = false, b_deep_debug = false;

    Matrix2d v_weight_input_to_hidden,v_weight_hidden_to_output;
    double f_learning_rate;

    /**
     * Constructor
     *
     * Erstellt zufällige Gewichtungen für alle Knotenverbindungen.
     *
     * @param i_input_nodes Anzahl der Neuronen auf der Input Schicht
     * @param i_hidden_nodes Anzahl der Neuronen auf der Hidden Schicht
     * @param i_output_nodes Anzahl der Neuronen auf der Output Schicht
     * @param learning_rate
     * @param debug
     */
    NeuronalNetwork(unsigned long i_input_nodes, unsigned long i_hidden_nodes, unsigned long i_output_nodes, double learning_rate, bool debug = false) {
        b_debug = debug;
        b_deep_debug = true;
        f_learning_rate = learning_rate;

        // input zu hidden layer Gewichtungen
        v_weight_input_to_hidden.resize(i_hidden_nodes,i_input_nodes);
        for(int i = 0; i < i_hidden_nodes; ++i) {
            for(int a = 0; a < i_input_nodes; ++a) {
                v_weight_input_to_hidden.v_mat[i][a] = random_double(-1.f,1.f);
            }
        }

        // hidden zu output layer Gewichtungen
        v_weight_hidden_to_output.resize(i_output_nodes,i_hidden_nodes);
        for(int i = 0; i < i_output_nodes; ++i) {
            for(int a = 0; a < i_hidden_nodes; ++a) {
                v_weight_hidden_to_output.v_mat[i][a] = random_double(-1.f,1.f);
            }
        }
    }

    /**
     * Trainiert das Neuronale Netz
     *
     * @param input_list vector<double> aller Inputs.
     * @param target_list vector<double> aller Outputs.
     */
    void train(vector<double> input_list,vector<double> target_list) {
        cout << "starting train\n" << "--------------" << endl;

        // Transponieren der vectoren, wie geht das besser?
        Matrix2d inputs;
        inputs.set(input_list);
        inputs = *inputs.T();

        Matrix2d targets;
        targets.set(target_list);
        targets = *targets.T();

        //cout << "Forward pass \n";
        // === Forward pass === //
        // hidden schicht
        shared_ptr<Matrix2d> hidden_inputs = v_weight_input_to_hidden * inputs;

        //cout << "gewichte: \n" << v_weight_input_to_hidden << "inputs\n" << inputs;
        //cout << "hidden input" << *hidden_inputs;
        shared_ptr<Matrix2d> hidden_outputs = activation_function_mat(*hidden_inputs);

        // output schicht
        //cout << *v_weight_hidden_to_output.T();
        //cout << *hidden_outputs;
        shared_ptr<Matrix2d> output_inputs = v_weight_hidden_to_output * *hidden_outputs;

        shared_ptr<Matrix2d> output_outputs = output_inputs;

        cout << "after forward pass: output from output-layer \n" << *output_outputs;

        cout << "\n\nback propagation start\n\n";
        // === back propagation == //
        // output schicht
        shared_ptr<Matrix2d> subt_targets_output_outputs = targets - *output_outputs;
        //cout << *subt_targets_output_outputs;
        shared_ptr<Matrix2d> output_errors = *subt_targets_output_outputs * 1.f;

        //cout << *output_errors;

        // hidden schicht
        shared_ptr<Matrix2d> hidden_errors = *output_errors * v_weight_hidden_to_output;
        //cout << *hidden_errors;

        //shared_ptr<Matrix> output_errors_T = T(*output_errors);
        //shared_ptr<Matrix> hidden_errors = multi(*output_errors_T,v_weight_hidden_to_output);

        shared_ptr<Matrix2d> hidden_errors_T = (*hidden_errors).T();
        // müsste 1 - hidden_outputs sein, aber das kann ich nicht :D => (hidden_outputs - 1) * (-1)
        shared_ptr<Matrix2d> subt_hidden_outputs = *(*hidden_outputs - 1.f) * (-1.f);
        shared_ptr<Matrix2d> multi_hidden_errors_hidden_outputs = (*hidden_errors_T * *hidden_outputs);

        // TODO: warum muss ich hier Transponieren?
        shared_ptr<Matrix2d> hidden_grad = *(*multi_hidden_errors_hidden_outputs).T() * *subt_hidden_outputs;

        v_weight_hidden_to_output = *(*(*output_errors * *(*hidden_outputs).T()) * f_learning_rate);
        v_weight_input_to_hidden = *(*(*hidden_grad * *(inputs).T()) * f_learning_rate);

        cout << "after back propagation: hidden grad\n" << *hidden_grad;
        return;
    }

    Matrix2d run(vector<double> input_list) {
        cout << "\n\n\n" << "================\n"
                         << "   start run    \n"
                         << "================\n";

        // Transponieren der vectoren, wie geht das besser?
        Matrix2d inputs;
        inputs.set(input_list);
        inputs = *inputs.T();

        // hidden layer
        shared_ptr<Matrix2d> hidden_inputs = v_weight_input_to_hidden * inputs;
        return v_weight_input_to_hidden;
        shared_ptr<Matrix2d> hidden_outputs = activation_function_mat(*hidden_inputs);

        // output layer
        shared_ptr<Matrix2d> output_outputs = v_weight_hidden_to_output * *hidden_outputs;

        return *output_outputs;
    }

    /**
     * sigmoide Aktivierungsfunktion
     *
     * @param x double Eingabewert
     * @return double Rückgabewert der Funktion
     */
    double activation_function(double x) {
        return 1/(1+exp(x));
    }

    /**
     * Ruft für jeden Wert in der übergebenen Matrize die @see activation_function() auf und speichert den Wert in Rückgabematrize.
     *
     * @param mat Matrix Eingabematrize
     * @return Matrix Rückgabematrize
     */
    shared_ptr<Matrix2d> activation_function_mat(Matrix2d mat) {
        Matrix2d return_mat;
        unsigned long rows = mat.count_rows();
        unsigned long cols = mat.count_cols();

        return_mat.resize(rows,cols);
        for(int i = 0; i < rows; ++i) {
            for(int j=0;j < cols; ++j) {
                return_mat.v_mat[i][j] = activation_function(mat.v_mat[i][j]);
            }
        }

        return std::make_shared<Matrix2d>(return_mat);
    }

//private:
    /**
     * Gibt eine Zufallszahl im gegebenen Intervall
     * TODO: wirklich random machen!
     *
     * @param a Minimalwert
     * @param b Maximalwert
     * @return double Zufallswert zwischen den Parametern.
     */
    double random_double(double a, double b) {
        double random = ((double) rand()) / (double) RAND_MAX;
        double diff = b - a;
        double r = random * diff;
        return a + r;
    }
};

int main() {
    NeuronalNetwork nn = {3,3,1,0.1f, true};

    nn.train({1,2,3},{4,5,6});

    Matrix2d mat = nn.run({1,2,3});
    cout << mat;


    return 0;
}