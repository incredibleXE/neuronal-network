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
        resize_rows(list.size());
        resize_cols(1);
        for (unsigned long i = 0; i < list.size(); ++i) {
            v_mat[i][0] = list[i];
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
            // matrizen lassen sich nicht subtrahieren da sie nicht vom gleichen Typ sind
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

        // Berechnung: pro Feld: Summe(a(ij)*b(jk))
        return_vector.resize(rows,1);
        cout << "Reihen: " << rows << "\n";
        // alle Reihen von a
        //
        // TODO: FEHLER
        //
        for (int j = 0; j < a.count_cols(); ++j) {
            return_vector.v_mat[j][0] = 0;
            for (int k = 0; k < a.count_rows(); ++k) {
                cout << return_vector.v_mat[j][0]  << " + "
                     << "(" << a.v_mat[k][j] << "*" << b.v_mat[k][0] << ")";

                return_vector.v_mat[j][0] += (a.v_mat[k][j] * b.v_mat[k][0]);

                cout << " = " << return_vector.v_mat[j][0] << "\n";
            }
            cout << "------" << endl;
        }

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
        v_weight_input_to_hidden.resize(i_input_nodes,i_hidden_nodes);
        for(int i = 0; i < i_input_nodes; ++i) {
            for(int a = 0; a < i_hidden_nodes; ++a) {
                v_weight_input_to_hidden.v_mat[i][a] = random_double(-1.f,1.f);
            }
        }

        // hidden zu output layer Gewichtungen
        v_weight_hidden_to_output.resize(i_hidden_nodes,i_output_nodes);
        for(int i = 0; i < i_hidden_nodes; ++i) {
            for(int a = 0; a < i_output_nodes; ++a) {
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

        Matrix2d targets;
        targets.set(target_list);

        //cout << "Forward pass \n";
        // === Forward pass === //
        // hidden schicht
        shared_ptr<Matrix2d> hidden_inputs = v_weight_input_to_hidden * inputs;

        //cout << "gewichte: \n" << v_weight_input_to_hidden << "inputs\n" << inputs;
        //cout << "hidden input" << *hidden_inputs;
        shared_ptr<Matrix2d> hidden_outputs = activation_function_mat(*hidden_inputs);

        // output schicht
        cout << *v_weight_hidden_to_output.T();
        cout << *hidden_outputs;
        shared_ptr<Matrix2d> output_inputs = *v_weight_hidden_to_output.T() * *hidden_outputs;

        shared_ptr<Matrix2d> output_outputs = output_inputs;

        return;
        cout << "after forward pass: \n" << *output_outputs;

        cout << "\n\nback propagation start\n";
        // === back propagation == //
        // output schicht
        shared_ptr<Matrix2d> output_errors = *(targets - *output_outputs) * 1.f;

        //cout << *output_errors;

        // hidden schicht
        // TODO: im Beispiel output_errors * v_weight_hidden_to_output, geht aber aufgrund von Matrixregeln nicht. Richtig?
        shared_ptr<Matrix2d> hidden_errors = *(*output_errors).T() * v_weight_hidden_to_output;
        //cout << hidden_errors;

        //shared_ptr<Matrix> output_errors_T = T(*output_errors);
        //shared_ptr<Matrix> hidden_errors = multi(*output_errors_T,v_weight_hidden_to_output);

        shared_ptr<Matrix2d> hidden_errors_T = (*hidden_errors).T();
        shared_ptr<Matrix2d> subt_hidden_outputs = *hidden_outputs - -1.f;
        shared_ptr<Matrix2d> multi_hidden_errors_hidden_outputs = (*hidden_errors * *hidden_outputs);
        shared_ptr<Matrix2d> hidden_grad = *multi_hidden_errors_hidden_outputs * *(*subt_hidden_outputs).T();

        cout << "hidden grad" << *hidden_grad;
        return;
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



    return 0;
}