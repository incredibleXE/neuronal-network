#include <iostream>
#include <vector>
#include <math.h>
// für shared_ptr
#include <memory>

using namespace std;

/**
 * Neuronales Netz
 */
class NeuronalNetwork {
public:
    /**
     * b_debug wird vom Konstruktor gesteuert
     *  wenn true => train-function gibt JEDEN vector<vector<double>> auf der Konsole aus.
     * deep_debug ist nur zum testen gedacht:
     *  wenn true => die Berechnungsfunktionen geben die Eingabe- und Ausgabematrizen auf der Konsole aus
     */
    bool b_debug = false, deep_debug = false;

    vector<vector<double>> v_weight_input_to_hidden,v_weight_hidden_to_output;
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
        deep_debug = true;
        f_learning_rate = learning_rate;

        // input zu hidden layer Gewichtungen
        v_weight_input_to_hidden.resize(i_input_nodes);
        for(int i = 0; i < i_input_nodes; ++i) {
            v_weight_input_to_hidden[i].resize(i_hidden_nodes);
            for(int a = 0; a < i_hidden_nodes; ++a) {
                v_weight_input_to_hidden[i][a] = random_double(-1.f,1.f);
            }
        }
        output_matrix_to_console(v_weight_input_to_hidden,"v_weight_input_to_hidden");

        // hidden zu output layer Gewichtungen
        v_weight_hidden_to_output.resize(i_hidden_nodes);
        for(int i = 0; i < i_hidden_nodes; ++i) {
            v_weight_hidden_to_output[i].resize(i_output_nodes);
            for(int a = 0; a < i_output_nodes; ++a) {
                v_weight_hidden_to_output[i][a] = random_double(-1.f,1.f);
            }
        }
        output_matrix_to_console(v_weight_hidden_to_output,"v_weight_hidden_to_output");
    }

    /**
     * Trainiert das Neuronale Netz
     *
     * @param input_list vector<double> aller Inputs.
     * @param target_list vector<double> aller Outputs.
     */
    void train(vector<double> input_list,vector<double> target_list) {
        cout << "starting train\n" << "--------------" << endl;

        // Transponieren der vectoren
        vector<vector<double>> inputs = T(input_list);
        output_matrix_to_console(inputs,"train->inputs");
        vector<vector<double>> targets = T(target_list);
        output_matrix_to_console(targets,"train->targets");

        // === Forward pass === //
        // hidden schicht
        vector<vector<double>> hidden_inputs = multi(v_weight_input_to_hidden,inputs);
        output_matrix_to_console(hidden_inputs,"train->hidden_inputs");
        vector<vector<double>> hidden_outputs = activation_function_mat(hidden_inputs);
        output_matrix_to_console(hidden_outputs,"train->hidden_outputs");

        // output schicht
        vector<vector<double>> output_inputs = multi(v_weight_hidden_to_output,hidden_outputs);
        output_matrix_to_console(output_inputs,"train->output_inputs");
        vector<vector<double>> output_outputs = output_inputs;
        output_matrix_to_console(output_outputs,"train->output_outputs (= output_input)");

        /** @fgr
         *  Bis hier hin komme ich, dann geht es nicht weiter.
         *  meine Gedanken dazu:
         *  Ich glaube ich versuche bei den Funktionen (multi, subt, etc...) Variablen zurückzugeben, die zur Funktion gehören.
         *  Bis hier hin hat das funktioniert, da die Matrizen stets in der nächsten Zeile verwendet werden. Nun wird ab hier aber auf Ressourcen
         *  zugegriffen, die eventuell bereits wieder überschrieben wurden und es kommt zum Fehler.
         *
         *  Mein Ansatz war die Rückgabetypen um shared_ptr zu erweitern. Aber da bin ich leider an meine Grenzen gestoßen.
         *  Wie geht es nun weiter?
         */

        // === back propagation == //
        // output schicht
        vector<vector<double>> output_errors = multi(subt(targets,output_outputs),1.f);
        output_matrix_to_console(output_errors,"train->output_errors");

        // hidden schicht
        // TODO: im Beispiel output_errors * v_weight_hidden_to_output, geht aber aufgrund von Matrixregeln nicht. Richtig?
        vector<vector<double>> hidden_errors = multi(T(output_errors),v_weight_hidden_to_output);
        output_matrix_to_console(hidden_errors,"train->hidden_errors");

        vector<vector<double>> hidden_grad = multi(multi(T(hidden_errors),hidden_outputs),subt(-1.f,hidden_outputs));
        output_matrix_to_console(hidden_grad,"train->hidden_grad");

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
     * @param mat vector<vector<double>> Eingabematrize
     * @return vector<vector<double>> Rückgabematrize
     */
    vector<vector<double>> activation_function_mat(vector<vector<double>> mat) {
        vector<vector<double>> return_mat;
        unsigned long rows = count_rows(mat);
        unsigned long cols = count_cols(mat);

        return_mat.resize(rows);
        for(int i = 0; i < rows; ++i) {
            return_mat[i].resize(cols);
            for(int j=0;j < cols; ++j) {
                return_mat[i][j] = activation_function(mat[i][j]);
            }
        }

        if(deep_debug) {
            output_matrix_to_console(return_mat, "activation_function_mat->return_mat");
        }

        return return_mat;
    }

    void output_matrix_to_console(vector<vector<double>> mat, string name = "Unknown") {
        if(b_debug) {
            cout << "--------------------------------------------" << endl
                 << "Print "<< name << "-Matrix " << count_rows(mat) << "x" << count_cols(mat) << endl
                 << "--------------------------------------------" << endl;
            for (int i = 0; i < count_rows(mat); ++i) {
                for (int j = 0; j < count_cols(mat); ++j) {
                    cout << mat[i][j] << " | ";
                }
                cout << endl;
            }
        }
    }
//private:
    /**
     * Gibt eine Zufallszahl im gegebenen Intervall
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

    /**
     * Transkription einer vector<double> Matrize
     *
     * @param input_list
     * @return input_list.T
     */
    vector<vector<double>> T(vector<double> input_list) {
        vector<vector<double>> outputs;
        outputs.resize(input_list.size());
        for(int i=0;i < input_list.size();++i) {
            outputs[i].resize(1);
            outputs[i][0] = input_list[i];
        }

        return outputs;
    }

    /**
     * Transkription einer vector<vector<double>> Matrize
     *
     * @param input_list
     * @return input_list.T
     */
    vector<vector<double>> T(vector<vector<double>> mat) {
        unsigned long rows = count_rows(mat);
        unsigned long cols = count_cols(mat);

        vector<vector<double>> return_mat = {{0}};

        return_mat.resize(cols);
        for(int i=0; i < rows; ++i) {
            return_mat[i].resize(rows);
            for (int j = 0; j < cols; ++j) {
                return_mat[j][i] = mat[i][j];
            }
        }

        return return_mat;
    }

    /**
     * Multipliziert zwei Matrizen
     *
     * @param a erste Matrize
     * @param b zweite Matrize
     * @return berechnete Matrize
     */
    vector<vector<double>> multi(vector<vector<double>> a, vector<vector<double>> b) {
        if(deep_debug) {
            output_matrix_to_console(a, "multi->a");
            output_matrix_to_console(b, "multi->b");
        }

        // @fgr: ist {{}} nötig?
        vector<vector<double>> return_vector = {{0}};

        // falls ein vector von beiden leer ist => gib einen leeren vector zurück
        if(a.size() == 0 or b.size() == 0) {
            return return_vector;
        }

        if(count_cols(a) != count_rows(b)) {
            // Abbruch, weil Spaltenanzahl von a != Zeilenanzahl von b (// https://de.wikipedia.org/wiki/Matrizenmultiplikation) => hinreichende Bedingung
            return return_vector;
        }

        // https://de.wikipedia.org/wiki/Matrizenmultiplikation
        unsigned long rows = count_rows(a);
        unsigned long cols = count_cols(b);

        // Berechnung: pro Feld: Summe(a(ij)*b(jk))
        return_vector.resize(rows);
        for(int i=0; i < rows; ++i) {
            return_vector[i].resize(cols);
            for(int j=0; j < cols; ++j) {
                return_vector[i][j] = 0;
                for(int k=0; k < count_rows(b); ++k) {
                    return_vector[i][j] = return_vector[i][j] + (a[i][j] * b[k][j]);
                }
            }
        }

        if(deep_debug) {
            output_matrix_to_console(return_vector, "multi->return_vector");
        }
        return return_vector;
    }

    /**
     * Subtrahiert Matrize b von double a
     * (baut Matrize für a auf und ruft subt() mit zwei Matrizen auf)
     *
     * @param a Minuend double wert
     * @param b Subtrahend Matrize
     * @return Subtraktionsmatrize
     */
    vector<vector<double>> subt(double a, vector<vector<double>> b) {
        unsigned long cols_b = count_cols(b);
        unsigned long rows_b = count_rows(b);

        vector<vector<double>> tmp_return_mat = {{0}};

        tmp_return_mat.resize(rows_b);
        for(int i=0; i < rows_b; ++i) {
            tmp_return_mat[i].resize(cols_b);
            for (int j = 0; j < cols_b; ++j) {
                tmp_return_mat[i][j] = a;
            }
        }

        return subt(tmp_return_mat,b);
    }

    /**
     * Subtrahiert Matrize b von Matrize a
     *
     * @param a Minuend
     * @param b Subtrahend
     * @return Subtraktionsmatrize
     */
    vector<vector<double>> subt(vector<vector<double>> a, vector<vector<double>> b) {
        if(deep_debug) {
            output_matrix_to_console(a, "subtraktion->a");
            output_matrix_to_console(b, "subtraktion->b");
        }

        unsigned long cols_a = count_cols(a), cols_b = count_cols(b);
        unsigned long rows_a = count_rows(a), rows_b = count_rows(b);

        vector<vector<double>> return_mat = {{0}};

        if(cols_a != cols_b || rows_b != rows_a) {
            // matrizen lassen sich nicht subtrahieren da sie nicht vom gleichen Typ sind
            return return_mat;
        }

        return_mat.resize(rows_a);
        for(int i=0; i < rows_a; ++i) {
            return_mat[i].resize(cols_a);
            for (int j = 0; j < cols_a; ++j) {
                return_mat[i][j] = a[i][j] - b[i][j];
            }
        }

        if(deep_debug) {
            output_matrix_to_console(return_mat, "subtraktion->return_mat");
        }

        return return_mat;
    }

    /**
     * Multipliziert zwei Matrizen
     *
     * @param a Matrizenfaktor
     * @param b Matrizenfaktor
     * @return Multiplikationsmatrize
     */
    vector<vector<double>> multi(vector<vector<double>> a, double b) {
        if(deep_debug) {
            output_matrix_to_console(a, "multi_with_double->a");
        }

        unsigned long cols = count_cols(a);
        unsigned long rows = count_rows(a);

        vector<vector<double>> return_mat = {{0}};
        return_mat.resize(rows);
        for(int i=0; i < rows; ++i) {
            return_mat[i].resize(cols);
            for (int j = 0; j < cols; ++j) {
                return_mat[i][j] = a[i][j] * b;
            }
        }

        return return_mat;
    }

    /**
     * Zählt die Reihen einer Matrize
     *
     * @param mat zu untersuchende Matrize
     * @return Anzahl der Reihen der Matrize
     */
    unsigned long count_rows(vector<vector<double>> mat) {
        return mat.size();
    }

    /**
     * Zählt die Spalten einer Matrize
     *
     * @param mat zu untersuchende Matrize
     * @return Anzahl der Spalten der Matrize
     */
    unsigned long count_cols(vector<vector<double>> mat) {
        if(mat.size() == 0) {
            return 0;
        }
        return mat[0].size();
    }
};

int main() {
    NeuronalNetwork nn = {3,3,3,0.1f, true};

    // zum Testen der Multifunktion
    //nn.multi({{1,1,1},{1,1,1}},{{1,1},{1,1},{3,2}});  // 5|4
                                                        // 5|4
    nn.train({1,2,3},{4,5,6});

    return 0;
}