#include <iostream>
#include <vector>
#include <math.h>
// für shared_ptr
#include <memory>

using namespace std;

class NeuronalNetwork {
public:
    bool b_debug = false, deep_debug = false;

    vector<vector<float>> v_weight_input_to_hidden,v_weight_hidden_to_output;
    float f_learning_rate;

    NeuronalNetwork(int i_input_nodes, int i_hidden_nodes, int i_output_nodes, float learning_rate, bool debug = false) {
        b_debug = debug;
        deep_debug = true;
        f_learning_rate = learning_rate;

        // input zu hidden layer Gewichtungen
        v_weight_input_to_hidden.resize((unsigned long) i_input_nodes);
        for(int i = 0; i < i_input_nodes; ++i) {
            v_weight_input_to_hidden[i].resize((unsigned long) i_hidden_nodes);
            for(int a = 0; a < i_hidden_nodes; ++a) {
                v_weight_input_to_hidden[i][a] = random_float(-1.f,1.f);
            }
        }
        output_matrix_to_console(v_weight_input_to_hidden,"v_weight_input_to_hidden");

        // hidden zu output layer Gewichtungen
        v_weight_hidden_to_output.resize((unsigned long) i_hidden_nodes);
        for(int i = 0; i < i_hidden_nodes; ++i) {
            v_weight_hidden_to_output[i].resize((unsigned long) i_output_nodes);
            for(int a = 0; a < (unsigned long) i_output_nodes; ++a) {
                v_weight_hidden_to_output[i][a] = random_float(-1.f,1.f);
            }
        }
        output_matrix_to_console(v_weight_hidden_to_output,"v_weight_hidden_to_output");
    }

    void train(vector<float> input_list,vector<float> target_list) {
        cout << "starting train\n" << "--------------" << endl;

        // Transponieren der vectoren
        vector<vector<float>> inputs = T(input_list);
        output_matrix_to_console(inputs,"train->inputs");
        vector<vector<float>> targets = T(target_list);
        output_matrix_to_console(targets,"train->targets");

        // === Forward pass === //
        // hidden schicht
        vector<vector<float>> hidden_inputs = multi(v_weight_input_to_hidden,inputs);
        output_matrix_to_console(hidden_inputs,"train->hidden_inputs");
        vector<vector<float>> hidden_outputs = activation_function_mat(hidden_inputs);
        output_matrix_to_console(hidden_outputs,"train->hidden_outputs");

        // output schicht
        vector<vector<float>> output_inputs = multi(v_weight_hidden_to_output,hidden_outputs);
        output_matrix_to_console(output_inputs,"train->output_inputs");
        vector<vector<float>> output_outputs = output_inputs;
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
        vector<vector<float>> output_errors = multi(subt(targets,output_outputs),1.f);
        output_matrix_to_console(output_errors,"train->output_errors");

        // hidden schicht
        // TODO: im Beispiel output_errors * v_weight_hidden_to_output, geht aber aufgrund von Matrixregeln nicht. Richtig?
        vector<vector<float>> hidden_errors = multi(T(output_errors),v_weight_hidden_to_output);
        output_matrix_to_console(hidden_errors,"train->hidden_errors");

        vector<vector<float>> hidden_grad = multi(multi(T(hidden_errors),hidden_outputs),subt(-1.f,hidden_outputs));
        output_matrix_to_console(hidden_grad,"train->hidden_grad");

        return;
    }

    float activation_function(float x) {
        return 1/(1+exp(x));
    }

    vector<vector<float>> activation_function_mat(vector<vector<float>> mat) {
        vector<vector<float>> return_mat;
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

    void output_matrix_to_console(vector<vector<float>> mat, string name = "Unknown") {
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
    float random_float(float a, float b) {
        float random = ((float) rand()) / (float) RAND_MAX;
        float diff = b - a;
        float r = random * diff;
        return a + r;
    }

    vector<vector<float>> T(vector<float> input_list) {
        vector<vector<float>> outputs;
        outputs.resize(input_list.size());
        for(int i=0;i < input_list.size();++i) {
            outputs[i].resize(1);
            outputs[i][0] = input_list[i];
        }

        return outputs;
    }

    vector<vector<float>> T(vector<vector<float>> mat) {
        unsigned long rows = count_rows(mat);
        unsigned long cols = count_cols(mat);

        vector<vector<float>> return_mat = {{0}};

        return_mat.resize(cols);
        for(int i=0; i < rows; ++i) {
            return_mat[i].resize(rows);
            for (int j = 0; j < cols; ++j) {
                return_mat[j][i] = mat[i][j];
            }
        }

        return return_mat;
    }

    vector<vector<float>> multi(vector<vector<float>> a, vector<vector<float>> b) {
        if(deep_debug) {
            output_matrix_to_console(a, "multi->a");
            output_matrix_to_console(b, "multi->b");
        }

        // @fgr: ist {{}} nötig?
        vector<vector<float>> return_vector = {{0}};

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

    vector<vector<float>> subt(float a, vector<vector<float>> b) {
        unsigned long cols_b = count_cols(b);
        unsigned long rows_b = count_rows(b);

        vector<vector<float>> tmp_return_mat = {{0}};

        tmp_return_mat.resize(rows_b);
        for(int i=0; i < rows_b; ++i) {
            tmp_return_mat[i].resize(cols_b);
            for (int j = 0; j < cols_b; ++j) {
                tmp_return_mat[i][j] = a;
            }
        }

        return subt(tmp_return_mat,b);
    }

    vector<vector<float>> subt(vector<vector<float>> a, vector<vector<float>> b) {
        if(deep_debug) {
            output_matrix_to_console(a, "subtraktion->a");
            output_matrix_to_console(b, "subtraktion->b");
        }

        unsigned long cols_a = count_cols(a), cols_b = count_cols(b);
        unsigned long rows_a = count_rows(a), rows_b = count_rows(b);

        vector<vector<float>> return_mat = {{0}};

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

    vector<vector<float>> multi(vector<vector<float>> a, float b) {
        if(deep_debug) {
            output_matrix_to_console(a, "multi_with_float->a");
        }

        unsigned long cols = count_cols(a);
        unsigned long rows = count_rows(a);

        vector<vector<float>> return_mat = {{0}};
        return_mat.resize(rows);
        for(int i=0; i < rows; ++i) {
            return_mat[i].resize(cols);
            for (int j = 0; j < cols; ++j) {
                return_mat[i][j] = a[i][j] * b;
            }
        }

        return return_mat;
    }

    unsigned long count_rows(vector<vector<float>> mat) {
        return mat.size();
    }

    unsigned long count_cols(vector<vector<float>> mat) {
        if(mat.size() == 0) {
            return 0;
        }
        return mat[0].size();
    }
};

int main() {
    NeuronalNetwork nn = {3,3,3,0.1f, true};

    //nn.multi({{1,1,1},{1,1,1}},{{1,1},{1,1},{3,2}});  // 5|4
                                                        // 5|4
    nn.train({1,2,3},{4,5,6});

    return 0;
}