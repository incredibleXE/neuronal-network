#include <iostream>
#include "src/neuronal_network.cpp"
#include <fstream>
#include <ctime>

using namespace std;

std::vector<string> explode(string const & s, char delim)
{
    std::vector<string> result;
    istringstream iss(s);

    for (string token; getline(iss, token, delim); )
    {
        result.push_back(move(token));
    }

    return result;
}

int random_int(int a, int b) {
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> dis(a, b);
    return (int) dis(gen);
}

double math_mean(std::vector<double> vec) {
    double ret_val = 0;

    for(unsigned int i=0;i < vec.size();i++) {
        ret_val += vec[i];
    }

    ret_val = ret_val / vec.size();

    return ret_val;
}

double varianz(std::vector<double> vec) {
    double ret_val = 0;

    double mean = math_mean(vec);

    for(unsigned int i=0;i < vec.size();i++) {
        ret_val += (pow(vec[i]-mean,2));
    }
    ret_val = ret_val / vec.size();

    return ret_val;
}

double std_varianz(std::vector<double> vec) {
    double ret_val = varianz(vec);

    return sqrt(ret_val);
}

/**
 * Mittlerer Fehler
 * @param y eingabe
 * @param Y Erwartet
 * @return
 */
double MSE(matrix<double> y, matrix<double> Y) {
    if(y.size2() != Y.size1())
        throw invalid_argument("die beiden Eingabevectoren müssen die gleiche Größe haben!");

    std::vector<double> v_ret(y.size2());
    for(matrix<double>::size_type ds = 0; ds < v_ret.size();ds++) {
        //cout << pow(y(0,ds)-Y(ds,2),2) << " = " << y(0,ds) << "-" << Y(ds,2)<< "^2" << endl;
        v_ret[ds] = pow(y(0,ds)-Y(ds,2),2);
    }

    return math_mean(v_ret);
}

std::vector<double> get_vec(matrix<double> mat,matrix<double>::size_type row) {
    std::vector<double> ret_vec(mat.size2());
    for(matrix<double>::size_type i = 0; i < mat.size2(); i++) {
        ret_vec[i] = mat(row,i);
    }

    return ret_vec;
}

/*
 * Mittelwer, Varianz und Std.abweichung Test
 * TODO: als catch Test formulieren
 */
/*int main() {
    *//*std::vector<double> t = {8,7,9,10,6};
    cout << math_mean(t) << endl;
    cout << varianz(t) << endl;
    cout << std_varianz(t);*//*

    *//*while(true) {
        cout << random_int(0,15439) << endl;
    }*//*

    return 0;
}*/
int main() {
    /*
     * Vars
     */
    // wenn true, kommt es nur zu einer Konsolenausgabe (erste Zeile ist der Tabellenkopf)
    bool ignore_first_line = true;
    // Zeilenendzeichen
    char line_end = '\r',
        // Comma Seperated => ',' Trennzeichen für Daten einer Zeile
         explode_char=',';

    // Durchläufe
    unsigned int epochs = 5000, train_epochs = 128;
    // Lernrate
    double lr = 0.005;
    // Anzahl der Hidden und Output Neuronen
    unsigned int hidden_nodes = 25, output_nodes = 1;
    // Pfad zur CSV Datei
    string data_path = "/home/incrediblexe/projects/clion/neuronal-network/data/hour.csv";
    // input und target matrizen
    matrix<double> m_csv_input(17379,8);
    matrix<double> m_train_features,m_train_targets,
                    m_test_features,m_test_targets,
                    m_val_features,m_val_targets;

    /**
     * 0 => wird ignoriert
     * 1 => wird nicht ignoriert
     */
    int important_fields[] = {
            0,      //instant
            0,      //dteday
            0,      //season
            1,      //yr
            0,      //mnth
            0,      //hr
            1,      //holiday
            0,      //weekday
            0,      //workingday
            0,      //weathersit
            1,      //temp
            0,      //atemp
            1,      //hum
            1,      //windspeed
            1,      //casual
            1,      //registered
            1       //cnt
    };

    /**
     * 0 => wird ignoriert
     * 1 => input_list
     * 2 => target_list
     */
    int input_target_selector[] = {
            1,1,1,1,1,2,2,2
    };

    /*
     * Einlesen der CSV Datei
     */
    ifstream csv_read;
    csv_read.open(data_path, ios::in);
    matrix<double>::size_type i_lines = 0;
    if(csv_read){
        //Datei bis Ende einlesen und bei '\r' strings trennen
        string s="";
        // erste Zeile enthaelt Tabellenkopf
        getline(csv_read, s, line_end);

        bool error = false;
        unsigned long important_fields_length = (sizeof(important_fields)/sizeof(*important_fields));
        while(getline(csv_read, s, line_end))
        {
            if(i_lines > m_csv_input.size1()) {
                m_csv_input.resize(m_csv_input.size1()+1,m_csv_input.size2());
            }

            // string bei "," trennen
            std::vector<string> line = explode(s,explode_char),input_tupel,target_tupel;
            std::string::size_type sz;

            if(line.size() != important_fields_length) {
                if(error)
                    throw invalid_argument( "Der Bewertungsarray important_fields deckt nicht alle Felder aus der CSV Datei ab oder anders herum.");

                error = true;
                continue;
            }

            matrix<double>::size_type i_matrix_count = 0;
            for(unsigned long i=0;i < important_fields_length;i++) {
                if(important_fields[i] != 0) {
                    m_csv_input(i_lines,i_matrix_count) = stod(line[i], &sz);
                    i_matrix_count++;
                }
            }

            i_lines++;

            line.erase (line.begin(),line.end());
        }

        csv_read.close();
    }
    else{
        if(!csv_read.is_open())
            throw invalid_argument( "Fehler beim Lesen der Datei! Die Datei konnte nicht geöffnet werden." );

        throw invalid_argument( "Fehler beim Lesen der Datei! Ein allgemeiner Fehler ist aufgetreten." );
    }

    /*cout << m_csv_input << endl;*/

    /*
     * Scalierung der Daten für das NN
     */
    matrix<double> m_csv_input_t = trans(m_csv_input);
    std::vector<double> v_mean(m_csv_input_t.size1()), v_std(m_csv_input_t.size1());

    for(matrix<double>::size_type i = 0; i < m_csv_input_t.size1();i++) {
        auto vec = get_vec(m_csv_input_t,i);
        v_mean[i] = math_mean(vec);
        v_std[i] = std_varianz(vec);

        /*cout <<"Mittelwert: " << m_mean[i] << " | Standardabweichung: " << m_std[i] << endl;*/
    }
    // "delete" matrix
    m_csv_input_t.resize(0,0,false);

    matrix<double> m_data(m_csv_input.size1(),m_csv_input.size2());
    for(matrix<double>::size_type ds=0; ds < m_csv_input.size1(); ds++) {
        for(matrix<double>::size_type val=0; val < m_csv_input.size2(); val++) {
            // val = 2 weil die ersten beiden Werte (yr ,holiday) nicht angefasst werden müssen
            double d_val = (val>1) ? (m_csv_input(ds,val) - v_mean[val]) / v_std[val]
                                   : m_csv_input(ds,val);
            m_data(ds,val) = d_val;
        }
    }
    // "delete" csv matrix
    m_csv_input.resize(0,0,false);

    //cout << m_data_new.size1() << " | " << m_data_new.size2() << endl;
    // 1944 => 504 für Testdaten , 1440 für Validation Daten
    m_train_features.resize(m_data.size1()-1944,5);
    m_train_targets.resize(m_data.size1()-1944,3);
    m_val_features.resize(1440,5);
    m_val_targets.resize(1440,3);
    m_test_features.resize(504,5);
    m_test_targets.resize(504,3);

    matrix<double>::size_type offset = m_data.size1() - 504, offset_val = m_data.size1() - 1944;
    for(matrix<double>::size_type i_row = 0;i_row < m_data.size1();i_row ++) {
        for(matrix<double>::size_type i_val = 0;i_val < m_data.size2();i_val ++) {
            if(i_row < m_data.size1()-504) {
                if(i_row < m_data.size1()-1944) {
                    //cout << m_features.size1() << "|" << m_data.size1()-1440 << ": " << i_row << endl;
                    if (i_val < 5) {
                        m_train_features(i_row, i_val) = m_data(i_row, i_val);
                    } else {
                        m_train_targets(i_row, i_val - 5) = m_data(i_row, i_val);
                    }
                } else {
                    if (i_val < 5) {
                        m_val_features(i_row - offset_val, i_val) = m_data(i_row, i_val);
                    } else {
                        m_val_targets(i_row - offset_val, i_val - 5) = m_data(i_row, i_val);
                    }
                }
            } else {
                if(i_val < 5) {
                    m_test_features(i_row - offset, i_val) = m_data(i_row, i_val);
                } else {
                    m_test_targets(i_row - offset, i_val-5) = m_data(i_row, i_val);
                }

            }
        }
    }

    // m_data löschen
    m_data.resize(0,0,false);

    // 504 sind die letzten 21 Tage (aus Vorlage)
    // 1440 sind die letzten 60 Tage (aus Vorlage)
    // m_test_targets => alle Zielwerte der Test matrix (504x3)
    // m_test_features => alle Ausgangswerte der Test matrix (504x5)
    // m_val_targets => alle Zielwerte der Validation matrix (1440x3)
    // m_val_features => alle Ausgangswerte der Validation matrix (1440x5)
    // m_train_targets => alle Zielwerte der Train matrix (15435x3)
    // m_train_features => alle Ausgangswerte der Train matrix (15435x5)
    //
    // 17379 DS = 504 DS + 1440 DS + 15435 DS
    //

    // Testausgabe ob alles stimmt
    /*cout << "m_train_features:        " << m_train_features.size1()<<"|"<<m_train_features.size2()<<endl;
    cout << "m_train_targets:         " << m_train_targets.size1()<<"|"<<m_train_targets.size2()<<endl;
    cout << "m_val_features:    " << m_val_features.size1()<<" |"<<m_val_features.size2()<<endl;
    cout << "m_val_targets:     " << m_val_targets.size1()<<" |"<<m_val_targets.size2()<<endl;
    cout << "m_test_features:   " << m_test_features.size1()<<"  |"<<m_test_features.size2()<<endl;
    cout << "m_test_targets:    " << m_test_targets.size1()<<"  |"<<m_test_targets.size2()<<endl;*/

    neuronal_network nn = {m_train_features.size2(),hidden_nodes,output_nodes,lr};

    // Zeitanalyse
    clock_t start;
    float elapsed;
    start = clock();

    for(unsigned int e=0;e <= epochs;e++) {
        // die ersten 128 (train_epochs) Zeilen auslesen
        matrix<double>::size_type coincidence = random_int(0,m_test_features.size1()-train_epochs);
        for(matrix<double>::size_type row=0;row <= train_epochs && row+coincidence < m_test_features.size1();row++) {
            matrix<double> m_input_tupel(1,m_test_features.size2());
            for(matrix<double>::size_type val=0;val < m_input_tupel.size2();val++) {
                m_input_tupel(0,val) = m_train_features(row+coincidence,val);
            }

            nn.train(m_input_tupel, {m_test_targets(row,2)});
        }

        double train_loss = MSE(nn.run(m_train_features),m_train_targets);
        double val_loss = MSE(nn.run(m_val_features),m_val_targets);

        if((e % 100) == 0 || e < 5) {
            float val = (100 * e / epochs);
            cout << endl << "Progress: " << val
                 << "% ... Train loss: " << train_loss
                 << " ... Validation loss: " << val_loss;
        }
    }

    elapsed = (float)(clock() - start) / CLOCKS_PER_SEC;
    cout << "fertig - benötigte Zeit: " << elapsed << endl;

    return 0;
}

