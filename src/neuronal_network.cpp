//
// Created by incrediblexe on 07.06.17.
//
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <math.h>
#include <stdexcept>
#include <memory>
#include <random>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

using namespace boost::numeric::ublas;
using namespace std;

class neuronal_network {
public:
    neuronal_network (unsigned long l_input_nodes,
                      unsigned long l_hidden_nodes,
                      unsigned long l_output_nodes,
                      double learning_rate,double start=-.5,double end=.5) : m_weight_input_to_hidden{l_hidden_nodes,l_input_nodes}, m_weight_hidden_to_output{l_output_nodes,l_hidden_nodes} {
        d_learning_rate = learning_rate;


        fill_random(&m_weight_input_to_hidden,start,end);
        fill_random(&m_weight_hidden_to_output,start,end);

        /*std::cout << m_weight_input_to_hidden << std::endl;
        std::cout << m_weight_hidden_to_output << std::endl;*/
    }

    void train(matrix<double> inputs,std::vector<double> targets) {
        m_inputs = trans(inputs);
        m_targets = insert_vector(targets);

        // === Forward pass === //
        // lz verkürzt
        /*hidden_inputs = prod(m_weight_input_to_hidden, m_inputs);*/
        hidden_outputs = prod(m_weight_input_to_hidden, m_inputs);
        activation_function_mat(&hidden_outputs);

        // output_inputs wird nicht gebraucht
        output_outputs = prod(m_weight_hidden_to_output, hidden_outputs);
        /*output_outputs = output_inputs;*/

        // === back propagation == //
        // output schicht
        output_errors = (m_targets - output_outputs) * 1;

        // hidden schicht
        hidden_errors = prod(output_errors , m_weight_hidden_to_output); // 1|25

        if(m_one.size1() != hidden_outputs.size1() || m_one.size2() != hidden_outputs.size2()) {
            m_one.resize(hidden_outputs.size1(),hidden_outputs.size2());
            fill_with_double(&m_one, 1.d);
        }

        m_prod_ho_sub = (prod(hidden_outputs,trans((m_one - hidden_outputs)))); // 25|25
        hidden_grad = trans(prod(hidden_errors,m_prod_ho_sub));

        matrix<double> m_tmp = prod(output_errors,trans(hidden_outputs)),
                m_tmp2 = prod(hidden_grad,trans(m_inputs));

        multi_each(d_learning_rate,&m_tmp);
        multi_each(d_learning_rate,&m_tmp2);
        m_weight_hidden_to_output = m_weight_hidden_to_output + m_tmp;
        m_weight_input_to_hidden = m_weight_input_to_hidden + m_tmp2;

        delete_all_mat();
    }

    matrix<double> run(matrix<double> inputs) {
        matrix<double> final_outputs,hidden_inputs,
                hidden_outputs,final_inputs;

        matrix<double> m_inputs = trans(inputs);

        // === Forward pass === //

        hidden_outputs = prod(m_weight_input_to_hidden, m_inputs);
        /*cout << hidden_inputs << endl;*/

        activation_function_mat(&hidden_outputs);
        final_inputs = prod(m_weight_hidden_to_output, hidden_outputs);

        // unnötig
        /*final_outputs = final_inputs;*/

        return final_inputs;
    }

private:
    double d_learning_rate;
    matrix<double> m_weight_input_to_hidden,m_weight_hidden_to_output,m_one;

    // train matrix
    matrix<double> hidden_grad,m_prod_ho_sub,hidden_errors,output_errors,output_outputs,hidden_outputs,
            m_targets,m_inputs;

    void delete_all_mat() {
        hidden_grad.resize(0,0,false);
        m_prod_ho_sub.resize(0,0,false);
        hidden_errors.resize(0,0,false);
        output_errors.resize(0,0,false);
        output_outputs.resize(0,0,false);
        hidden_outputs.resize(0,0,false);
        m_targets.resize(0,0,false);
        m_inputs.resize(0,0,false);
    }

    void fill_random(matrix<double> *matrix, double start, double end) {
        for (unsigned i = 0; i < matrix->size1 (); ++ i)
            for (unsigned j = 0; j < matrix->size2 (); ++ j)
                (*matrix)(i, j) = random_double(-1.d,1.d);
    }

    void multi_each(double value,matrix<double> *mat) {
        for(matrix<double>::size_type ds=0; ds < (*mat).size1(); ds++) {
            for(matrix<double>::size_type val=0; val < (*mat).size2(); val++) {
                (*mat)(ds,val) = value * (*mat)(ds,val);
            }
        }
    }

    void fill_with_double(matrix<double> *mat, double val) {

        for(matrix<double>::size_type i = 0; i < mat->size1(); ++i) {
            for(matrix<double>::size_type j=0;j < mat->size2(); ++j) {
                (*mat)(i,j) = val;
            }
        }
    }

    matrix<double> insert_vector(std::vector<double> vec) {
        matrix<double> matrix(vec.size(),1.l);
        for ( unsigned long i = 0; i < vec.size(); ++i) {
            matrix(i,0) = vec[i];
        }

        return matrix;
    }

    matrix<double> insert_vector(std::vector<std::vector<double>> vec) {
        matrix<double> matrix(vec.size(),vec[0].size());
        for ( unsigned long ds = 0; ds < vec.size(); ds++) {
            for ( unsigned long val = 0; val < vec[ds].size(); ++val) {
                matrix(ds, val) = vec[ds][val];
            }
        }

        return matrix;
    }

    /**
     * Ruft für jeden Wert in der übergebenen Matrize die @see activation_function() auf und speichert den Wert in Rückgabematrize.
     *
     * @param mat Matrix Eingabematrize
     * @return Matrix Rückgabematrize
     */
    void activation_function_mat(matrix<double> *mat) {
        for(matrix<double>::size_type i = 0; i < mat->size1(); ++i)
            for(matrix<double>::size_type j=0;j < mat->size2(); ++j)
                (*mat)(i,j) = activation_function((*mat)(i,j));
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
     * Gibt eine Zufallszahl im gegebenen Intervall
     *
     * @param a Minimalwert
     * @param b Maximalwert
     * @return double Zufallswert zwischen den Parametern.
     */
    double random_double(double a, double b) {
        random_device rd;  //Will be used to obtain a seed for the random number engine
        mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        uniform_real_distribution<> dis(a, b);
        return dis(gen);
    }
};
