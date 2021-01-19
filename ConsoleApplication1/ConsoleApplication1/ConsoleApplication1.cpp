#include <iostream>
#include <vector>
#include <chrono> 
#include <functional>
#include <algorithm>

using namespace std;
using namespace std::chrono;


long double fun(vector<vector<long double>> x)
{
    return 100.0 * (x[1][0] - x[0][0] * x[0][0]) * (x[1][0] - x[0][0] * x[0][0]) + 1.0 * (1 - x[0][0]) * (1 - x[0][0]);
    //return (x[1][0]*x[1][0]) + (x[0][0])*(x[0][0]); 
}


vector<vector<long double>> sub_vect(vector<vector<long double>> A, vector<vector<long double>> B)
{
    vector<vector<long double>> C(A.size(), vector<long double>(A[0].size(), 0));
    for (int j = 0; j < A[0].size(); j++)
    {
        for (int i = 0; i < A.size(); i++)
        {

            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

vector<vector<long double>> add_vect(vector<vector<long double>> A, vector<vector<long double>> B)
{
    vector<vector<long double>> C(A.size(), vector<long double>(A[0].size(), 0));
    for (int j = 0; j < A[0].size(); j++)
    {
        for (int i = 0; i < A.size(); i++)
        {

            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}
vector<vector<long double>> scalar_multiplier(vector<vector<long double>> vect, long double multiplier)
{
    for (int j = 0; j < vect[0].size(); j++)
    {
        for (int i = 0; i < vect.size(); i++)
        {
            vect[i][j] = vect[i][j] * multiplier;
        }
    }
    return vect;
}

vector<vector<long double>> transpose_vect(vector<vector<long double>> vect)
{
    vector<vector<long double>> trans_vec(vect[0].size(), vector<long double>(vect.size(), 0));
    for (int i = 0; i < vect.size(); i++)
    {
        for (int j = 0; j < vect[0].size(); j++)
        {
            trans_vec[j][i] = vect[i][j];
        }
    }
    return trans_vec;
}

vector<vector<long double>> dot_product(vector<vector<long double>> A, vector<vector<long double>> B)
{
    vector<vector<long double>> C(A.size(), vector<long double>(B[0].size(), 0));
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
    return C;
}

long double dot_product_1by2(vector<vector<long double>> A, vector<vector<long double>> B)
{
    long double product = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    return product;
}

vector<vector<long double>> gradient(vector<vector<long double>> x)
{
    vector<vector<long double>> grad;
    long double xx0;
    vector <long double> temp;
    for (int i = 0; i < x.size(); i++)
    {
        temp.clear();
        long double eps = 0.00001492;
        xx0 = 1. * x[i][0];
        long double f0 = fun(x);
        x[i][0] = x[i][0] + eps;
        long double f1 = fun(x);
        long double f3 = (f1 - f0) / eps;
        temp.push_back(f3);
        grad.push_back(temp);
        x[i][0] = xx0;
    }
    return grad;
}

vector<vector<long double>> hessianMatrix(vector<vector<long double>> x, long double f)
{
    vector<vector<long double>> hessian(x.size(), vector<long double>(x.size(), 0));
    vector<vector<long double>> H_inv(x.size(), vector<long double>(x.size(), 0));
    vector<vector<long double>> gd_0 = gradient(x);
    long double eps = 0.00001492;
    long double eps_1 = 1 / eps;
    long double xx0;
    for (int i = 0; i < x.size(); i++)
    {
        xx0 = x[i][0];
        x[i][0] = xx0 + eps;
        vector<vector<long double>> gd_1 = gradient(x);
        vector<vector<long double>>gd_2 = sub_vect(gd_1, gd_0);

        vector<vector<long double>>gd_3 = scalar_multiplier(gd_2, eps_1);
        hessian[0][i] = gd_3[0][0];
        hessian[1][i] = gd_3[1][0];
        x[i][0] = xx0;
    }
    long double determinant = 0;
    determinant = (hessian[0][0] * hessian[1][1]) - (hessian[0][1] * hessian[1][0]);
    if (determinant < 0)
    {
        cout << "Not positive Semi definite, Newton Method fails" << endl;
        exit(0);
    }
    H_inv[0][0] = hessian[1][1] / determinant;
    H_inv[0][1] = -1 * hessian[1][0] / determinant;
    H_inv[1][0] = -1 * hessian[0][1] / determinant;
    H_inv[1][1] = hessian[0][0] / determinant;
    return H_inv;
}

long double calc_alpha(vector<vector<long double>> x_dash, vector<vector<long double>> del_x, vector<vector<long double>> Pk, vector<vector<long double>> x0, long double alpha, long double dk, long double c, long double rho)
{
    int i = 0;
    long double func = fun(x0);
    while (fun(x_dash) > func + c * alpha * dk) // f(x0 + alpha_K*Pk) > f(x0) + c * alpha_k * (del_f)^T * Pk
    {
        alpha = alpha * rho;
        del_x = scalar_multiplier(Pk, alpha);
        x_dash = add_vect(x0, del_x);
        i++;
    }
    return alpha;
}


long double backtrack(vector<vector<long double>> x0, long double alpha, long double c, long double rho, int dir)
{
    vector<vector<long double>> grad = gradient(x0);
    long double func = fun(x0);
    vector<vector<long double>> trans_grad = transpose_vect(grad);
    vector<vector<long double>> Pk = scalar_multiplier(grad, -1);

    if (dir == 1)
    {
        long double dk = dot_product_1by2(trans_grad, Pk);
        vector<vector<long double>> del_x = scalar_multiplier(Pk, alpha);
        vector<vector<long double>> x_dash = add_vect(x0, del_x);
        alpha = calc_alpha(x_dash, del_x, Pk, x0, alpha, dk, c, rho);
    }

    if (dir == 2)
    {
        vector<vector<long double>> H_inv = hessianMatrix(x0, func);
        //cout << H_inv[0][0] << H_inv[1][0] << H_inv[0][1] << H_inv[1][1] << endl;
        vector<vector<long double>> Pk_h = dot_product(H_inv, Pk);
        long double dk = dot_product_1by2(trans_grad, Pk_h);
        vector<vector<long double>> del_x = scalar_multiplier(Pk_h, alpha);
        vector<vector<long double>> x_dash = add_vect(x0, del_x);
        alpha = calc_alpha(x_dash, del_x, Pk_h, x0, alpha, dk, c, rho);
    }

    return alpha;
}


int main()
{
    auto start = high_resolution_clock::now();
    vector<vector<long double>> x0 = {
        {1.2},
        {1.2}
    };
    long double alpha = 1;
    long double c = 0.6;
    long double rho = 0.9;
    long double f_prev = fun(x0);
    int i = 0;
    long double eps = 0.000001;

    int dir;
    cout << "Choose Search Direction :-" << endl;
    cout << "Steepest Descent: Press 1" << endl;
    cout << "Newton Method: Press 2" << endl;
    cin >> dir;
    while (true)
    {
        if ((dir == 1) || (dir == 2))
        {
            break;
        }
        cout << "Please Enter correct value: " << endl;
        cin >> dir;
    }


    while (true)
    {
        cout << "###Iteration number: " << i + 1 << endl;
        long double alpha_k = backtrack(x0, alpha, c, rho, dir);
        cout << " Step Size: " << alpha_k << endl;
        vector<vector<long double>> grad = gradient(x0);
        vector<vector<long double>> Pk = scalar_multiplier(grad, alpha_k);


        if (dir == 1)
        {
            x0 = sub_vect(x0, Pk);
        }

        if (dir == 2)
        {
            vector<vector<long double>> H_inv = hessianMatrix(x0, fun(x0));
            vector<vector<long double>> Pk_h = dot_product(H_inv, Pk);
            x0 = sub_vect(x0, Pk_h);
        }

        //cout << x0[0][0] << "," << x0[1][0] << endl;
        alpha = 1;
        long double f_current = fun(x0);
        //cout << f_prev << " " << f_current << " " << f_current - f_prev << " " << "Stopping margin" << endl;
        i = i + 1;
        if (((f_prev - f_current) < eps))
        {
            break;
        }
        f_prev = f_current;
    }
    cout << "Minumum Value is: ( " << x0[0][0] << " " << x0[1][0] << " )" << endl;
    cout << "Function value is:" << fun(x0) << endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    //cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    return 0;
}
