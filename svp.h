#pragma once
#include <omp.h>
#include <vector>
using namespace std;
#define typeElem int64_t

struct Tabl_elem {
    typeElem min;
    typeElem up;
    typeElem x;
    omp_lock_t lock;
};

struct Struct_Smith {
    vector<vector<typeElem>> L;
    vector<vector<typeElem>> S;
    vector<vector<typeElem>> R;
    typeElem d;
};

class svp {
private:
    typeElem d;
    int n;
    int p;
    int rankE;
    vector<typeElem> vectorSVP;
    vector<typeElem> S;
    vector<vector<typeElem>> tP;
    vector<vector<Tabl_elem>> Table;
public:
    int getN();
    int getRankE();
    int getPow();
    void setPow(int _newP);
    typeElem getMin();
    typeElem getDelta();
    vector<typeElem> getVectorSVP();

    svp();
    ~svp();
    svp(vector<vector<typeElem>> _mat, int _p = 2);
    vector<typeElem> Start_SVP(int _num_threads = 1);
private:
    Struct_Smith Normal_Form_Smith(vector<vector<typeElem>> _mat);
    typeElem Euclid(typeElem _a, typeElem _b, typeElem& _x, typeElem& _y);
    typeElem mod_S(typeElem _b, typeElem _s);
    vector<typeElem> diff_Vec(vector<typeElem> _v1, vector<typeElem> _v2, typeElem _k = 1);
    vector<typeElem> mult_k(vector<typeElem> _v, typeElem _k);
    
    vector<int> Create_Reverse_Notation(int _n, int _k);
    vector<typeElem> Search_Start_List();
    vector<typeElem> Search_Left_Right(typeElem _delta2, int _side = 1);
    typeElem Recursive_Search_Min(typeElem _b, int _m);
    typeElem Search_Min_One_Level(typeElem _b, int _m);
    Tabl_elem Search_Min_Left_Right(typeElem _b, int _m, Tabl_elem _minElem, typeElem _delta2, int _side = 1);
};
