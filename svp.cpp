#include "svp.h"



svp::svp()
{
    this->d = 0;
    this->n = 1;
    this->p = 2;
    this->rankE = 1;
    this->vectorSVP = vector<typeElem>(0);
    this->S = vector<typeElem>(1);
    this->tP = vector<vector<typeElem>>(1, vector<typeElem>(1));
    this->Table = vector<vector<Tabl_elem>>(1, vector<Tabl_elem>(1));
}



svp::svp(vector<vector<typeElem>> _mat, int _p)
{
    Struct_Smith form_smith = Normal_Form_Smith(_mat);
    d = form_smith.d;
    p = _p;
    n = _mat.size();
    if (d < 0) {
        d *= -1;
    }

    tP = vector<vector<typeElem>>(n, vector<typeElem>(n));
    Table = vector<vector<Tabl_elem>>(d, vector<Tabl_elem>(n));
    Table[0][n - 1].min = 0;

    for (int i = 0; i < n; i++) {
        Table[0][n - 1].min += pow(abs(_mat[i][0]), _p);
        vectorSVP.push_back(_mat[i][0]);
        S.push_back(form_smith.S[i][i]);
        for (int j = 0; j < n; j++) {
            tP[j][i] = mod_S(form_smith.L[i][j], S[i]);
        }
    }

    rankE = n - 1;
    for (int i = 0; i < n; i++) {
        if (S[i] != 1) {
            rankE = i;
            break;
        }
    }

    for (int i = 0; i < d; i++) {
        for (typeElem j = 0; j < n; j++) {
            omp_init_lock(&Table[i][j].lock);
            Table[i][j].up = -1;
        }
    }
}



int svp::getN() { return this->n; }
int svp::getRankE() { return this->rankE; }
int svp::getPow() { return this->p; }
void svp::setPow(int _new_p) { this->p = _new_p; }
typeElem svp::getMin() { return this->Table[0][n - 1].min; }
typeElem svp::getDelta() { return this->d; }
vector<typeElem> svp::getVectorSVP() { return this->vectorSVP; }



svp::~svp()
{
    for (int i = 0; i < d; i++) {
        for (typeElem j = 0; j < n; j++) {
            omp_destroy_lock(&Table[i][j].lock);
        }
    }
}



vector<int> svp::Create_Reverse_Notation(int _n, int _k) {
    int higher = -1;
    vector<int> rev(_n);
    rev[0] = 0;
    for (int i = 1; i < _n; i++) {
        if ((i & (i - 1)) == 0)
            higher++;
        rev[i] = rev[i ^ (1 << higher)];
        rev[i] |= 1 << (_k - higher - 1);
    }
    return rev;
}



vector<typeElem> svp::Start_SVP(int _num_threads) {

    if (_num_threads < 1) {
        _num_threads = 1;
    }

    int _n = 1;
    int _k = 0;
    while (_n < _num_threads) {
        _n = _n << 1;
        _k++;
    }
    vector<int> rev = Create_Reverse_Notation(_n, _k);

    vector<typeElem> list = Search_Start_List();
    vector<int> size_data(_num_threads, list.size() / _num_threads);
    int remain = list.size() % _num_threads;
    if (remain > 0) {
        for (int i = 0; i < remain; i++) {
            size_data[i]++;
        }
    }
    vector<int> id_data(_num_threads);
    id_data[0] = 0;
    for (int i = 1; i < id_data.size(); i++) {
        id_data[i] = id_data[i - 1] + size_data[i - 1];
    }

    vector<int> id_rev_data(_num_threads);
    vector<int> size_rev_data(_num_threads);
    int tmp = 0;
    for (int i = 0; i < _n; i++) {
        if (rev[i] >= _num_threads) {
            tmp++;
            continue;
        }
        id_rev_data[i - tmp] = id_data[rev[i]];
        size_rev_data[i - tmp] = size_data[rev[i]];
    }

    omp_set_dynamic(0);
    omp_set_num_threads(_num_threads);
#pragma omp parallel for
    for (int i = 0; i < _num_threads; i++) {
        for (int j = 0; j < size_rev_data[i]; j++) {
            Recursive_Search_Min(list[id_rev_data[i] + j], n - 2);
        }
    }

    typeElem resultMin = Recursive_Search_Min(0, n - 1);
    typeElem id_up = Table[0][n - 1].up;
    if (id_up == -1) {
        return vectorSVP;
    }
    vector<typeElem> resultMinX(n, 0);
    resultMinX[n - 1] = Table[0][n - 1].x;
    for (int i = n - 2; i >= 0; i--) {
        if (id_up == -2) {
            break;
        }
        id_up = Table[id_up][i].up;
        resultMinX[i] = Table[id_up][i].x;
    }
    vectorSVP = resultMinX;
    return resultMinX;
}



Struct_Smith svp::Normal_Form_Smith(vector<vector<typeElem>> _mat) {

    int k = 0, flag_1 = 3, flag_2 = 0;
    int _col = _mat[0].size();
    int _row = _mat.size();
    typeElem det = 1;
    vector<vector<typeElem>> res = _mat;
    vector<vector<typeElem>> res_R(_col, vector<typeElem>(_row, 0));
    vector<vector<typeElem>> res_L(_col, vector<typeElem>(_row, 0));

    for (int i = 0; i < _row && i < _col; i++) {
        res_R[i][i] = 1;
        res_L[i][i] = 1;
    }


    while (k < _col && k < _row) {
        if (flag_1 == 3) {
            flag_2 = 0;
            typeElem min;
            int min_i, min_j;
            for (int i = k; i < _row; i++) {
                for (int j = k; j < _col; j++) {
                    if (res[i][j] != 0) {
                        min = abs(res[i][j]);
                        min_j = j;
                        min_i = i;
                        flag_2 = 1;
                        break;
                    }
                }
                if (flag_2 == 1) {
                    break;
                }
            }
            if (flag_2 == 0) {
                break;
            }
            for (int i = min_i; i < _row; i++) {
                for (int j = min_j; j < _col; j++) {
                    if (res[i][j] != 0 && abs(res[i][j]) < min) {
                        min = abs(res[i][j]);
                        min_j = j;
                        min_i = i;
                    }
                }
            }
            if (k != min_j) {
                det *= -1;
                for (int i = 0; i < _row; i++) {
                    typeElem tmp = res[i][k];
                    res[i][k] = res[i][min_j];
                    res[i][min_j] = tmp;
                }
            }
            if (k != min_i) {
                res_L[min_i].swap(res_L[k]);
            }
            if (k != min_j) {
                for (int i = 0; i < _col; i++) {
                    typeElem tmp = res_R[i][k];
                    res_R[i][k] = res_R[i][min_j];
                    res_R[i][min_j] = tmp;
                }
            }
            if (k != min_i) {
                det *= -1;
                res[min_i].swap(res[k]);
            }
            if (res[k][k] < 0) {
                res[k] = mult_k(res[k], -1);
                det *= -1;
                res_L[k] = mult_k(res_L[k], -1);
            }
            flag_1 = 4;
        }
        if (flag_1 == 4) {
            for (int i = k + 1; i < _row; i++) {
                if (res[i][k] != 0) {
                    typeElem koef = res[i][k] / res[k][k];
                    res[i] = diff_Vec(res[i], res[k], koef);
                    res_L[i] = diff_Vec(res_L[i], res_L[k], koef);
                    flag_1 = 3;
                    break;
                }
            }
            if (flag_1 == 4) {
                flag_1 = 5;
            }
        }
        if (flag_1 == 5) {
            for (int j = k + 1; j < _col; j++) {
                if (res[k][j] != 0) {
                    typeElem koef = res[k][j] / res[k][k];
                    for (int i = 0; i < _row; i++) {
                        res[i][j] -= res[i][k] * koef;
                    }
                    for (int i = 0; i < _col; i++) {
                        res_R[i][j] -= res_R[i][k] * koef;
                    }
                    flag_1 = 3;
                    break;
                }
            }
            if (flag_1 == 5) {
                flag_1 = 6;
            }
        }
        if (flag_1 == 6) {
            for (int i = k + 1; i < _row; i++) {
                for (int j = k + 1; j < _col; j++) {
                    if (res[i][j] % res[k][k] != 0) {
                        for (int l = 0; l < _row; l++) {
                            res[l][k] += res[l][j];
                        }
                        for (int l = 0; l < _col; l++) {
                            res_R[l][k] += res_R[l][j];
                        }
                        flag_1 = 4;
                        break;
                    }
                }
                if (flag_1 == 4) {
                    break;
                }
            }
            if (flag_1 == 6) {
                k++;
                flag_1 = 3;
            }
        }
    }
    if (_col == _row) {
        for (int i = 0; i < _row; i++) {
            det *= res[i][i];
        }
    }
    Struct_Smith form_Smith;
    form_Smith.d = det;
    form_Smith.S = res;
    form_Smith.L = res_L;
    form_Smith.R = res_R;
    return form_Smith;
}



typeElem svp::Euclid(typeElem _a, typeElem _b, typeElem& _x, typeElem& _y) {

    if (_b == 0) {
        _y = 0;
        _x = 1;
        return _a;
    }
    typeElem d1, x1 = 0, y1 = 0;
    d1 = Euclid(_b, _a % _b, x1, y1);
    _x = y1;
    _y = x1 - ((_a / _b) * y1);
    return d1;
}




typeElem svp::mod_S(typeElem _b, typeElem _s) {

    if (_s == 1 || _b == 0) {
        return 0;
    }
    typeElem result = _b - (_s * (_b / _s));
    if (result == 0) {
        return 0;
    }
    if (_b > 0) {
        return result;
    }
    else {
        return _s + result;
    }
}



vector<typeElem> svp::mult_k(vector<typeElem> _v, typeElem _k) {

    vector<typeElem> res = _v;
    for (int j = 0; j < _v.size(); j++) {
        res[j] *= _k;
    }
    return res;
}



vector<typeElem> svp::diff_Vec(vector<typeElem> _v1, vector<typeElem> _v2, typeElem _k) {

    if (_v1.size() == _v2.size()) {
        vector<typeElem> res(_v1.size(), 0);
        for (int j = 0; j < _v1.size(); j++) {
            res[j] = _v1[j] - (_v2[j] * _k);
        }
        return res;
    }
    return vector<typeElem>(1, 0);
}



vector<typeElem> svp::Search_Start_List() {

    typeElem delta_2 = (d / 2) + 1;
    vector<typeElem> list = Search_Left_Right(delta_2, 1);
    vector<typeElem> list2 = Search_Left_Right(delta_2, -1);

    list.insert(list.begin(), 0);
    list.insert(list.end(), list2.begin(), list2.end());
    return list;
}



vector<typeElem> svp::Search_Left_Right(typeElem _delta_2, int _side) {

    vector<typeElem> list(0);
    for (typeElem l = 1; l < _delta_2; l++) {

        vector<typeElem> test_real_b(n, 0);
        vector<typeElem> new_B(n, 0);
        typeElem b_Px = -1;

        for (int i = rankE; i < n; i++) {
            new_B[i] = mod_S(-l * _side * tP[n - 1][i], S[i]);
        }
        for (int i = rankE; i < n; i++) {
            test_real_b[i] = mod_S(new_B[n - 1], S[i]);
        }

        for (int i = rankE; i < n; i++) {
            if (new_B[i] != test_real_b[i]) {
                b_Px = -2;
                break;
            }
        }
        if (b_Px == -2) {
            continue;
        }

        b_Px = new_B[n - 1];
        if (b_Px == 0) {
            break;
        }
        list.push_back(b_Px);
    }
    return list;
}




typeElem svp::Recursive_Search_Min(typeElem _b, int _m) {

    omp_set_lock(&Table[_b][_m].lock);
    if (Table[_b][_m].up != -1) {
        typeElem res = Table[_b][_m].min;
        omp_unset_lock(&Table[_b][_m].lock);
        return res;
    }
    if (_m == 0) {
        Table[_b][0].x = Search_Min_One_Level(_b, 0);
        if (Table[_b][0].x == d || Table[_b][0].x == 0) {
            omp_unset_lock(&Table[_b][0].lock);
            return -1;
        }
        Table[_b][0].up = _b;
        Table[_b][0].min = pow(abs(Table[_b][0].x), p);

        typeElem res = Table[_b][0].min;
        omp_unset_lock(&Table[_b][0].lock);
        return res;
    }

    Tabl_elem cur_min;
    cur_min.x = 0;
    cur_min.up = _b;
    cur_min.min = Recursive_Search_Min(_b, _m - 1);

    typeElem x_one_level = Search_Min_One_Level(_b, _m);
    if (x_one_level != d && x_one_level != 0) {
        typeElem minOneLevel = pow(abs(x_one_level), p);
        if (minOneLevel <= cur_min.min || cur_min.min == -1) {
            cur_min.x = x_one_level;
            cur_min.up = -2;
            cur_min.min = minOneLevel;
        }
    }

    typeElem delta2 = (d / 2) + 1;
    cur_min = Search_Min_Left_Right(_b, _m, cur_min, delta2, 1);
    cur_min = Search_Min_Left_Right(_b, _m, cur_min, delta2, -1);
    if (cur_min.min == -1) {
        return -1;
    }
    
    Table[_b][_m].x = cur_min.x;
    Table[_b][_m].up = cur_min.up;
    Table[_b][_m].min = cur_min.min;

    typeElem res = Table[_b][_m].min;
    omp_unset_lock(&Table[_b][_m].lock);
    return res;
}



typeElem svp::Search_Min_One_Level(typeElem _b, int _m) {


    int fl = 0;
    typeElem res = 1, sum_p = 0, sum_b = 0;
    for (int i = rankE; i < n; i++) {
        if (tP[_m][i] != mod_S(_b, S[i])) {
            fl = 1;
            break;
        }
    }
    if (fl == 0) {
        return 1;
    }


    for (int i = rankE; i < n; i++) {
        typeElem b_mod_S = mod_S(_b, S[i]);
        if (b_mod_S != 0 && tP[_m][i] == 0) {
            return d;
        }
        typeElem k = S[n - 1] / S[i];
        sum_p = mod_S(sum_p + tP[_m][i] * k, S[n - 1]);
        sum_b = mod_S(sum_b + b_mod_S * k, S[n - 1]);
    }


    typeElem d, x, y;
    d = Euclid(sum_p, S[n - 1], x, y);
    res = sum_b * x;
    return res;
}



Tabl_elem svp::Search_Min_Left_Right(typeElem _b, int _m, Tabl_elem _min_elem, typeElem _delta_2, int _side) {

    for (typeElem l = 1; l < _delta_2; l++) {
        vector<typeElem> test_real_b(n, 0);
        vector<typeElem> new_b(n, 0);
        typeElem b_Px = -1;

        for (int i = rankE; i < n; i++) {
            new_b[i] = mod_S(_b - l * _side * tP[_m][i], S[i]);
        }
        for (int i = rankE; i < n; i++) {
            test_real_b[i] = mod_S(new_b[n - 1], S[i]);
        }

        for (int i = rankE; i < n; i++) {
            if (new_b[i] != test_real_b[i]) {
                b_Px = -2;
                break;
            }
        }
        if (b_Px == -2) {
            continue;
        }
        b_Px = new_b[n - 1];
        if (b_Px == _b) {
            break;
        }

        typeElem cur_min = Recursive_Search_Min(b_Px, _m - 1);
        if (cur_min == -1) {
            continue;
        }
        cur_min += static_cast<typeElem>(pow(l * _side, p));
        if (cur_min < _min_elem.min || _min_elem.min == -1) {
            _min_elem.x = l * _side;
            _min_elem.up = b_Px;
            _min_elem.min = cur_min;
        }
    }
    return _min_elem;
}
