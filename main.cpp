#include <iostream>
#include <chrono>
#include <ctime>
#include "svp.h"

void RunTest() {
    vector<vector<typeElem>> matrix = { {1, 0, 0},
                                        {0, 1, 0},
                                        {3, 7, 11} };
    cout << "matrix: " << endl;
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix.size(); j++) {
            cout << matrix[i][j] << "   ";
        }
        cout << endl;
    }

    svp lattice(matrix);
    int num_threads = 2;

    auto start = std::chrono::steady_clock::now();

    vector<typeElem> result = lattice.Start_SVP(num_threads);

    auto finish = std::chrono::steady_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << "The time: " << time_ms.count() << " ms\n";

    cout << "vector_SVP: " << endl;
    for (int j = 0; j < result.size(); j++) {
        cout << result[j] << "   ";
    }
    cout << endl;
}

int main() {
    srand(time(0));
    RunTest();
    return 0;
}
