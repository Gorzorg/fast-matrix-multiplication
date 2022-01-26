#include "Matrices.h"
#include <iostream>
#include <fstream>
#include <chrono>

int main() {

	using namespace std::chrono;

	std::ofstream output_file;
	output_file.open("matrix_mul_performance_test_log.txt");
	int size0, size1, size2;
	int number_of_test_repetitions = 10;

	matrix<int> M1, M2, M3, M4;

	//we start with the most basic test:
	// can we multiply two matrices with naive_matrix_multiplication
	// and with basic_Strassen_matrix_multiplication
	// and obtain the same output?

	M1 = matrix<int>( 52, 79 );
	M2 = matrix<int>(79, 31);
	int counter = 0;
	for (int i = 0; i < 52; i++) {
		for (int j = 0; j < 79; j++) {
			M1[i][j] = (counter++) % 17;
		}
	}
	for (int j = 0; j < 79; j++) {
		for (int k = 0; k < 31; k++) {
			M2[j][k] = (counter++) % 13;
		}
	}
	M3 = naive_matrix_multiplication(M1, M2);

	M4 = basic_Strassen_matrix_multiplication(M1, M2, 2);

	if (M3 == M4) {
		std::cout << "consistency check successful, proceeding to performance test" << std::endl;
	}
	else {
		std::cout << "consistency check failed, aborting" << std::endl;
		throw;
	}

	for (int dim = 64; dim < 256; dim++) {
		std::cout << dim << std::endl;
		size0 = dim;
		size1 = dim;
		size2 = dim;

		M1 = matrix<int>(size0, size1, 0);
		M2 = matrix<int>(size1, size2, 0);
		counter = 0;
		for (int i = 0; i < size0; i++) {
			for (int j = 0; j < size1; j++) {
				M1[i][j] = counter++ % 10;
			}
		}
		for (int i = 0; i < size1; i++) {
			for (int j = 0; j < size2; j++) {
				M2[i][j] = counter++ % 10;
			}
		}

		steady_clock::time_point crono_start = steady_clock::now();
		for (int i = 0; i < number_of_test_repetitions; i++) {
			M3 = naive_matrix_multiplication(M1, M2);
		}
		steady_clock::time_point crono_end = steady_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(crono_end - crono_start);

		output_file << "sizes = [" << size0 << ", " << size1 << ", " << size2 <<
			"], naive_mul_time = " << time_span.count() / number_of_test_repetitions << std::endl;


		for (int switch_size = 32; switch_size <= 64; switch_size++) {
			crono_start = steady_clock::now();
			for (int i = 0; i < number_of_test_repetitions; i++) {
				M4 = basic_Strassen_matrix_multiplication(M1, M2, 32);
			}
			crono_end = steady_clock::now();
			time_span = duration_cast<duration<double>>(crono_end - crono_start);
			output_file << "sizes = [" << size0 << ", " << size1 << ", " << size2 <<
				"], switch_size = " << switch_size <<
				", Strassen_mul_time = " << time_span.count() / number_of_test_repetitions << std::endl;
		}

		if (!(M3 == M4)) {
			std::cout << "consistency test failed, aborting" << std::endl;
			output_file.close();
			throw;
		}
	}
	output_file.close();
	return 0;
}