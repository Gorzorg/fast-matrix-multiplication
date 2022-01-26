#pragma once

#include<vector>
#include<stdexcept>
#include<iostream>

//this header defines a way to represent matrices, and some basic operations for them.
// the main feature is the implementation of the Strassen algorithm for matrix multiplication
// which allows to execute matrix multiplication with less than O(size^3) operations,
// when size -> infinity. In the demo file, I tested the performance of naive_matrix_multiplication
// vs. the performance of basic_Strassen_matrix_multiplication on square matrices.
// the results on my machine tell that the Strassen algorithm is already convenient for multiplying
// 64 x 64 matrices, and spares more or less 30% of computation time when applied to 1024 x 1024
// matrices. These benchmarks are most likely machine dependent, sodo not take them for granted.
// I called my implementation basic_Strassen_matrix_multiplication, since many variations of the
//algorithm exist, and some of which further improve performance on non-square matrices.
// If I have spare time and will to do so, in the future I might implement some of those variations.

template<typename T>
using vecvec = std::vector<std::vector<T> >;

template<typename T>
class matrix;

class submatrix_index_delimiters{
public:
	size_t begin0;
	size_t begin1;
	size_t end0;
	size_t end1;
	__forceinline submatrix_index_delimiters();
	__forceinline submatrix_index_delimiters(size_t _begin0, size_t _begin1, size_t _end0, size_t _end1);
	__forceinline submatrix_index_delimiters(size_t params[4]);
	template<typename T>
	__forceinline submatrix_index_delimiters(matrix<T> M);
	__forceinline size_t size0();
	__forceinline size_t size1();
	__forceinline size_t& operator[](size_t x);
	__forceinline submatrix_index_delimiters shift_to_origin();
};



template<typename T>
class matrix {
public:

	size_t sizes[2];
	std::shared_ptr<vecvec<T>> entries;

	//~matrix() {
	//	delete entries
	//}

	__forceinline matrix() {
		sizes[0] = 0;
		sizes[1] = 0;
		entries = std::shared_ptr<vecvec<T> >(nullptr);
	}

	__forceinline matrix(size_t _sizes[2]) {
		sizes[0] = _sizes[0];
		sizes[1] = _sizes[1];
		entries = std::make_shared<vecvec<T> >(sizes[0]);
		for (size_t i = 0; i < sizes[0]; i++) {
			(*entries)[i] = std::vector<T>(sizes[1]);
		}
	}

	__forceinline matrix(size_t size0, size_t size1) {
		sizes[0] = size0;;
		sizes[1] = size1;
		entries = std::make_shared<vecvec<T> >(sizes[0]);
		for (size_t i = 0; i < sizes[0]; i++) {
			(*entries)[i] = std::vector<T>(sizes[1]);
		}
	}

	__forceinline matrix(const size_t _sizes[2],const T value) {

		sizes[0] = _sizes[0];
		sizes[1] = _sizes[1];
		entries = std::make_shared<vecvec<T> >(sizes[0], std::vector<T>(sizes[1], value));
	}

	__forceinline matrix(size_t size0, size_t size1, T value) {
		sizes[0] = size0;
		sizes[1] = size1;
		entries = std::make_shared<vecvec<T> >(sizes[0], std::vector<T>(sizes[1], value));
	}

	__forceinline matrix(size_t _sizes[2], vecvec<T>* const _entries) {
		sizes[0] = _sizes[0];
		sizes[1] = _sizes[1];
		entries = std::make_shared<vecvec<T> >(_entries);
	}

	__forceinline matrix(size_t _sizes[2], std::shared_ptr<vecvec<T> > _entries) {
		sizes[0] = _sizes[0];
		sizes[1] = _sizes[1];
		entries = _entries;
	}

	matrix(size_t _sizes[2], std::vector<T>& _entries) {
		sizes[0] = _sizes[0];
		sizes[1] = _sizes[1];
		if (_entries.size() < sizes[0] * sizes[1]) {
			throw std::invalid_argument("not enough data to fill the matrix");
		}
		entries = std::make_shared<vecvec<T>>();
		entries->reserve(sizes[0]);
		typename std::vector<T>::iterator begin = _entries.begin();
		for (size_t i = 0; i < sizes[0]; i++) {
			entries->push_back(std::vector<T>(begin + sizes[1] * i, begin + sizes[1] * (i + 1)));
		}
	}

	__forceinline matrix(const matrix<T>& M) {
		sizes[0] = M.sizes[0];
		sizes[1] = M.sizes[1];
		entries = M.entries;
	}

	__forceinline matrix<T> copy() const {
		vecvec<T>* copy_entries = new vecvec<T>(this->entries);//should perform a deep copy of *M.entries.
		return matrix<T>(this->sizes, copy_entries);
	}

	__forceinline void crop(size_t new_size0, size_t new_size1) {
		sizes[0] = std::min(sizes[0], new_size0);
		entries->resize(sizes[0]);
		sizes[1] = std::min(sizes[1], new_size1);
		for (int i = 0; i < sizes[0]; i++) {
			(*entries)[i].resize(sizes[1]);
		}
	}

	__forceinline void expand(size_t new_size0, size_t new_size1, T value) {
		sizes[0] = std::max(sizes[0], new_size0);
		sizes[1] = std::max(sizes[1], new_size1);
		entries->resize(sizes[0], std::vector<T>(sizes[1], value));
		for (size_t i = 0; i < sizes[0]; i++) {
			(*entries)[i].resize(sizes[1], value);
		}
	}

	bool has_enough_entries() {
		if (sizes[0] > entries->size()) {
			return false;
		}
		for (size_t i = 0; i < sizes[1]; i++) {
			if (sizes[1] > (*entries)[i].size()) {
				return false;
			}
		}
		return true;
	}

	__forceinline std::vector<T>& operator [] (size_t x) const {
		return (*entries)[x];
	}

	__forceinline matrix<T> submatrix(submatrix_index_delimiters delim) const {
		// given a matrix M, and a delimiter delim, it returns a matrix corresponding to
		// M[delim.begin0 : delim.end0][delim.begin1 : delim.end1]
		//
		// for more flexibility in submatrix construction, use the other submatrix method.
		size_t subm_sizes[2];
		subm_sizes[0] = delim.size0();
		subm_sizes[1] = delim.size1();
		
		std::shared_ptr<vecvec<T> > subm_entries = std::make_shared<vecvec<T> >(subm_sizes[0]);
		for (size_t i = 0; i < subm_sizes[0]; i++) {
			(*subm_entries)[i] = std::vector<T>(subm_sizes[1]);
		}
		std::vector<T>* current_row;
		std::vector<T>* current_subm_row;

		for (size_t i = delim.begin0, subm_i = 0; i < delim.end0; i++, subm_i++) {
			current_row =  & (*entries)[i];
			current_subm_row = &(*subm_entries)[subm_i];

			for (size_t j = delim.begin1, subm_j = 0; j < delim.end1; j++, subm_j++) {
				(*current_subm_row)[subm_j] = (*current_row)[j];
			}
		}
		matrix<T> R= matrix<T>(subm_sizes, subm_entries);
		return R;
	}

	__forceinline matrix<T> safe_submatrix(submatrix_index_delimiters delim) {
		if (delim.end0 <= delim.begin0 || delim.end1 <= delim.begin1
			|| delim.end0 > sizes[0] || delim.end1 > sizes[1] || entries->size() < delim.end0) {
			throw std::invalid_argument("invalid delimiters");
		}
		if (has_enough_entries()) {
			return submatrix(delim);
		}
		else {
			throw;
		}
	}

	template <typename ContainerType>
	matrix<T> submatrix(const ContainerType& indexes0, const ContainerType& indexes1) const {
		//this function returns, given a matrix, and a couple of iterable containers,
		// a sub matrix specified by the containers. One should note that we deliberately chose
		// not to check for repetitions in the containers. While this allows for return values
		// which are not proper submatrices, it also allows to create all sorts of repetition
		// patterns based on the matrix M (e.g. expanding M periodically).

		//we compute sizes[1] in advance: if it is 0, we can avoid a lot of empty loops,
		//and we have to compute it anyways sooner or later.
		int subm_sizes[2] = { 0,0 };
		auto begin1 = indexes1.begin();
		auto end1 = indexes1.end();
		for (; begin1 != end1; begin1++) {
			subm_sizes[1]++;
		}
		if (subm_sizes[1] == 0) {
			throw std::invalid_argument("second index container is empty");
		}

		vecvec<T> subm_entries = new vecvec<T>();
		std::vector<T>* new_row;

		auto begin0 = indexes0.begin();
		auto end0 = indexes0.end();

		for (; begin0 != end0; begin0++) {
			subm_sizes[0]++;//this line is here to compute size0.
			// if no cycles of this loop are executed, size0 will stay 0, and we will throw an exception
			try {
				//we try allocating the memory for the new line
				subm_entries->push_back(std::vector<T>());
				new_row = &subm_entries->back();
				new_row->reserve(subm_sizes[1]);

				//and then we populate the new line with the appropriate elements.
				auto begin1 = indexes1.begin();
				auto end1 = indexes1.end();
				for (; begin1 != end1; begin1++) {
					new_row->push_back((entries->at(*begin0)).at(*begin1));
				}
			}
			catch (...) {
				delete subm_entries;
				throw;
			}
		}//when a cycle of this loop ends, we move to the next line.

		if (subm_sizes[0] == 0) {//similarly to before, we throw if the first dimension is 0.
			delete subm_entries;//with the difference that now we deallocate subm_entries
			throw std::invalid_argument("first index container is empty");
		}

		//now we have all the necessary information to create the submatrix.
		return matrix<T>(subm_sizes, subm_entries);
	}
	template<bool subtract_instead_of_adding>
	void zero_pad_sum(const matrix<T> M, size_t offset0 = 0, size_t offset1 = 0) {
		// adds to "this" a matrix M which may have incompatible dimensions.
		// if needed, zeros are added as a placeholder for non-existent entries.
		//
		// M will be (virtually) padded with offset0 (T)0 lines, and with offset1 (T)0 columns
		// before the sum takes place.
		// 
		// if some memory allocation issues are encountered, the entries of this are not modified.
		if (offset0 < 0 || offset1 < 0) {
			throw std::invalid_argument("offsets must be non-negative");
		}
		size_t nr_rows = std::max( M.sizes[0] + offset0 , sizes[0]);
		size_t row_length = std::max( M.sizes[1] + offset1 , sizes[1]);

		std::vector<T>* current_row;
		std::vector<T>* current_M_row;
		size_t i, j, M_i, M_j;
		try {
			if (nr_rows != sizes[0]) {
				//if M (taking the offset into account) has more rows than this, append some rows to M
				i = sizes[0];
				for (; i < offset0; i++) {
					// if offset0 > sizes[0], then add some (T)0 lines
					entries->push_back(std::vector<T>(row_length, (T)0));
				}
				for (M_i = 0; i < nr_rows; i++, M_i++) {
					//then, add the appropriate non-(T)0 lines whose entries are taken from M
					entries->push_back(std::vector<T>());
					current_row = & (*entries)[i];
					current_M_row = & M[M_i];
					current_row->reserve(row_length);

					for (j = 0; j < offset0; j++) {
						(*current_row)[j] = (T)0;
					}
					for (M_j = 0; M_j < M.sizes[1]; j++, M_j++) {
						if (subtract_instead_of_adding) {
							(*current_row)[j] = -(*current_M_row)[M_j];
						}
						else {
							(*current_row)[j] = (*current_M_row)[M_j];
						}
					}
					for (; j < row_length; j++) {
						(*current_row)[j] = (T)0;
					}
				}
			}
			if (row_length != sizes[1]) {
				// if the rows of M are longer than the rows of this, allocate memory
				// in the rows of this to match the length.
				//as before, expand the lines using either (T)0, or the entries of M when appropriate.
				for (i = 0; i < offset0; i++) {
					(*entries)[i].resize(row_length, (T)0);
				}
				for (M_i = 0; M_i < M.sizes[0]; i++, M_i++) {
					current_row = &(*entries)[i];
					current_M_row = &M[M_i];
					current_row->resize(row_length);
					for (j = sizes[1]; j < offset1; j++) {
						(*current_row)[j] = (T)0;
					}
					for (M_j = 0; j < row_length; j++, M_j++) {
						(*current_row)[j] = (*current_M_row)[M_j];
					}
				}
				for (; i < sizes[0]; i++) {
					(*entries)[i].resize(row_length, (T)0);
				}
			}
		}
		catch (...) {
			this->crop(sizes[0], sizes[1]);
			throw;
		}
		//if no exception occurred, by this point this has been expanded, and we perform the sum.
		i = offset0;
		for (M_i = 0; M_i < M.sizes[0]; i++, M_i++) {
			j = offset1;
			current_row = &(*entries)[i];
			current_M_row = &M[M_i];
			for (M_j = 0; M_j < M.sizes[1]; j++, M_j++) {
				if (subtract_instead_of_adding) {
					(*current_row)[j] -= (*current_M_row)[M_j];
				}
				else {
					(*current_row)[j] += (*current_M_row)[M_j];
				}
			}
		}
		sizes[0] = nr_rows;
		sizes[1] = row_length;
	}

	matrix<T>& operator += (const matrix<T>& rhs) {
		if (this->sizes[0] != rhs.sizes[0] || this->sizes[1] != rhs.sizes[1]) {
			throw std::invalid_argument("matrix dimensions are not consistent for +");
		}

		for (int i = 0; i < sizes[0]; i++) {
			for (int j = 0; j < sizes[1]; j++) {
				(*entries)[i][j] += (*rhs.entries)[i][j];
			}
		}
		return *this;
	}

	matrix<T>& operator -= (const matrix<T>& rhs) {
		if (sizes[0] != rhs.sizes[0] || sizes[1] != rhs.sizes[1]) {
			throw std::invalid_argument("matrix dimensions are not consistent for -");
		}
		
		for (int i = 0; i < sizes[0]; i++) {
			for (int j = 0; j < sizes[1]; j++) {
				(*entries)[i][j] -= (*rhs.entries)[i][j];
			}
		}
		return *this;
	}

	bool operator == (const matrix<T>& rhs) const {
		std::vector<T>* current_line;
		std::vector<T>* current_rhs_line;
		if (sizes[0] != rhs.sizes[0] || sizes[1] != rhs.sizes[1]) {
			return false;
		}
		for (size_t i = 0; i < sizes[0]; i++) {
			current_line = &(*entries)[i];
			current_rhs_line = &rhs[i];
			for (size_t j = 0; j < sizes[1]; j++) {
				if ((*current_line)[j] != (*current_rhs_line)[j]) {
					return false;
				}
			}
		}
		return true;
	}
};
	

__forceinline submatrix_index_delimiters::submatrix_index_delimiters() {
	begin0 = 0;
	begin1 = 0;
	end0 = 0;
	end1 = 0;
};

__forceinline submatrix_index_delimiters::submatrix_index_delimiters(size_t _begin0, size_t _begin1, size_t _end0, size_t _end1) {
		begin0 = _begin0;
		begin1 = _begin1;
		end0 = _end0;
		end1 = _end1;
	}

__forceinline submatrix_index_delimiters::submatrix_index_delimiters(size_t params[4]) {
		begin0 = params[0];
		begin1 = params[1];
		end0 = params[2];
		end1 = params[3];
	}

template<typename T>
__forceinline submatrix_index_delimiters::submatrix_index_delimiters(matrix<T> M) {
		begin0 = 0;
		begin1 = 0;
		end0 = M.sizes[0];
		end1 = M.sizes[1];
	}

__forceinline size_t submatrix_index_delimiters::size0() {
	return end0 - begin0;
}

__forceinline size_t submatrix_index_delimiters::size1() {
	return end1 - begin1;
}

__forceinline size_t& submatrix_index_delimiters::operator[](size_t x) {
	switch (x) {
	case 0:
		return begin0;
	case 1:
		return begin1;
	case 2:
		return end0;
	case 3:
		return end1;
	default:
		throw std::invalid_argument("index out of range");
	}
}

std::ostream& operator << (std::ostream& os, submatrix_index_delimiters D) {
	os << "((" << D.begin0 << ", " << D.begin1 << "), (" << D.end0 << ", " << D.end1 << "))";
	return os;
}

__forceinline submatrix_index_delimiters submatrix_index_delimiters::shift_to_origin() {
	return submatrix_index_delimiters(0, 0, end0 - begin0, end1 - begin1);
}

vecvec<size_t> get_roughly_square_chunk_delimiters(size_t a, size_t b, size_t c) {
	//given a matrix M, the goal of this function is to return the information on
	//how to partition it into matrices with sizes[0]/sizes[1] close to 1.
	// the parameter min_size tells that we want, if possible, the resulting chunks to be
	// bigger than min_size x min_size.
	//
	//in order to keep things simple, i decided to cut the matrix in only one direction.
	// the result is an easily optimizeable problem, which grants that for the resulting chunks
	// will hold max( (sizes[0] -1) / sizes[1] , (sizes[1] -1) / sizes[0] ) < sqrt(2).
	// 
	// keep in mind that, if the matrix is too small, min_size alters the procedure.
	vecvec<size_t> chunk_delimiters(3, std::vector<size_t>());

	auto best_split = [](size_t big, size_t small) {
		//given two int numbers big and small, such that big >= small,
		// it returns the int number n which minimizes the quantity
		// max( big/(n*small), (n*small)/big )
		size_t n = big / small;
		if (big * big > n * (n + 1) * small * small) {
			n += 1;
		}
		return n;
	};

	auto create_split = [](std::vector<size_t>& delimiters, size_t quantity, size_t number_of_pieces) {
		for (int i = 0; i <= number_of_pieces; i++) {
			delimiters.push_back(quantity * i / number_of_pieces);
		}
	};

	size_t n;

	if (b > a || b > c) {
		size_t max_ac = (a > c) ? a : c;
		size_t min_ac = (a > c) ? c : a;
		n = best_split(b, max_ac);
		if (b * b > 2 * n * n * min_ac * min_ac) {
			n = best_split(b, min_ac);
		}
		create_split(chunk_delimiters[1], b, n);
		if (a * n > b) {
			create_split(chunk_delimiters[0], a, best_split(a * n, b));
		}
		else {
			chunk_delimiters[0].push_back(0);
			chunk_delimiters[0].push_back(a);
		}
		if (c * n > b) {
			create_split(chunk_delimiters[2], c, best_split(c * n, b));
		}
		else {
			chunk_delimiters[2].push_back(0);
			chunk_delimiters[2].push_back(c);
		}
	}
	else {
		chunk_delimiters[1].push_back(0);
		chunk_delimiters[1].push_back(b);
		create_split(chunk_delimiters[0], a, best_split(a, b));
		create_split(chunk_delimiters[2], c, best_split(c, b));
	}

	return chunk_delimiters;
}

template<typename T, bool subtract_instead_of_adding>
matrix<T> zero_pad_sum(const matrix<T> A, const matrix<T> B,
	submatrix_index_delimiters delimitersA, submatrix_index_delimiters delimitersB){
	
	size_t max_sizes[2], min_sizes[2];
	max_sizes[0] = std::max(delimitersA.size0(), delimitersB.size0());
	max_sizes[1] = std::max(delimitersA.size1(), delimitersB.size1());
	min_sizes[0] = std::min(delimitersA.size0(), delimitersB.size0());
	min_sizes[1] = std::min(delimitersA.size1(), delimitersB.size1());

	matrix<T> R(max_sizes);

	size_t R_i, R_j, A_i, A_j, B_i, B_j;
	std::vector<T>* current_A_row;
	std::vector<T>* current_B_row;
	std::vector<T>* current_R_row;

	A_i = delimitersA.begin0;
	B_i = delimitersB.begin0;
	for (R_i = 0; R_i <min_sizes[0]; R_i++, A_i++, B_i++) {
		current_A_row = &A[A_i];
		current_B_row = &B[B_i];
		current_R_row = &R[R_i];
		A_j = delimitersA.begin1;
		B_j = delimitersB.begin1;
		for (R_j = 0; R_j < min_sizes[1]; R_j++, A_j++, B_j++) {
			//as far as i understand, since the compiler knows the template argument
			//at compile time, it should optimize the if/else statement out
			// so we can improve performance, whitout commuting the if statement with the for statement
			if (subtract_instead_of_adding) {
				(*current_R_row)[R_j] = (*current_A_row)[A_j] - (*current_B_row)[B_j];
			}
			else {
				(*current_R_row)[R_j] = (*current_A_row)[A_j] + (*current_B_row)[B_j];
			}
		}
		//I assume one does not know which matrix is bigger at compile time
		// so in this case I accept some code duplication in order to avoid
		// checking the same condition for every loop repetition.
		if (delimitersB.size1() == min_sizes[1]) {
			for (; R_j < max_sizes[1]; R_j++, A_j++) {
				(*current_R_row)[R_j] = (*current_A_row)[A_j];
			}
		}
		else {
			for (; R_j < max_sizes[1]; R_j++, B_j++) {
				if (subtract_instead_of_adding) {
					(*current_R_row)[R_j] = - (*current_B_row)[B_j];
				}
				else {
					(*current_R_row)[R_j] = (*current_B_row)[B_j];
				}
			}
		}
	}

	//likewise, here I assume I do not know which matrix is bigger at compile time,
	//so I check the condition outside the loop, and I duplicate some code.
	if (delimitersB.size0() == min_sizes[0]) {
		for (; R_i < max_sizes[0]; R_i++, A_i++) {
			current_A_row = &A[A_i];
			current_R_row = &R[R_i];
			R_j = 0;
			for (A_j = delimitersA.begin1; A_j < delimitersA.end1; A_j++, R_j++) {
				(*current_R_row)[R_j] = (*current_A_row)[A_j];
			}
			for (; R_j < max_sizes[1]; R_j++) {
				(*current_R_row)[R_j] = 0;
			}
		}
	}
	else {
		for (; R_i < max_sizes[0]; R_i++, B_i++) {
			current_B_row = &B[B_i];
			current_R_row = &R[R_i];
			R_j = 0;
			for (B_j = delimitersB.begin1; B_j < delimitersB.end1; B_j++, R_j++) {
				if (subtract_instead_of_adding) {
					(*current_R_row)[R_j] = -(*current_B_row)[B_j];
				}
				else {
					(*current_R_row)[R_j] = (*current_B_row)[B_j];
				}
			}
			for (; R_j < max_sizes[1]; R_j++) {
				(*current_R_row)[R_j] = 0;
			}
		}
	}
	return R;
}

template <typename T>
matrix<T> safe_zero_pad_sum(const matrix<T> A, const matrix<T> B,
	submatrix_index_delimiters delimitersA, submatrix_index_delimiters delimitersB,
	bool subtract_instead_of_adding = false) {
	for (size_t i = 0; i < 2; i++) {
		if (delimitersA[i] < 0 || delimitersA[i + 2] < delimitersA[i] ||
			delimitersB[i] < 0 || delimitersB[i + 2] < delimitersB[i]) {
			throw std::invalid_argument("invalid delimiters");
		}
	}
	if (subtract_instead_of_adding) {
		return zero_pad_sum<T, true>(A, B, delimitersA, delimitersB);
	}
	else {
		return zero_pad_sum<T, false>(A, B, delimitersA, delimitersB);
	}
}

template<typename T>
std::ostream& operator << (std::ostream& os, matrix<T> M) {
	
	auto print_line = [](std::ostream& os, std::vector<T>& line, size_t line_length) {
		size_t line_length_minus_1 = line_length - 1;
		size_t j = 0;
		for (; j < line_length_minus_1; j++) {
			os << line[j] << ", ";
		}
		os << line[j];
	};

	os << "[";
	size_t number_of_lines_minus_1 = M.sizes[0] - 1;
	size_t i = 0;

	for (; i < number_of_lines_minus_1; i++) {
		print_line(os, M[i], M.sizes[1]);
		os << std::endl;
	}
	print_line(os, M[i], M.sizes[1]);
	return os << " ]";
}

template <typename T>
inline matrix<T> operator +(matrix<T> M1, const matrix<T>& M2) {
	matrix<T> R = M1.copy();
	R += M2;
	return R;
}

template <typename T>
inline matrix<T> operator -(matrix<T> M1, const matrix<T>& M2) {
	matrix<T> R = M1.copy();
	R -= M2;
	return R;
}

template<typename T>
matrix<T> naive_matrix_multiplication(const matrix<T> M1, const matrix<T> M2) {
	if (M1.sizes[1] != M2.sizes[0]) {
		throw std::invalid_argument("matrix dimensions not consistent with *");
	}
	size_t middle_size = M1.sizes[1];
	size_t sizes[2];
	sizes[0] = M1.sizes[0];
	sizes[1] = M2.sizes[1];
	matrix<T> R(sizes, (T)0);
	T temp;
	for (size_t i = 0; i < sizes[0]; i++) {
		for (size_t j = 0; j < sizes[1]; j++) {
			temp = (T)0;
			for (size_t k = 0; k < middle_size; k++) {
				temp += M1[i][k] * M2[k][j];
			}
			R[i][j] = temp;
		}
	}
	return R;
}

template<typename T>
matrix<T> naive_matrix_multiplication(const matrix<T> A, const matrix<T> B,
	submatrix_index_delimiters delimitersA, submatrix_index_delimiters delimitersB){
//	int offsetA0, int offsetA1, int offsetB0, int offsetB1, int size0, int common_size, int size1) {
	//like the normal naive matrix multiplication, but for multiplying submatrices.
	// the basic version naive_matrix_multiplication(X,Y) is equivalent to
	// naive_matrix_multiplication(X, Y, {0, 0, A.sizes[0], A.sizes[1]}, {0, 0, B.sizes[0], B.sizes[1]})
	// 
	// to be precise, the matrices we are going to multiply are
	// A[delimitersA[0] : delimitersA[2]][delimitersA[1] : delimitersA[3]] and
	// B[delimitersB[0] : delimitersB[2]][delimitersB[1] : delimitersB[3]]

	size_t sizes[2];
	sizes[0] = delimitersA.size0();
	sizes[1] = delimitersB.size1();

	if (sizes[0] <= 0 || sizes[1] <= 0 || delimitersA.end1 <= delimitersA.begin1
		|| delimitersA.end1 - delimitersA.begin1 != delimitersB.end0 -delimitersB.begin0 ) {
		throw std::invalid_argument("invalid matrix sizes");
	}
	
	if (A.sizes[0] < delimitersA.end0 || A.sizes[1] < delimitersA.end1
		|| B.sizes[0] < delimitersB.end0 || B.sizes[1] < delimitersB.end1) {
		throw std::invalid_argument("index out of range");
	}

	matrix<T> R(sizes);
	size_t i0, i1, j0, j1;
	std::vector<T>* current_A_row;

	i0 = delimitersA.begin0;
	for (size_t i = 0; i < sizes[0]; i++, i0++) {
		current_A_row = & A[i0];

		j1 = delimitersB.begin1;
		for (size_t j = 0; j < sizes[1]; j++, j1++) {
			i1 = delimitersA.begin1;
			j0 = delimitersB.begin0;
			T temp = (T)0;
			for (; i1 < delimitersA.end1; i1++, j0++) {
				temp += (*current_A_row)[i1] * B[j0][j1];
			}
			R[i][j] = temp;
		}
	}
	return R;
}

template <typename T>
matrix<T> basic_Strassen_matrix_multiplication(const matrix<T> M1, const matrix<T> M2, int switch_size) {
	// implementation of the basic Strassen matrix multiplication algorithm.
	// custom implementation details include the chunking of matrices before multiplication, and
	// the switch to naive matrix multiplication under certain sizes.
	if (M1.sizes[1] != M2.sizes[0]) {
		throw std::invalid_argument("matrix dimensions not consistent with *");
	}
	size_t sizes[2];
	sizes[0] = M1.sizes[0];
	sizes[1] = M2.sizes[1];
	vecvec<size_t> delimiters = get_roughly_square_chunk_delimiters(sizes[0], M1.sizes[1], sizes[1]);

	size_t chunks_along_size0 = delimiters[0].size() -1;
	size_t chunks_along_common_size = delimiters[1].size() -1;
	size_t chunks_along_size1 = delimiters[2].size() -1;
	matrix<T> R(sizes, (T)0);

	//now i will do block matrix multiplication using the naive algorithm, but every
	// matrix multiplication at block level will be handled by the Strassen algorithm.
	for (size_t i = 0; i < chunks_along_size0; i++) {
		for (size_t j = 0; j < chunks_along_common_size; j++) {
			for (size_t k = 0; k < chunks_along_size1; k++) {
				submatrix_index_delimiters chunkM1(delimiters[0][i], delimiters[1][j],
					delimiters[0][i + 1], delimiters[1][j + 1]);
				submatrix_index_delimiters chunkM2(delimiters[1][j], delimiters[2][k],
					delimiters[1][j + 1], delimiters[2][k + 1]);
//				std::cout << "multiplying chunks " <<std::endl << chunkM1 << std::endl << chunkM2 << std::endl << std::endl;
				R.zero_pad_sum<false>(basic_Strassen_recursion(M1, M2, switch_size, chunkM1, chunkM2),
					delimiters[0][i], delimiters[2][k]);
//				std::cout << "i, j, k  = " << i << ", " << j << ", " << k << std::endl;
//				std::cout << "R = " << R << std::endl;
			}
		}
	}
	return R;
}

template <typename T>
matrix<T> basic_Strassen_recursion(const matrix<T> A, const matrix<T> B, int switch_size,
	submatrix_index_delimiters delimitersA, submatrix_index_delimiters delimitersB){
	
	//returns the result of the multiplication of
	// A[offsetA0 : offsetA0 + size0][offsetA1 : offsetA1 + common_size] and
	// B[offsetB0 : offsetB0 + common_size][offsetB1 : offsetB1 + size1]

	size_t sizes[3], upper_mid_sizes[3], lower_mid_sizes[3];
	sizes[0] = delimitersA.size0();
	sizes[1] = delimitersA.size1();
	sizes[2] = delimitersB.size1();
	for (size_t i = 0; i < 3; i++) {
		upper_mid_sizes[i] = (sizes[i] + 1) / 2;
		lower_mid_sizes[i] = sizes[i] / 2;
	}

	if (switch_size * (sizes[0] + sizes[2]) > sizes[0] * sizes[2]) {
		return naive_matrix_multiplication(A, B, delimitersA, delimitersB);
	}

	size_t delimiters_1D[4][3];
	for (size_t i = 0; i < 4; i++) {
		for (size_t j = 0; j < 3; j++) {
			switch (i) {
			case 0:
				delimiters_1D[i][j] = ((2 - j) * delimitersA[0] + j * delimitersA[2] + 1) / 2;
				break;
			case 1:
				delimiters_1D[i][j] = ((2 - j) * delimitersA[1] + j * delimitersA[3] + 1) / 2;
				break;
			case 2:
				delimiters_1D[i][j] = ((2 - j) * delimitersB[0] + j * delimitersB[2] + 1) / 2;
				break;
			case 3:
				delimiters_1D[i][j] = ((2 - j) * delimitersB[1] + j * delimitersB[3] + 1) / 2;
				break;
			}
		}
	}

	//in the following block of code, we encode the delimiters needed to divide A and B into 2x2
	// blocks of roughly equal sizes. if A = [A11,A12;A21,A22], and one or more of the sizes of A are odd,
	// then the condition ( A11.sizes[0] >= A22.sizes[0] && A11.sizes[1] >= A22.sizes[1] ) applies.
	submatrix_index_delimiters chunksA[2][2], chunksB[2][2];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				for (int l = 0; l < 2; l++) {
					chunksA[i][j][2 * k + l] = delimiters_1D[    l][((l == 0) ? i : j) + k];
					chunksB[i][j][2 * k + l] = delimiters_1D[2 + l][((l == 0) ? i : j) + k];
				}
			}
//			std::cout << "chunkA[" << i << "][" << j << "] = " << chunksA[i][j] << std::endl;
//			std::cout << "chunkB[" << i << "][" << j << "] = " << chunksB[i][j] << std::endl;
		}
	}

	matrix<T> R = matrix<T>(sizes[0], sizes[2], (T)0);
	matrix<T> temp;

	//compute temp = (A11 + A22) * (B11 + B22)
	temp = basic_Strassen_recursion(zero_pad_sum<T,false>(A, A, chunksA[0][0], chunksA[1][1]),
		zero_pad_sum<T,false>(B, B, chunksB[0][0], chunksB[1][1]), switch_size,
		chunksA[0][0].shift_to_origin(), chunksB[0][0].shift_to_origin());

	// splitting R into blocks analogously to A and B, we now perform
	// R11 += temp
	R.zero_pad_sum<false>(temp, 0, 0);

	temp.sizes[0] = lower_mid_sizes[0];// setting sizes in a way such that 
	temp.sizes[1] = lower_mid_sizes[2];// R22 and temp have matching sizes.
	//this is better than calling the function temp.crop(size0/2, size1/2) as it does not waste time rewriting
	// the entries of temp. We are going to replace temp with another matrix anyway, so we don't care
	// about it having sizes different than the actual size of its entries.

	//and now we perform R22 += temp.
	R.zero_pad_sum<false>(temp, upper_mid_sizes[0], upper_mid_sizes[2]);

	// temp = (A21 + A22) * B11
	temp = basic_Strassen_recursion(zero_pad_sum<T,false>(A, A, chunksA[1][0], chunksA[1][1]),
		B, switch_size,
		chunksA[1][0].shift_to_origin(), chunksB[0][0]);


	// R21 += temp
	R.zero_pad_sum<false>(temp, upper_mid_sizes[0], 0);

	// R22 -= temp
	temp.sizes[1] = lower_mid_sizes[2];
	R.zero_pad_sum<true>(temp, upper_mid_sizes[0], upper_mid_sizes[2]);

	// temp = A11 * (B12 - B22)
	temp = basic_Strassen_recursion(A, zero_pad_sum<T,true>(B, B, chunksB[0][1], chunksB[1][1]), switch_size,
		chunksA[0][0], chunksB[0][1].shift_to_origin());

	// R12 += temp
	R.zero_pad_sum<false>(temp, 0, upper_mid_sizes[2]);

	// R22 += temp
	temp.sizes[0] = lower_mid_sizes[0];
	R.zero_pad_sum<false>(temp, upper_mid_sizes[0], upper_mid_sizes[2]);

	// temp = A22 * (B21 - B11)
	//in order to bring A22 to the right size for later addition with R11, we have to zero_pad it.
	temp = A.submatrix(chunksA[1][1]);

	temp.expand(upper_mid_sizes[0], upper_mid_sizes[1], (T)0);

	temp = basic_Strassen_recursion(temp, zero_pad_sum<T,true>(B, B, chunksB[1][0], chunksB[0][0]), switch_size,
		chunksA[0][0].shift_to_origin(), chunksB[0][0].shift_to_origin());

	// R11 += temp
	R.zero_pad_sum<false>(temp, 0, 0);

	// R21 += temp
	temp.sizes[0] = lower_mid_sizes[0];
	R.zero_pad_sum<false>(temp, upper_mid_sizes[0], 0);

	// temp = (A11 + A12) * B22
	temp = B.submatrix(chunksB[1][1]);
	temp.expand(upper_mid_sizes[1], upper_mid_sizes[2], (T)0);
	temp = basic_Strassen_recursion(zero_pad_sum<T,false>(A, A, chunksA[0][0], chunksA[0][1]), temp, switch_size,
		chunksA[0][0].shift_to_origin(), chunksB[0][0].shift_to_origin());

	// R11 -= temp
	R.zero_pad_sum<true>(temp, 0, 0);

	// R12 += temp
	temp.sizes[1] = lower_mid_sizes[2];
	R.zero_pad_sum<false>(temp, 0, upper_mid_sizes[2]);

	//temp = (A21 - A11) * (B11 + B12)
	temp = basic_Strassen_recursion(zero_pad_sum<T,true>(A, A, chunksA[1][0], chunksA[0][0]),
		zero_pad_sum<T,false>(B, B, chunksB[0][0], chunksB[0][1]), switch_size,
		chunksA[0][0].shift_to_origin(), chunksB[0][0].shift_to_origin());

	// R22 += temp
	temp.sizes[0] = lower_mid_sizes[0];
	temp.sizes[1] = lower_mid_sizes[2];
	R.zero_pad_sum<false>(temp, upper_mid_sizes[0], upper_mid_sizes[2]);

	//temp = (A12 - A22) * (B21 + B22)
	temp = basic_Strassen_recursion(zero_pad_sum<T,true>(A, A, chunksA[0][1], chunksA[1][1]),
		zero_pad_sum<T,false>(B, B, chunksB[1][0], chunksB[1][1]), switch_size,
		chunksA[0][1].shift_to_origin(), chunksB[1][0].shift_to_origin());

	// R11 += temp
	R.zero_pad_sum<false>(temp, 0, 0);
	return R;
}