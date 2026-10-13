// scipy.ndimage.median_filter, 1-D, mode "reflect": see ndfilter.hpp.
//
// The heap ("Mediator") is a port of scipy 1.18.1's
// scipy/ndimage/src/_rank_filter_1d.cpp (itself after Ashelly's
// streaming-median heap), restricted to mode REFLECT and origin 0.
// SciPy: Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers,
// BSD-3-Clause.

#include "math/ndfilter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sstvae::math {
namespace {

// Indices into `data` arranged as a max heap (heap[-1], heap[-2], ...:
// the values below the rank), the rank's value at heap[0], and a min heap
// (heap[1], heap[2], ...: the values above). `pos` maps a data slot to its
// heap index.
struct Mediator {
    std::vector<int> mem;
    int* pos;
    int* heap;
    int N;
    int idx = 0;
    int minCt;
    int maxCt;

    Mediator(int nItems, int rank) : mem(static_cast<std::size_t>(2 * nItems)) {
        pos = mem.data();
        heap = mem.data() + nItems + rank;
        N = nItems;
        minCt = nItems - rank - 1;
        maxCt = rank;
        while (nItems--) {
            pos[nItems] = nItems - rank;
            heap[pos[nItems]] = nItems;
        }
    }
};

template <typename T>
int mmless(const T* data, const Mediator& m, int i, int j) {
    return data[m.heap[i]] < data[m.heap[j]];
}

int mmexchange(Mediator& m, int i, int j) {
    int t = m.heap[i];
    m.heap[i] = m.heap[j];
    m.heap[j] = t;
    m.pos[m.heap[i]] = i;
    m.pos[m.heap[j]] = j;
    return 1;
}

template <typename T>
int mmCmpExch(const T* data, Mediator& m, int i, int j) {
    return mmless(data, m, i, j) && mmexchange(m, i, j);
}

template <typename T>
void minSortDown(const T* data, Mediator& m, int i) {
    for (i *= 2; i <= m.minCt; i *= 2) {
        if (i < m.minCt && mmless(data, m, i + 1, i)) ++i;
        if (!mmCmpExch(data, m, i, i / 2)) break;
    }
}

template <typename T>
void maxSortDown(const T* data, Mediator& m, int i) {
    for (i *= 2; i >= -m.maxCt; i *= 2) {
        if (i > -m.maxCt && mmless(data, m, i, i - 1)) --i;
        if (!mmCmpExch(data, m, i / 2, i)) break;
    }
}

template <typename T>
int minSortUp(const T* data, Mediator& m, int i) {
    while (i > 0 && mmCmpExch(data, m, i, i / 2)) i /= 2;
    return i == 0;
}

template <typename T>
int maxSortUp(const T* data, Mediator& m, int i) {
    while (i < 0 && mmCmpExch(data, m, i / 2, i)) i /= 2;
    return i == 0;
}

template <typename T>
void MediatorInsert(T* data, Mediator& m, T v) {
    int p = m.pos[m.idx];
    T old = data[m.idx];
    data[m.idx] = v;
    m.idx++;
    if (m.idx == m.N) m.idx = 0;

    if (p > 0) {  // new item is in minHeap
        if (v > old) {
            minSortDown(data, m, p);
            return;
        }
        if (minSortUp(data, m, p) && mmCmpExch(data, m, 0, -1)) maxSortDown(data, m, -1);
    } else if (p < 0) {  // new item is in maxheap
        if (v < old) {
            maxSortDown(data, m, p);
            return;
        }
        if (maxSortUp(data, m, p) && mmCmpExch(data, m, 1, 0)) minSortDown(data, m, 1);
    } else {  // new item is at rank
        if (maxSortUp(data, m, -1)) maxSortDown(data, m, -1);
        if (minSortUp(data, m, 1)) minSortDown(data, m, 1);
    }
}

// _rank_filter_1d.cpp's _rank_filter, mode REFLECT, origin 0, lim2 >= 0.
template <typename T>
void rank_filter_reflect(const T* in_arr, int rank, int arr_len, int win_len, T* out_arr) {
    const int lim = (win_len - 1) / 2;
    const int lim2 = arr_len - lim;
    Mediator m(win_len, rank);
    std::vector<T> data(static_cast<std::size_t>(win_len), T());
    int i;
    for (i = win_len - lim - 1; i > -1; i--) MediatorInsert(data.data(), m, in_arr[i]);
    for (i = 0; i < lim; i++) MediatorInsert(data.data(), m, in_arr[i]);
    for (i = lim; i < arr_len; i++) {
        MediatorInsert(data.data(), m, in_arr[i]);
        out_arr[i - lim] = data[static_cast<std::size_t>(m.heap[0])];
    }
    const int arr_len_thresh = arr_len - 1;
    for (i = 0; i < lim; i++) {
        MediatorInsert(data.data(), m, in_arr[arr_len_thresh - i]);
        out_arr[lim2 + i] = data[static_cast<std::size_t>(m.heap[0])];
    }
}

// ni_filters.c's NI_Select: Hoare-partition quickselect on the window as
// the generic filter lays it out (footprint order, in double), pivot the
// first element. Which of two equal values (0.0 and -0.0) it returns, and
// what a NaN does, follow from this exact sequence of comparisons and
// swaps, so it is copied rather than replaced by std::nth_element.
double ni_select(double* buffer, std::int64_t min, std::int64_t max, std::int64_t rank) {
    for (;;) {
        if (min == max) return buffer[min];
        const double x = buffer[min];
        std::int64_t ii = min - 1;
        std::int64_t jj = max + 1;
        for (;;) {
            do
                jj--;
            while (buffer[jj] > x);
            do
                ii++;
            while (buffer[ii] < x);
            if (ii < jj) {
                const double t = buffer[ii];
                buffer[ii] = buffer[jj];
                buffer[jj] = t;
            } else {
                break;
            }
        }
        ii = jj - min + 1;
        if (rank < ii) {
            max = jj;  // NI_Select(buffer, min, jj, rank)
        } else {
            min = jj + 1;  // NI_Select(buffer, jj + 1, max, rank - ii)
            rank -= ii;
        }
    }
}

// ndimage's "reflect" index for position i of an array of n > 0.
std::int64_t reflect_index(std::int64_t i, std::int64_t n) {
    const std::int64_t period = 2 * n;
    std::int64_t j = i % period;
    if (j < 0) j += period;
    return j < n ? j : period - 1 - j;
}

template <typename T>
std::vector<T> median_filter_impl(std::span<const T> x, int size) {
    if (size < 1 || size % 2 == 0)
        throw std::invalid_argument("median_filter: size must be odd and positive, not " + std::to_string(size));
    const std::size_t n = x.size();
    std::vector<T> out(n);
    if (n == 0) return out;
    if (size == 1) {  // rank 0 of 1: scipy's minimum_filter of size 1, a copy
        std::copy(x.begin(), x.end(), out.begin());
        return out;
    }
    if (n > static_cast<std::size_t>(INT32_MAX)) throw std::invalid_argument("median_filter: array too long");
    const int arr_len = static_cast<int>(n);
    const int rank = size / 2;
    const int lim = (size - 1) / 2;
    if (arr_len == 1) {
        out[0] = x[0];
        return out;
    }
    if (arr_len - lim > 0) {
        rank_filter_reflect(x.data(), rank, arr_len, size, out.data());
        return out;
    }
    // scipy's generic filter (_nd_image.rank_filter, NI_RankFilter) where
    // lim2 < 0: the window, reflected, copied into a double buffer in
    // footprint order and NI_Select'ed, the result cast back to T.
    // lim2 == 0 also comes here: scipy's fast path then primes its heap
    // with in_arr[arr_len], one past the end -- a read out of bounds whose
    // value is evicted before the first output, but whose comparisons have
    // already shaped the heap, so which of two equal values (or, for a NaN
    // read, whether the heap is still a heap) depends on memory past the
    // array. That is not reproducible; the generic filter's answer is the
    // value scipy gives whenever its heap stayed valid, and the same bits
    // except possibly for the sign of a zero.
    std::vector<double> win(static_cast<std::size_t>(size));
    for (std::int64_t i = 0; i < arr_len; ++i) {
        for (int k = 0; k < size; ++k)
            win[static_cast<std::size_t>(k)] =
                static_cast<double>(x[static_cast<std::size_t>(reflect_index(i - lim + k, arr_len))]);
        out[static_cast<std::size_t>(i)] = static_cast<T>(ni_select(win.data(), 0, size - 1, rank));
    }
    return out;
}

}  // namespace

std::vector<double> median_filter(std::span<const double> x, int size) { return median_filter_impl(x, size); }
std::vector<float> median_filter(std::span<const float> x, int size) { return median_filter_impl(x, size); }

}  // namespace sstvae::math
