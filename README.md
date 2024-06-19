# Test_vector_and_matrix
 test and compare cases

Conclusion - Use Armadillo with GPU or Eigen without GPU

Vector Addition

C Style
Size  Allocation   Initialize   Calculation  Verifying    ReleaseMem   Verified
10^4  7.90000e-06  8.46000e-05  1.31000e-05  9.80000e-06  1.08000e-05  true
10^5  2.07000e-05  9.83500e-04  1.20200e-04  6.07000e-05  3.56000e-05  true
10^6  1.79000e-05  7.09660e-03  7.95100e-04  6.46500e-04  4.89200e-04  true
10^7  3.22000e-05  7.15690e-02  6.91470e-03  6.14650e-03  2.83240e-03  true
10^8  2.64000e-05  7.14039e-01  1.10699e-01  6.17082e-02  5.26058e-02  true
C/C++ Style
Size  Allocation   Initialize   Calculation  Verifying    ReleaseMem   Verified
10^4  1.03000e-05  9.37000e-05  1.26000e-05  7.20000e-06  4.08000e-05  true
10^5  1.02000e-05  1.11560e-03  1.40700e-04  5.78000e-05  5.29000e-05  true
10^6  2.22000e-05  7.32310e-03  7.09200e-04  6.09700e-04  2.91000e-04  true
10^7  2.17000e-05  7.21821e-02  7.18920e-03  8.64060e-03  3.25810e-03  true
10^8  3.47000e-05  6.90669e-01  8.37545e-02  6.46192e-02  4.68013e-02  true
C++ Style
Size  Allocation   Initialize   Calculation  Verifying    Verified
10^4  4.99000e-05  1.29800e-04  2.30000e-06  7.90000e-06  true
10^5  3.92500e-04  1.14690e-03  3.41000e-05  8.78000e-05  true
10^6  1.22950e-03  7.88440e-03  5.20400e-04  6.75000e-04  true
10^7  9.71840e-03  7.50923e-02  5.19150e-03  6.38070e-03  true
10^8  1.00109e-01  7.54848e-01  4.91236e-02  5.74186e-02  true
C++ Vector
Size  Allocation   Initialize   Cal_vector   Cal_vecdata  Verifying    Verified
10^4  1.10000e-05  6.16000e-05  1.60000e-06  1.10000e-06  5.70000e-06  true
10^5  3.69300e-04  7.28700e-04  1.93000e-05  1.94000e-05  8.54000e-05  true
10^6  1.26650e-03  6.07270e-03  4.87000e-04  2.47400e-04  5.60400e-04  true
10^7  9.86270e-03  6.02535e-02  5.85680e-03  5.32210e-03  5.95520e-03  true
10^8  1.02476e-01  6.44444e-01  5.00874e-02  5.06467e-02  5.81083e-02  true
boost ublas
Size  Allocation   Initialize   Cal_adddvec2 Cal_ublas    Verifying    Verified
10^4  1.80000e-06  7.69000e-05  1.58000e-05  3.00000e-07  8.60000e-06  true
10^5  7.30000e-06  9.19900e-04  1.10900e-04  3.00000e-07  1.24700e-04  true
10^6  2.23000e-05  7.08120e-03  7.43400e-04  5.00000e-07  6.54800e-04  true
10^7  2.74000e-05  6.94926e-02  7.35050e-03  2.10000e-06  6.36780e-03  true
10^8  4.82000e-05  6.87293e-01  8.03478e-02  2.20000e-06  7.13197e-02  true
Cuda C Style
Size  Allocation   Initialize   memcpy       Calculation  memcpy       Verifying    memRelease   Verified
10^4  6.06100e-04  7.41000e-05  9.26500e-04  1.72693e-02  4.55000e-05  7.00000e-07  1.53994e-02 false
blocksPerGrid: 3        threadsPerBlock: 3840
10^5  4.01412e-02  1.05960e-03  1.38300e-04  5.23700e-04  1.77800e-04  4.00000e-07  1.64971e-02 false
blocksPerGrid: 27       threadsPerBlock: 3840
10^6  3.99856e-02  7.10050e-03  1.23530e-03  6.28800e-04  7.23400e-04  7.00000e-07  1.59191e-02 false
blocksPerGrid: 261      threadsPerBlock: 3840
10^7  4.14080e-02  7.01397e-02  1.23863e-02  8.81300e-04  6.26010e-03  1.10000e-06  1.95301e-02 false
blocksPerGrid: 2605     threadsPerBlock: 3840
10^8  4.71502e-02  7.03348e-01  1.22104e-01  1.60110e-03  7.44929e-02  3.30000e-06  5.76897e-02 false
blocksPerGrid: 26042    threadsPerBlock: 3840


Matrix - Vector Multiplication

C Style
Size  Allocation   Initialize   Calculation  Verifying    ReleaseMem   Verified
10^2  6.00000e-06  5.02000e-05  1.89000e-05  4.00000e-07  1.10000e-06  true
10^3  1.19000e-05  3.82540e-03  1.32630e-03  1.10000e-06  9.80000e-05  true
10^4  2.24000e-05  3.39609e-01  1.21682e-01  1.04000e-05  2.27622e-02  true
C/C++ Style
Size  Allocation   Initialize   Calculation  Verifying    ReleaseMem   Verified
10^2  1.80000e-06  3.84000e-05  7.40000e-06  2.00000e-07  1.00000e-06  true
10^3  1.85000e-05  3.66610e-03  1.29870e-03  1.20000e-06  9.43000e-05  true
10^4  2.04000e-05  3.35410e-01  1.27171e-01  9.80000e-06  1.06344e-02  true
C++ Style
Size  Allocation   Initialize   Calculation  Verifying    Verified
10^2  9.50000e-06  5.19000e-05  1.35000e-05  2.00000e-07  true
10^3  5.21700e-04  3.88010e-03  7.38500e-04  1.20000e-06  true
10^4  3.22334e-02  3.64850e-01  7.28256e-02  1.60000e-05  true
C++ Vector
Size  Allocation   Initialize   Cal_vector   Cal_ptr      Verifying    Verified
10^2  7.50000e-06  4.42000e-05  8.10000e-06  6.10000e-06  5.00000e-07  true
10^3  5.99600e-04  4.00290e-03  7.26400e-04  7.33000e-04  1.20000e-06  true
10^4  3.11225e-02  3.70513e-01  7.32456e-02  7.19890e-02  1.04000e-05  true
boost ublas
Size  Allocation   Initialize   Cal_prod     Verifying    Verified
10^2  2.40000e-06  4.80000e-05  9.30000e-06  2.00000e-07  true
10^3  1.62000e-05  3.99260e-03  6.96300e-04  9.00000e-07  true
10^4  1.38000e-05  3.78760e-01  7.19251e-02  7.80000e-06  true
Cuda C Style
Size  Allocation   Initialize   cuda_mem&cpy Cal_cuda     memcpy_back  cudablas     memcpy_back  verifying    memRelease   Verified
10^2  1.30000e-06  4.36000e-05  1.35566e+00  1.36243e+00  6.05000e-05  2.23180e-02  2.24165e-02  2.50000e-06  3.91000e-04  true
2.64666e+02 2.64666e+02 1.83105e-04
10^3  1.04000e-05  3.87040e-03  5.16720e-03  5.18740e-03  6.51700e-04  1.04020e-03  1.08740e-03  5.36800e-04  3.93700e-04 false
2.49870e+03 2.49871e+03 6.83594e-03
10^4  1.48000e-05  3.60053e-01  4.35247e-01  4.35272e-01  5.35750e-03  1.16232e-02  1.29452e-02  5.54800e-04  1.75030e-03 false
Eigen
Size  Allocation   Initialize   Cal_*        Verifying    Verified
10^2  1.40000e-06  3.67000e-05  1.08000e-05  3.00000e-07  true
10^3  1.38000e-05  3.57500e-03  2.15640e-03  3.70000e-06  true
10^4  1.57000e-05  3.39361e-01  2.72578e-01  1.68100e-04  true
Armadillo
Size  Alloc/w Rand Cal_*        Verifying    Verified
10^2  1.05700e-04  1.74160e-03  1.90000e-06  true
10^3  8.56470e-03  2.78400e-04  9.00000e-06  true
10^4  8.63397e-01  3.56457e-02  6.98000e-05 false


Matrix - Matrix Multiplication

C Style
Size  Allocation   Initialize   Calculation  Verifying    ReleaseMem   Verified
10^1  1.00000e-06  1.70000e-06  1.60000e-06  2.00000e-07  7.00000e-07  true
10^2  1.03000e-05  1.20400e-04  1.17250e-03  5.00000e-07  9.70000e-06  true
10^3  2.55000e-05  6.59810e-03  7.54784e-01  5.40000e-06  3.15700e-04  true
C/C++ Style
Size  Allocation   Initialize   Calculation  Verifying    ReleaseMem   Verified
10^1  3.70000e-06  5.30000e-06  1.30000e-06  3.00000e-07  1.20000e-06  true
10^2  3.90000e-06  1.04600e-04  1.06630e-03  2.00000e-07  7.10000e-06  true
10^3  2.42000e-05  6.51260e-03  7.60659e-01  5.90000e-06  3.21500e-04  true
C++ Style
Size  Allocation   Initialize   Calculation  Verifying    Verified
10^1  4.80000e-06  6.70000e-06  1.80000e-06  3.00000e-07  true
10^2  7.93000e-05  1.44700e-04  7.74500e-04  2.00000e-07  true
10^3  1.19990e-03  7.83690e-03  7.54539e-01  6.40000e-06  true
C++ Vector
Size  Allocation   Initialize   Cal_vector   Cal_ptr      Verifying    Verified
10^1  4.20000e-06  5.90000e-06  1.10000e-06  9.00000e-07  3.00000e-07  true
10^2  3.26000e-05  1.11500e-04  5.86900e-04  8.63000e-04  3.00000e-07  true
10^3  1.17460e-03  7.55240e-03  7.41505e-01  7.42130e-01  5.30000e-06  true
boost ublas
Size  Allocation   Initialize   Cal_prod     Verifying    Verified
10^1  5.20000e-06  7.00000e-06  1.80000e-06  1.00000e-07  true
10^2  7.40000e-06  1.12400e-04  5.69900e-04  3.00000e-07  true
10^3  2.04000e-05  7.50630e-03  7.33669e-01  3.00000e-06  true
Cuda C Style
Size  Allocation   Initialize   cuda_mem&cpy Cal_cuda     memcpy_back  cudablas     memcpy_back  verifying    memRelease   Verified
10^1  3.90000e-06  5.90000e-06  1.30237e+00  1.30604e+00  4.21000e-05  4.18550e-03  4.23410e-03  1.30000e-06  3.30400e-04  true
10^2  6.70000e-06  8.62000e-05  6.53300e-04  6.70300e-04  4.40500e-04  5.05500e-04  5.19600e-04  7.00000e-07  2.32400e-04  true
2.58448e+02 2.58448e+02 1.52588e-04
10^3  1.58000e-05  7.13210e-03  1.00524e-02  1.00769e-02  3.41630e-03  3.50690e-03  3.55720e-03  5.29100e-04  6.92300e-04 false
Eigen
Size  Allocation   Initialize   Cal_*        Verifying    Verified
10^1  1.60000e-06  3.20000e-06  7.00000e-07  1.00000e-07  true
10^2  5.90000e-06  5.13000e-05  9.70000e-06  4.00000e-07  true
10^3  1.25000e-05  3.24960e-03  2.21930e-03  4.30000e-06  true
Armadillo
Size  Alloc/w Rand Cal_*        Verifying    Verified
10^1  3.70000e-06  1.15000e-05  4.00000e-07  true
10^2  7.78000e-05  9.91000e-05  7.00000e-07  true
10^3  8.53290e-03  3.48900e-04  6.70000e-06  true

