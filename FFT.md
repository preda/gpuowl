## Table of small FFTs


The "cost" is estimated as 2xFMAs + 1xADDs.
Excess vGPRs is #vGPRs - 4*N.


| N  | FMAs | ADDs | Excess vGPRs | Cost | Cost/N^2 | Source                  |
|----|------|------|--------------|------|----------|-------------------------|
|  2 |   0  |   4  |              |    4 |     1    |                         |
|  3 |   6  |   6  |      1       |   18 |     2    |[FFT-3](src/cl/fft3.cl)  |
|  4 |   0  |  16  |      5       |   16 |     1    |[FFT-4](src/cl/fft4.cl)  |
|  5 |  12  |  24  |      4       |   28 |     1.1  |[FFT-5](src/cl/fft5.cl)  |
|  6 |  12  |  24  |     11       |   28 |     0.8  |[FFT-6](src/cl/fft6.cl)  |
|  7 |  10  |  62  |     13       |   82 |     1.7  |[FFT-7](src/cl/fft7.cl)  |
|  8 |   4  |  52  |      9       |   60 |     0.9  |[FFT-8](src/cl/fft8.cl)  |
|  9 |  20  |  72  |     17       |  112 |     1.4  |[FFT-9](src/cl/fft9.cl)  |
| 10 |  24  |  68  |     13       |  116 |     1.2  |[FFT-10](src/cl/fft10.cl)|
| 11 | 110  |  30  |     33       |  250 |     2.1  |[FFT-11](src/cl/fft11.cl)|
| 12 |  24  |  72  |     13       |  120 |     0.8  |[FFT-12](src/cl/fft12.cl)|
| 13 | 156  |  36  |     25       |  348 |     2.1  |[FFT-13](src/cl/fft13.cl)|
| 14 |  20  | 152  |     13       |  192 |     1    |[FFT-14](src/cl/fft14.cl)|
| 15 |  66  | 102  |     15       |  234 |     1    |[FFT-15](src/cl/fft15.cl)|

