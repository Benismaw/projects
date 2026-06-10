#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "fast_idct.h"

#define PI    3.14159265358979323846

#define c1_1  0.98078528040323043 // cos(pi/16)
#define c1_2 -0.78569495838710224 // sin(pi/16)-cos(pi/16)
#define c1_3 -1.17587560241935862 // -(cos(pi/16)+sin(pi/16))

#define c3_1  0.83146961230254524 // cos(3*pi/16)
#define c3_2 -0.27589937928294306 // sin(3*pi/16)-cos(3*pi/16)
#define c3_3 -1.38703984532214752 // -(sin(3*pi/16)+cos(3*pi/16))

#define c6_1  0.27059805007309851 // cos(6*pi/16)/sqrt(2)
#define c6_2  0.38268343236508967 // (sin(6*pi/16) - cos(6*pi/16))/ sqrt(2)
#define c6_3 -0.92387953251128674 // -(cos(6pi/16)+sin(6pi/16))/ sqrt(2)

#define sqrt2 1.41421356237309515 // sqrt(2)


double sum_div_2(double a, double b){ 
    return (a+b) / 2.0;
}

double minus_div_2(double a , double b){
    return  (a-b) / 2.0;
}

void loffler_code(double temp[8], double matrix[8]) {
        // J'alterne entre les vecteurs pour ne pas utiliser 4 vecteurs de 8 mais plutot 2
        double a[8];
        for (int i = 0; i < 8; i++){
            a[i] = matrix[i];
        }
        double b[8];

        // STAGE 4 INVERSE
        b[0] = a[0];
        b[1] = a[4];
        b[2] = a[2];
        b[3] = a[6];
        b[4] = minus_div_2(a[1], a[7]);
        b[5] = a[3] / sqrt2;
        b[6] = a[5] / sqrt2;
        b[7] = sum_div_2(a[1], a[7]);

        // STAGE 3 INVERSE
        a[0] = sum_div_2(b[0], b[1]);
        a[1] = minus_div_2(b[0], b[1]);
        double z = c6_1*(b[3]+b[2]);
        a[2] = c6_3*b[3]+ z;
        a[3] = c6_2 *b[2] +z;
        a[4] = sum_div_2(b[4], b[6]);
        a[5] = minus_div_2(b[7], b[5]);
        a[6] = minus_div_2(b[4], b[6]);
        a[7] = sum_div_2(b[7], b[5]);

        // STAGE 2 INVERSE
        b[0] = sum_div_2(a[0], a[3]);
        b[1] = sum_div_2(a[1], a[2]);
        b[2] = minus_div_2(a[1], a[2]);
        b[3] = minus_div_2(a[0], a[3]);
        z = c3_1*(a[4]+a[7]);
        b[4] = c3_3*a[7] + z;
        b[7] = c3_2*a[4] + z;
        z = c1_1*(a[5]+a[6]);
        b[5] = c1_3*a[6] + z;
        b[6] = c1_2*a[5] + z;

        // STAGE 1 INVERSE
        temp[0] = sum_div_2(b[0], b[7]);
        temp[1] = sum_div_2(b[1], b[6]);
        temp[2] = sum_div_2(b[2], b[5]);
        temp[3] = sum_div_2(b[3], b[4]);
        temp[4] = minus_div_2(b[3], b[4]);
        temp[5] = minus_div_2(b[2], b[5]);
        temp[6] = minus_div_2(b[1], b[6]);
        temp[7] = minus_div_2(b[0], b[7]);
}
void fast_idct(uint8_t s[8][8], int16_t matrix[8][8]){
    double temp[8][8];
    double resultat[8][8];
  
    // IDCT HORIZONTAL
    for (int x = 0; x < 8; x++){
        double line[8];
        for (int i = 0; i < 8; i++){
            line[i] = (double)(matrix[x][i]);
        }
        loffler_code(temp[x], line);
    }
    // IDCT VERTICAL
    for (int x = 0; x < 8; x++){
        double column[8];
        for(int i = 0; i < 8; i++){
            column[i] = (temp[i][x]);
        }
        double res[8];
        loffler_code(res, column);
        for (int y = 0; y < 8; y++){
            resultat[y][x] = res[y];
        }
    }
    // Quantification finale et conversion vers uint8
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double val = (resultat[i][j]*8+128);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            s[i][j] = (uint8_t) (round(val));
        }
    }
}