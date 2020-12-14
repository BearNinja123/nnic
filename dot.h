#include <stdio.h>
#include <stdlib.h>

/*
 * Janky 2D array library I made, nearly all functions are void
 * since you can't return actual 2D arrays in C (pointers yes, 
 * arrays, not really). 
 *
 * General parameter names
 * m - number of rows in array
 * n - number of columns in array
 * a - input array
 * ret - array containing values from array
 */

// set all values in array to value "num"
void initArr(int m, int n, double a[m][n], double num)
{
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      a[i][j] = num;
}

// matrix multiplication (a*b -> ret)
void dot(int m, int n, int p, double a[m][n], double b[n][p], double ret[m][p])
{
  for(int i=0;i<m;++i)
    for(int j=0;j<p;++j)
    {
      double sume = 0;
      for(int k=0;k<n;++k)
        sume += a[i][k] * b[k][j];
      ret[i][j] = sume;
    }
}

// i tried doing cache-friendly matrix multiplication but didn't use it in the neural network
void dotCache(int m, int n, int p, double a[m][n], double b[n][p], double ret[m][p])
{
  int blockSize = 64 / sizeof(double); // since cache lines are 64 bytes
  for(int i=0;i<m;++i)
    for(int jB=0;jB<p;jB+=blockSize)
      for(int kB=0;kB<n;kB+=blockSize)

        for(int j=0;j<blockSize;++j)
        {
          double sum = 0;
          for(int k=0;k<blockSize;++k)
            sum += a[i][jB+j] * b[kB+k][jB+j];
          ret[i][jB+j] += sum;
        }
}

// transpose matrix (a' -> ret)
void trans(int m, int n, double a[m][n], double ret[n][m])
{
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      ret[j][i] = a[i][j];
}

// return values of certain column in mx1 array
void getCol(int m, int n, double a[m][n], int col, double ret[m][1])
{
  for(int i=0;i<m;++i)
    ret[i][0] = a[i][col];
}
 
// return values of certain row in 1xn array
void getRow(int m, int n, double a[m][n], int row, double ret[1][m])
{
  for(int i=0;i<n;++i)
    ret[0][i] = a[row][i];
}

// calculate mean or sum of array (option==0 -> mean, option==1 -> sum)
double mean(int m, int n, double a[m][n], int option)
{
  double sume = 0;
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      sume += a[i][j];
  if(option == 0)
    return sume / (m * n);
  else if(option == 1)
    return sume;
}

// use a scalar to do a math operation on an array (option=0, 1, 2, 3 -> +, -, *, /)
void oppS(int m, int n, double a[m][n], double s, int option, double ret[m][n])
{
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      switch(option)
      {
        case 0:
          ret[i][j] += s;
          break;
        case 1:
          ret[i][j] -= s;
          break;
        case 2:
          ret[i][j] *= s;
          break;
        case 3:
          ret[i][j] /= s;
          break;
      }
}

// use a matrix to do an element-wise math operation on an array
void oppM(int m, int n, double a[m][n], double s[m][n], int option, double ret[m][n])
{
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      switch(option)
      {
        case 0:
          ret[i][j] = a[i][j] + s[i][j];
          break;
        case 1:
          ret[i][j] = a[i][j] - s[i][j];
          break;
        case 2:
          ret[i][j] = a[i][j] * s[i][j];
          break;
        case 3:
          ret[i][j] = a[i][j] / s[i][j];
          break;
      }
}

// use a column to do a math operation on each column of an array
void oppC(int m, int n, double a[m][n], double s[m][1], int option, double ret[m][n])
{
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      switch(option)
      {
        case 0:
          ret[i][j] += s[i][0];
          break;
        case 1:
          ret[i][j] -= s[i][0];
          break;
        case 2:
          ret[i][j] *= s[i][0];
          break;
        case 3:
          ret[i][j] /= s[i][0];
          break;
      }
}

// use a row to do a math operation on each row of an array
void oppR(int m, int n, double a[m][n], double s[1][n], int option, double ret[m][n])
{
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      switch(option)
      {
        case 0:
          ret[i][j] += s[0][j];
          break;
        case 1:
          ret[i][j] -= s[0][j];
          break;
        case 2:
          ret[i][j] *= s[0][j];
          break;
        case 3:
          ret[i][j] /= s[0][j];
          break;
      }
}

// map a function on each element of an array
void oppF(int m, int n, double a[m][n], double func(double), double ret[m][n])
{
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      ret[i][j] = func(a[i][j]);
}

// 'mutable' functions - a is modified instead of ret (i.e. a+=1 instead of ret=a+1)
void oppSM(int m, int n, double a[m][n], double s, int option)
{
  return oppS(m, n, a, s, option, a);
}

void oppMM(int m, int n, double a[m][n], double s[m][n], int option)
{
  return oppM(m, n, a, s, option, a);
}

void oppCM(int m, int n, double a[m][n], double s[m][1], int option) 
{
  return oppC(m, n, a, s, option, a);
}

void oppRM(int m, int n, double a[m][n], double s[1][n], int option)
{
  return oppR(m, n, a, s, option, a);
}

void oppFM(int m, int n, double a[m][n], double func(double))
{
  return oppF(m, n, a, func, a);
}

// display array
void disp(int m, int n, double a[m][n])
{
  for(int i=0;i<m;++i)
  {
    for(int j=0;j<n;++j)
      printf("%f ", a[i][j]);
    printf("\n");
  }
}
