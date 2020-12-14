#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dot.h"
#define reluMult -1 // proper relu has an alpha of 0, but an absolute value activation doesn't cause dead neurons so i like using it
#define m 8 // number of training data samples
#define n 3 // number of features per data sample
#define l1 16 // number of hidden layer neurons
#define l2 16 // number of hidden layer 2 neurons
#define l3 1 // number of output neurons

double X[n][m] = 
{
  {1, 1, 1, 1, 0, 0, 0, 0},
  {1, 1, 0, 0, 1, 1, 0, 0},
  {1, 0, 1, 0, 1, 0, 1, 0},
};

double y[l2][m] = 
{
  {1, 0.66, 0.66, 0.33, 0.33, .33, 0.33, 0}
};

/*
 Array madness - i didn't want to declare any arrays during the loops since that would slow the program so i made this garbage.

 t1-b2 - weights and biases for layers 1 and 2.
 z1-a2 - weighted sums and activations
 lds, se - last deltas (ypred - y), squared errors
 t1g-b2g - gradients for weights and biases
 tg1l-z1Der - intermediary arrays to find gradients
 a1t-xi - shape-modified versions of certain existing arrays that are used in gradient calculation
*/

double t1[l1][n];
double b1[l1][1];
double t2[l2][l1];
double b2[l2][1];
double t3[l3][l2];
double b3[l3][1];

double z1[l1][m];
double a1[l1][m];
double z2[l2][m];
double a2[l2][m];
double z3[l3][m];
double a3[l3][m];

double lds[l2][m];
double se[l2][m];

double t1g[l1][n]; // theta 1 gradient (accumulated)
double b1g[l1][1]; 
double t2g[l2][l1];
double b2g[l2][1];
double t3g[l3][l2];
double b3g[l3][1];

double tg1l[l1][n]; // non-accumulated gradient
double d1[l1][1];
double z1Der[l1][m];
double tg2l[l2][l1];
double d2[l2][1];
double z2Der[l2][m];
double tg3l[l3][l2];
double d3[l3][1];
double z3Der[l3][m];

double xt[m][n];
double t1t[n][l1];
double a1t[m][l1];
double t2t[l1][l2];
double a2t[m][l2];
double t3t[l2][l3];

double xi[1][n];
double a1i[1][l1];
double a2i[1][l2];

double relu(double a)
{
  if(a < 0)
    return a * reluMult;
  return a;
}

double reluDer(double a)
{
  if(a < 0)
    return reluMult;
  return 1;
}

double sig(double a){ return 1 / (1 + pow(M_E, -a)); }
double sigDer(double a){ return sig(a) * (1 - sig(a)); }

void forward()
{
  dot(l1, n, m, t1, X, z1);
  oppCM(l1, m, z1, b1, 0);
  oppF(l1, m, z1, relu, a1);

  dot(l2, l1, m, t2, a1, z2);
  oppCM(l2, m, z2, b2, 0);
  oppF(l2, m, z2, relu, a2);

  dot(l3, l2, m, t3, a2, z3);
  oppCM(l3, m, z3, b3, 0);
  oppF(l3, m, z3, sig, a3);

  printf("Predictions: ");
  disp(l3, m, a3); // displays train predictions

  oppM(l3, m, a3, y, 1, lds); // calculate last deltas
}

double cost() 
{
  oppM(l2, m, lds, lds, 2, se);
  return mean(l2, m, se, 1) / 2;
}

void backprop()
{
  // set all the gradient array values to 0 
  initArr(l1, n, t1g, 0); 
  initArr(l1, 1, b1g, 0);
  initArr(l2, l1, t2g, 0);
  initArr(l2, 1, b2g, 0);
  initArr(l3, l2, t3g, 0);
  initArr(l3, 1, b3g, 0);

  // calculate activation derivatives and transpose some matrices
  oppF(l1, m, z1, reluDer, z1Der);
  oppF(l2, m, z2, reluDer, z2Der);
  oppF(l3, m, z3, sigDer, z3Der);
  trans(n, m, X, xt);
  trans(l1, m, a1, a1t);
  trans(l2, m, a2, a2t);
  trans(l1, n, t1, t1t);
  trans(l2, l1, t2, t2t);
  trans(l3, l2, t3, t3t);

  for(int i=0;i<m;++i) // gradient accumulation
  {
    // last layer gradients
    for(int j=0;j<l3;++j)
    {
      d3[j][0] = lds[j][i] * z3Der[j][i];
    }
    getRow(m, l2, a2t, i, a2i);
    dot(l3, 1, l2, d3, a2i, tg3l);
    oppMM(l3, l2, t3g, tg3l, 0);
    oppMM(l3, 1, b3g, d3, 0);

    dot(l2, l3, 1, t3t, d3, d2);
    for(int j=0;j<l2;++j)
      d2[j][0] *= z2Der[j][i];
    getRow(m, l1, a1t, i, a1i);
    dot(l2, 1, l1, d2, a1i, tg2l);
    oppMM(l2, l1, t2g, tg2l, 0);
    oppMM(l2, 1, b2g, d2, 0);

    // third to last layer gradients
    dot(l1, l2, 1, t2t, d2, d1);
    for(int j=0;j<l1;++j)
      d1[j][0] *= z1Der[j][i];
    getRow(m, n, xt, i, xi);
    dot(l1, 1, n, d1, xi, tg1l);
    oppMM(l1, n, t1g, tg1l, 0);
    oppMM(l1, 1, b1g, d1, 0);
  }
}

void descent(double lr) // normal gradient descent
{
  // scale down gradients (dividing by m since i didn't do that in backprop although i should have
  oppSM(l1, n, t1g, lr / m, 2);
  oppSM(l1, 1, b1g, lr / m, 2);
  oppSM(l2, l1, t2g, lr / m, 2);
  oppSM(l2, 1, b2g, lr / m, 2);
  oppSM(l3, l2, t3g, lr / m, 2);
  oppSM(l3, 1, b3g, lr / m, 2);

  // modify weights and biases by gradients
  oppMM(l1, n, t1, t1g, 1);
  oppMM(l1, 1, b1, b1g, 1);
  oppMM(l2, l1, t2, t2g, 1);
  oppMM(l2, 1, b2, b2g, 1);
  oppMM(l3, l2, t3, t3g, 1);
  oppMM(l3, 1, b3, b3g, 1);
}

double randVal(double _)
{
  return (rand()/(double)RAND_MAX) - 0.5;
}

void train(int epochs, double lr)
{
  // initialize random values to trainable parameters
  oppFM(l1, n, t1, randVal);
  oppFM(l1, 1, b1, randVal);
  oppFM(l2, l1, t2, randVal);
  oppFM(l2, 1, b2, randVal);
  oppFM(l3, l2, t3, randVal);
  oppFM(l3, 1, b3, randVal);

  for(int i=0;i<epochs;i++)
  {
    printf("Epoch %d\n", i);
    forward();
    printf("Cost: %f\n\n", cost());
    backprop();
    descent(lr);
  }
}

int main()
{
  train(1000, 0.1); // not sure why learning rate can be massive and still work
  printf("Final Cost: %f\n", cost());
  return 0;
}
