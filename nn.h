#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#include <vector>
#include "matrix.h"

class NN
{
private:
  std::vector<Matrix> matrices;
  std::vector<Matrix> biases;
  std::vector<float> inputs;

  void initRandom(Matrix* matrix);
  Matrix relu(const Matrix& matrix) const;

public:
  NN(std::vector<size_t> layers);
  void display(void) const;
  void initRandom(void);
  void setInputs(std::vector<float> inputs);
  std::vector<float> calculateOutputs(void);
};

#endif // NN_H_INCLUDED
