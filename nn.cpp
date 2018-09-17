#include "nn.h"
#include <iostream>
#include <ctime>

NN::NN(std::vector<size_t> layer)
{
  size_t nbNeuronesLastLayer = 1;
  for (size_t i = 0; i < layer.size(); i++)
  {
    Matrix matrix(layer[i], nbNeuronesLastLayer);
    this->matrices.push_back(matrix);
    Matrix bias(layer[i], 1);
    this->biases.push_back(bias);
    nbNeuronesLastLayer = layer[i];
  }
}

void NN::display(void) const
{
  for (size_t i = 0; i < this->matrices.size(); i++)
  {
    if (i == 0)
    {
      std::cout << "Inputs :" << std::endl;
      this->matrices[i].display();
      continue;
    }
    std::cout << "Layer #" << i << std::endl;
    std::cout << "Matrix :" << std::endl;
    this->matrices[i].display();
    std::cout << "Bias :" << std::endl;
    this->biases[i].display();
    std::cout << "----------" << std::endl;
  }
}

void NN::initRandom(void)
{
  for (size_t i = 0; i < this->matrices.size(); i++)
  {
    this->initRandom(&this->matrices[i]);
    this->initRandom(&this->biases[i]);
  }
}

void NN::initRandom(Matrix* matrix)
{
  for (size_t i = 0; i < matrix->getWidth(); i++)
  {
    for (size_t j = 0; j < matrix->getHeight(); j++)
    {
      float v = (float)(rand() % 10000) / 100.0;
      matrix->setValue(j, i, v);
    }
  }
}

void NN::setInputs(std::vector<float> inputs)
{
  this->inputs = inputs;
  // Set the Inputs
  if (this->inputs.size() != this->matrices[0].getHeight())
  {
    throw "The number of inputs is different of the number of weights in the seconde layer";
  }
  for (size_t i = 0; i < this->inputs.size(); i++)
  {
    this->matrices[0].setValue(i, 0, this->inputs[i]);
  }
}

std::vector<float> NN::calculateOutputs(void)
{
  // Set the Inputs
  if (this->inputs.size() != this->matrices[0].getHeight())
  {
    throw "The number of inputs is different of the number of weights in the seconde layer";
  }
  for (size_t i = 0; i < this->inputs.size(); i++)
  {
    this->matrices[0].setValue(i, 0, this->inputs[i]);
  }

  Matrix out;
  out = this->matrices[0];
  for (size_t i = 1; i < this->matrices.size(); i++)
  {
    out = this->relu(this->matrices[i] * out + this->biases[i]);
  }
  std::vector<float> output;
  for (size_t i = 0; i < out.getHeight(); i++)
  {
    float value = out.getValue(i, 0);
    output.push_back(value);
  }
  return output;
}

Matrix NN::relu(const Matrix& matrix) const
{
  Matrix out = matrix;

  for (size_t i = 0; i < out.getWidth(); i++)
  {
    for (size_t j = 0; j < out.getHeight(); j++)
    {
      float v = out.getValue(j, i);
      if (v < 0.0)
      {
        v = 0.0;
      }
      out.setValue(j, i, v < 0.0 ? 0.0 : v);
    }
  }

  return out;
}
