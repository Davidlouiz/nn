#include "matrix.h"
#include "nn.h"
#include <iostream>
#include <string>

int main(void)
{
  // Create a NN
  std::vector<size_t> layers;
  layers.push_back(2); // 2 inputs
  layers.push_back(2); // 2 neurones in hidden layer
  layers.push_back(1); // 1 output
  NN nn(layers);
  nn.initRandom();

  // Set inputs
  std::vector<float> inputs;
  inputs.push_back(12.34);
  inputs.push_back(56.78);
  nn.setInputs(inputs);

  // Display the NN
  nn.display();

  // Calculate outputs
  std::vector<float> outputs = nn.calculateOutputs();

  // Print outputs
  std::cout << "Ouput :" << std::endl;
  for (size_t i = 0; i < outputs.size(); i++)
  {
    std::cout << outputs[i] << std::endl;
  }
}
