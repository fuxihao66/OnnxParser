// OnnxParserTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "../OnnxParser.h"
int main()
{
    std::map<std::string, ONNX_PARSER::TensorInfo> inputMap;
    std::map<std::string, ONNX_PARSER::TensorInfo> outputMap;
    std::map<std::string, ONNX_PARSER::InitializerTensorInfo> graphInitializers;
    std::map<std::string, ONNX_PARSER::Op> graphNodes;
    std::vector<ONNX_PARSER::BindingInfo> bindings;
    std::vector<char> weights;
    unsigned int opsetVersion;

    //ONNX_PARSER::PERROR isOk = ONNX_PARSER::ParseFromFile(L"D:/candy-9.onnx", inputMap, outputMap, graphNodes, graphInitializers, bindings, weights, opsetVersion);

    ONNX_PARSER::OnnxParser* parser = new ONNX_PARSER::OnnxParser(L"D:/candy-9.onnx");
    graphInitializers = parser->GetGraphInitializers(); // error
    //std::cout << graphInitializers.size();
    outputMap = parser->GetOutputs();
    inputMap = parser->GetInputs();
    bindings = parser->GetBindings();
    weights = parser->GetWeights();
    graphNodes = parser->GetGraphNodes();
    opsetVersion = parser->GetOpsetVersion();
    delete(parser);
    //std::cout << graphNodes.size();
    std::cout << opsetVersion;
    //std::cout << graphInitializers.size();
    /*if (isOk == ONNX_PARSER::PERROR::O_OK) {
        std::cout << "ok" << std::endl;
    }*/
    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
