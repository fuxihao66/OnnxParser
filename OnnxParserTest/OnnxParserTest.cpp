// OnnxParserTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <optional>
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

    ONNX_PARSER::OnnxParser* parser = new ONNX_PARSER::OnnxParser(L"D:/optimized_model_opset9_fp32.onnx");
    graphInitializers = parser->GetGraphInitializers(); // error
    //std::cout << graphInitializers.size();
    outputMap = parser->GetOutputs();
    inputMap = parser->GetInputs();
    bindings = parser->GetBindings();
    weights = parser->GetWeights();
    graphNodes = parser->GetGraphNodes();
    opsetVersion = parser->GetOpsetVersion();
    delete(parser);

    /*for (auto it = graphInitializers.begin(); it != graphInitializers.end(); it++) {
        auto& initializer = it->second;
        std::cout << initializer.name << std::endl;
    }*/

    //std::map<std::string, int> constantName;
    for (auto it = graphNodes.begin(); it != graphNodes.end(); it++) {
        /*std::cout << it->first << std::endl;*/
        auto& node = it->second;
        
        
        //std::cout << node.opType << std::endl;

        /*if (node.opType == "Pad") {
            std::vector<char> tempAttri;

            bool hasMode = node.GetAttribute("mode", ONNX_PARSER::AttributeType::STRING, tempAttri);
            std::string mode;

            if (hasMode) {
                mode.resize(tempAttri.size());
                memcpy((void*)(mode.data()), tempAttri.data(), tempAttri.size());
            }

        }*/
        if (node.opType == "Slice") {
            //std::vector<char> temp;
            //std::vector<int> axes;
            std::vector<int> starts;
            std::vector<int> ends;
            {
                ONNX_PARSER::AttributeValWrapper result = node.GetAttribute("starts", ONNX_PARSER::AttributeType::INTS);
                starts.resize(result.getValue().size() / 4);
                memcpy(starts.data(), result.getValue().data(), result.getValue().size());
            }
            {
                ONNX_PARSER::AttributeValWrapper result = node.GetAttribute("ends", ONNX_PARSER::AttributeType::INTS);
                ends.resize(result.getValue().size() / 4);
                memcpy(ends.data(), result.getValue().data(), result.getValue().size());
            }
            
                                                                                                                 
                                                                                                                      //if (result != std::nullopt) {
            //    //const std::vector<char>& axis = (*result);
            //}
            /*if (hasAxis) {
                axes.resize(temp.size() / 4);
                memcpy(axes.data(), temp.data(), temp.size());
            }*/
            //constantName[node.outputName] = 1;
            //std::cout << node.outputName << std::endl;
        }
    }
    //for (auto it = graphNodes.begin(); it != graphNodes.end(); it++) {
    //    /*std::cout << it->first << std::endl;*/
    //    auto& node = it->second;
    //    std::cout << node.opType << std::endl;


    //    //for (auto& input : node.inputNames) {
    //        /*if (constantName.count(input)) {

    //            std::cout << node.opType << std::endl;
    //        }*/
    //        /*if (input == "132" || input == "161") {
    //            std::cout << "[";
    //            for (auto& input : node.inputNames) {
    //                std::cout << input << " ";
    //            }
    //            std::cout << "]" << std::endl;
    //        }
    //    }*/
    //}
    

    /*for (auto it = graphInitializers.begin(); it != graphInitializers.end(); it++) {
        if (constantName.count(it->first) > 0) {
            auto& initializer = it->second;
            std::cout << it->first << " [";
            for (int i = 0; i < initializer.dims; i++) {
                std::cout << initializer.shapes[i] << " ";

            }
            std::cout << "]" << std::endl;

        }
    }*/



    //std::cout << graphNodes.size();
    //std::cout << opsetVersion;
    //std::cout << graphInitializers.size();
    /*if (isOk == ONNX_PARSER::PERROR::O_OK) {
        std::cout << "ok" << std::endl;
    }*/
    /*ONNX_PARSER::TensorType type = ONNX_PARSER::OnnxTensorType2DmlTensorType(11);
    std::cout << static_cast<unsigned int>(type) << std::endl;*/
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
