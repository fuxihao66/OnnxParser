
#ifdef ONNXPARSER_EXPORTS
#define ONNXPARSER_API __declspec(dllexport)
#else
#define ONNXPARSER_API __declspec(dllimport)
#endif

#include <memory>
#include <iostream>
#include <xstring>
#include <fcntl.h>
#include <map>
#include <vector>
#include <DirectML.h>




extern "C" {
	namespace ONNX_DML {
		enum PERROR {
			O_OK = 0,
			O_NOTFOUND
		};

		enum TensorType {
			UNDEFINED = 0,
			FLOAT = 1,   // Default type, need to be casted to fp16 when upload resource to GPU
			UINT8 = 2,
			INT8 = 3,
			UINT16 = 4,
			INT16 = 5,
			INT32 = 6,
			INT64 = 7,
			STRING = 8,
			BOOL = 9,
			FLOAT16 = 10,
			DOUBLE = 11,
			UINT32 = 12,
			UINT64 = 13
		};

		struct TensorInfo {
			TensorType tensorType;
			std::string name;
			unsigned int  dims;
			unsigned int* shapes;

			TensorInfo() dims(0), shapes(nullptr) {}
			TensorInfo(const std::string& n, unsigned int d, TensorType t) {
				dims = d;
				name = n;
				tensorType = t;
				shapes = (unsigned int*)malloc(d * sizeof(unsigned int));
			}

			void SetShape(unsigned int dim, unsigned int v) {
				if (dim >= dims)
					return;
				shapes[d] = v;
			}
			
		};

		
		struct Op {
			std::vector<std::string> inputNames;
			std::string opName;
			std::string outputName;
			std::string opType;
			Op() {}
			Op(const std::vector<std::string>& input, const std::string& output, const std::string& name, const std::string& type) {
				inputNames = input;
				outputName = output;
				opName = name;
				opType = type;
			}
		};

		struct InitializerTensorInfo : public TensorInfo {
			
			unsigned int index;

			InitializerTensorInfo() : TensorInfo() {};
			InitializerTensorInfo(const std::string& n, unsigned int d, TensorType t,  unsigned int i) 
				: TensorInfo(n, d, t) {
				
				index = i;
			};

		};

		struct BindingInfo {
			unsigned int stride; // all initializer data is stored in a single buffer, use stride to indicate
			unsigned int byteSize;
			BindingInfo() = default;
			BindingInfo(unsigned int s, unsigned int w, ) {
				strid = s;
				byteSize = w;
			}
		};
		
		
		//ONNXPARSER_API PERROR CreateParserFromFile(const std::wstring& path_to_onnx, OnnxParser** pOnnxParser);
		//ONNXPARSER_API PERROR GetNetworkInputs(const OnnxParser& pOnnxParser);
		ONNXPARSER_API PERROR ParseFromFile(const std::wstring& path_to_onnx, std::map<std::string, TensorInfo>&, std::map<std::string, TensorInfo>&, std::map<std::string, Op>&, std::map<std::string, InitializerTensorInfo>&, std::vector<BindingInfo>&, char** pweights, unsigned int& weightBytes);
	}
	


}
