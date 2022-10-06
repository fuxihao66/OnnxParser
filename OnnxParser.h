
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


//message TensorProto{
//  enum DataType {
//	UNDEFINED = 0;
//	// Basic types.
//	FLOAT = 1;   // float
//	UINT8 = 2;   // uint8_t
//	INT8 = 3;    // int8_t
//	UINT16 = 4;  // uint16_t
//	INT16 = 5;   // int16_t
//	INT32 = 6;   // int32_t
//	INT64 = 7;   // int64_t
//	STRING = 8;  // string
//	BOOL = 9;    // bool
//
//	// IEEE754 half-precision floating-point format (16 bits wide).
//	// This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
//	FLOAT16 = 10;
//
//	DOUBLE = 11;
//	UINT32 = 12;
//	UINT64 = 13;
//	COMPLEX64 = 14;     // complex with float32 real and imaginary components
//	COMPLEX128 = 15;    // complex with float64 real and imaginary components
//
//	// Non-IEEE floating-point format based on IEEE754 single-precision
//	// floating-point number truncated to 16 bits.
//	// This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
//	BFLOAT16 = 16;
//
//	// Future extensions go here.
//}

extern "C" {
	namespace ONNX_DML {
		enum PERROR {
			O_OK = 0,
			O_NOTFOUND
		};

		// Same as dml 0x5100 definition
		enum TensorType {
			UKNOWN = 0,
			FLOAT,   // Default type, need to be casted to fp16 when upload resource to GPU
			FLOAT16,
			UINT32,
			UINT16,
			UINT8,
			INT32,
			INT16,
			INT8,
			DOUBLE,
			UINT64,
			INT64,
		};

		struct TensorInfo {
			TensorType tensorType;
			std::string name;
			unsigned int  dims;
			uint32_t* shapes;   // TODO: different from onnx Shape definition (which is int64_t)

			TensorInfo() dims(0), shapes(nullptr) {}
			TensorInfo(const std::string& n, unsigned int d, TensorType t) {
				dims = d;
				name = n;
				tensorType = t;
				shapes = (int64_t*)malloc(d * sizeof(int64_t));
			}

			uint64_t GetSize() const {
				uint64_t temp = 1;
				for (int i = 0; i < dims; i++) {
					temp *= shapes[i];
				}
			}
			void SetShape(unsigned int dim, int64_t v) {
				if (dim >= dims)
					return;
				shapes[dim] = v;
			}
			
		};

		
		struct Op {
			std::vector<std::string> inputNames;

			std::vector<TensorInfo> inputInfo;

			std::string outputName;
			TensorInfo outputInfo;
			/*std::optional<std::vector<int64_t>> inputShape;
			std::optional<std::vector<int64_t>> outputShape;*/
			std::string opName;
			std::string opType;

			/*TensorType inputTensorType;
			TensorType outputTensorType;*/
			unsigned int opIndex; // operator index inside network graph
			Op() {}
			Op(const std::vector<std::string>& input, const std::string& output, const std::string& name, const std::string& type, const unsigned int index) {
				inputNames = input;
				outputName = output;
				opName = name;
				opType = type;
				opIndex = index;
			}

			void GetAttribute(const std::string& attriName, void* returnVal);
		private:
			class AttributeHelper;

			std::unique_ptr<AttributeHelper> attriHelper;
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
