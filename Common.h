#pragma once
#ifdef ONNXPARSER_EXPORTS
#define ONNXPARSER_API __declspec(dllexport)
#else
#define ONNXPARSER_API __declspec(dllimport)
#endif
namespace ONNX_PARSER {
	enum class ONNXPARSER_API PERROR {
		O_OK = 0,
		O_NOTFOUND
	};

	enum class ONNXPARSER_API AttributeType {
		UNDEFINED = 0,
		FLOAT = 1,
		INT = 2,
		STRING = 3,
		TENSOR = 4,
		GRAPH = 5,
		SPARSE_TENSOR = 11,
		TYPE_PROTO = 13,
		FLOATS = 6,
		INTS = 7,
		STRINGS = 8,
		TENSORS = 9,
		GRAPHS = 10,
		SPARSE_TENSORS = 12,
		TYPE_PROTOS = 14,
	};

	// Same as dml 0x5100 definition
	enum class ONNXPARSER_API TensorType {
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


}
