#include "pch.h"
#include "OnnxParser.h"
#include "onnx.proto3.pb.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
using namespace ONNX_DML;

class OnnxParser {
private:
	onnx::ModelProto model;
	std::vector<char> weightValues;

	std::map<std::string, TensorInfo> inputMap;
	std::map<std::string, TensorInfo> outputMap;
	std::map<std::string, Op> nodeMap;
	std::map<std::string, InitializerTensorInfo> initializerMap;
	std::vector<BindingInfo> bindings;

	void ParseInputs();
	void ParseOutputs();
	void ParseGraphNodes(); // TODO: only support single output node
	void ParseGraphInitializers();
public:
	int64_t GetIrVersion() const;
	std::string GetProducerName() const;
	// structure
	std::map<std::string, TensorInfo> GetInputs() const;
	std::map<std::string, TensorInfo> GetOutputs() const;
	std::map<std::string, Op> GetGraphNodes() const;
	std::map<std::string, InitializerTensorInfo> GetGraphInitializers() const;
	std::vector<BindingInfo> GetBindings() const;
	// data
	unsigned int GetWeights(char** ppWeights) const;

};


OnnxParser::OnnxParser(google::protobuf::io::FileInputStream* fileStream) {
	model.ParseFromZeroCopyStream(fileStream);

	ParseInputs();
	ParseOutputs();
	ParseGraphNodes();
	ParseGraphInitializers();
}

int64_t OnnxParser::GetIrVersion() const {
	return model.ir_version();
}
std::string OnnxParser::GetProducerName() const {
	return model.producer_name();
}

std::map<std::string, TensorInfo> OnnxParser::GetInputs() const {
	return inputMap;
}
std::map<std::string, TensorInfo> OnnxParser::GetOutputs() const {
	return outputMap;
}
std::map<std::string, Op> OnnxParser::GetGraphNodes() const {
	return nodeMap;
}

std::vector<BindingInfo> OnnxParser::GetBindings() const {
	return bindings;
}


std::map<std::string, InitializerTensorInfo> OnnxParser::GetGraphInitializers() const {
	return initializerMap;
}
// pass a pointer to weight pointer, copy data, and return data in byte size
unsigned int OnnxParser::GetWeights(char** ppWeights) const {
	free(*ppWeights);
	*ppWeights = (char*)malloc(weightValues.size());
	memcpy(*ppWeights, weightValues.data(), weightValues.size());
	return weightValues.size();
}

void OnnxParser::ParseInputs() {
	for (int i = 0; i < graph.input_size(); i++) {
		const auto& input = graph.input(i);
		const auto& shape = input.type().tensor_type().shape();

		auto tf = TensorInfo(input.name(), shape.dim_size());

		for (int n = 0; n < shape.dim_size(); n++) {
			tf.SetShape(n, shape.dim(n).dim_value());
		}

		inputMap[input.name()] = tf;
	}
}
void OnnxParser::ParseOutputs() {
	for (int i = 0; i < graph.output_size(); i++) {
		const auto& output = graph.output(i);
		const auto& shape = output.type().tensor_type().shape();

		auto tf = TensorInfo(output.name(), shape.dim_size());

		for (int n = 0; n < shape.dim_size(); n++) {
			tf.SetShape(n, shape.dim(n).dim_value());
		}

		outputMap[output.name()] = tf;
	}
}


// onnx-runtime/core/providers/rknpu/onnx_converter.cc OnnxConverter::Convert
//if (op == "Conv") {
//	const auto strides = helper.get("strides", vector<int>{1, 1});
//	const auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
//	const auto dilations = helper.get("dilations", vector<int>{1, 1});
//	const auto group = helper.get("group", 1);
//	std::string bias;
//	if (node.input_size() >= 3) {
//		bias = m(node.input(2));
//	}
//	const auto auto_pad = helper.get("auto_pad", "NOTSET");
//
//	const auto ori_weight = m(node.input(1));
//	AddConv(m(node.input(0)), strides, pads, dilations, group,
//		ori_weight, bias, auto_pad, m(node.output(0)));
//}
//else if (op == "QLinearConv") {
//	const auto strides = helper.get("strides", vector<int>{1, 1});
//	const auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
//	const auto dilations = helper.get("dilations", vector<int>{1, 1});
//	const auto group = helper.get("group", 1);
//	const auto auto_pad = helper.get("auto_pad", "NOTSET");
//	std::string bias;
//	if (node.input_size() >= 9) {
//		bias = m(node.input(8));
//	}
//	AddQLinearConv(m(node.input(0)), m(node.input(1)), m(node.input(2)),
//		strides, pads, dilations, group, auto_pad,
//		m(node.input(3)), m(node.input(4)), m(node.input(5)),
//		bias, m(node.output(0)), m(node.input(6)),
//		m(node.input(7)));
//}
//else if (op == "AveragePool" || op == "MaxPool" ||
//	op == "GlobalAveragePool" || op == "GlobalMaxPool") {
//	const auto input = m(node.input(0));
//	const auto output = m(node.output(0));
//	vector<int> strides, pads, kernel_shape;
//	int ceil_mode;
//	if (op == "AveragePool" || op == "MaxPool") {
//		strides = helper.get("strides", vector<int>{1, 1});
//		pads = helper.get("pads", vector<int>{0, 0, 0, 0});
//		kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
//		ceil_mode = helper.get("ceil_mode", 0);
//		const auto count_include_pad =
//			helper.get("count_include_pad", 0);
//		if (count_include_pad == 1) {
//			throw std::invalid_argument(
//				"count_include_pad == 1 is not supported");
//		}
//		const auto storage_order = helper.get("storage_order", 0);
//		if (storage_order == 1) {
//			throw std::invalid_argument(
//				"storage_order == 1 is not supported");
//		}
//		if (helper.get("auto_pad", "NOTSET") != "NOTSET") {
//			throw std::invalid_argument("auto_pad is not supported");
//		}
//	}
//	else {
//		strides = { 0, 0 };
//		pads = { 0, 0, 0, 0 };
//		kernel_shape = { -1, -1 };  // -1 for global
//		ceil_mode = 0;
//	}
//	AddLayerPool(op, input, kernel_shape, pads, strides, ceil_mode,
//		output);
//}


void OnnxParser::ParseGraphNodes() {

	for (int i = 0; i < graph.node_size(); i++) {
		const auto& node = graph.node(i);

		std::vector<std::string> inputNames(node.input_size());

		// https://github.com/onnx/onnx/blob/main/docs/Operators.md
		std::string opType = node.op_type();

		for (int n = 0; n < node.input_size(); n++) {
			inputNames[n] = node.input(n);
		}

		auto outputName = node.output(0);

		Op op(inputNames, outputName, node.name(), opType);
		nodeMap[outputName] = op;
	}
}
void OnnxParser::ParseGraphInitializers() {
	unsigned int stride = 0;
	weightValues.resize(10000);
	bindings.reserve(10000);
	unsigned int index = 0;
	for (int i = 0; i < graph.initializer_size(); i++) {
		const auto& initializer = graph.initializer(i);

		char* ptr;
		unsigned int typeBytes;
		// reference: onnx-runtime/onnx_converter.cc
		if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
			ptr = initializer.float_data().empty()
				? initializer.raw_data().data()
				: reinterpret_cast<const char*>(
					initializer.float_data().data());
			typeBytes = 4;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
			ptr = initializer.float_data().empty()
				? initializer.raw_data().data()
				: reinterpret_cast<const char*>(
					initializer.float_data().data());
			typeBytes = 2;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT8) {
			ptr = initializer.int32_data().empty()
				? initializer.raw_data().data()
				: reinterpret_cast<const char*>(
					initializer.int32_data().data());
			typeBytes = 1;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
			ptr = initializer.int32_data().empty()
				? initializer.raw_data().data()
				: reinterpret_cast<const char*>(
					initializer.int32_data().data());
			typeBytes = 4;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
			ptr = initializer.int64_data().empty()
				? initializer.raw_data().data()
				: reinterpret_cast<const char*>(
					initializer.int64_data().data());
			typeBytes = 8;
		}
		else {
			assert(0);
		}

		auto ComputeWeightByteSize = [&]() {
			unsigned int arraySize = 1;
			for (int n = 0; n < initializer.dim_size(); n++) {
				arraySize *= initializer.dims(n);
			}
			return arraySize * typeBytes;
		};

		const unsigned int weightBytes = ComputeWeightByteSize();
		if (stride + weightBytes > weightValues.size())
			weightValues.resize((stride + weightBytes) * 2);

		memcpy(weightValues.data() + stride, ptr, weightBytes);

		// 
		auto tf = InitializerTensorInfo(initializer.name(), initializer.dim_size(), initializer.data_type(), index);
		for (int n = 0; n < initializer.dim_size(); n++) {
			tf.SetShape(n, initializer.dims(n));
		}

		initializerMap[initializer.name()] = tf;
		bindings.push_back(BindingInfo(stride, weightBytes));

		
		stride += weightBytes;
		index += 1;
	}

	weightValues.resize(stride);
}

extern "C" {

	ONNXPARSER_API PERROR ParseFromFile(const std::wstring& path_to_onnx, std::map<std::string, TensorInfo>& inputMap, std::map<std::string, TensorInfo>& outputMap, std::map<std::string, Op>& graphNodes, std::map<std::string, InitializerTensorInfo> graphInitializers, std::vector<BindingInfo>& bindings, char** pweights, unsigned int& weightBytes)
    {
		GOOGLE_PROTOBUF_VERIFY_VERSION;

		int file_descriptor;
		_wsopen_s(
			&file_descriptor,
			path_to_onnx.c_str(),
			O_RDONLY | _O_SEQUENTIAL | _O_BINARY,
			_SH_DENYWR,
			_S_IREAD | _S_IWRITE);
		errno_t err = 0;
		_get_errno(&err);
		if (err == ENOENT) {
			return PERROR::O_NOTFOUND;
		}

		if (0 > file_descriptor) {
			return PERROR::O_NOTFOUND;
		}

		google::protobuf::io::FileInputStream stream(file_descriptor);
		stream.SetCloseOnDelete(true);


		OnnxParser* parser = new OnnxParser(&stream);

		inputMap = parser->GetInputs();
		outputMap = parser->GetOutputs();
		graphNodes = parser->ParseGraphNodes();
		graphInitializers = parser->GetGraphInitializers();
		bindings = parser->GetBindings();
		weightBytes = parser->GetWeights(pweights);

		free(parser);

		return PERROR::O_OK;
    }
}