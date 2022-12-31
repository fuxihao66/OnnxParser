#include "pch.h"
#include "OnnxParser.h"
#include "onnx.proto3.pb.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "node_attr_helper.h"
using namespace ONNX_PARSER;


TensorInfo::TensorInfo(const std::string& n, unsigned int d, TensorType t) {
	dims = d;
	name = n;
	tensorType = t;
	//shapes = (int64_t*)malloc(d * sizeof(int64_t));
	shapes.resize(d); 
	//= (uint32_t*)malloc(d * sizeof(uint32_t));
}
uint64_t TensorInfo::GetSize() const {
	uint64_t temp = 1;
	for (int i = 0; i < dims; i++) {
		temp *= shapes[i];
	}
	return temp;
}

void TensorInfo::SetShape(unsigned int dim, uint32_t v) {
	if (dim >= dims)
		return;
	shapes[dim] = v;
}

InitializerTensorInfo::InitializerTensorInfo(const std::string& n, unsigned int d, TensorType t, unsigned int i)
	: TensorInfo(n, d, t) {

	index = i;
};

BindingInfo::BindingInfo(unsigned int s, unsigned int w) {
	stride = s;
	byteSize = w;
}

Op::Op() {}
Op::~Op() {
	delete(attriHelper);
}

//Op& Op::operator = (Op&& op) {
//	if (this != &op) {
//		inputNames = std::move(op.inputNames);
//		outputName = std::move(op.outputName);
//		opName = std::move(op.opName);
//		opType = std::move(op.opType);
//		opIndex = op.opIndex;
//
//		additionalAtrribute = std::move(op.additionalAtrribute);
//		attriHelper = std::move(op.attriHelper);
//	}
//	return *this;
//}
//Op::Op(Op&& op) {
//	inputNames = std::move(op.inputNames);
//	outputName = std::move(op.outputName);
//	opName = std::move(op.opName);
//	opType = std::move(op.opType);
//	opIndex = op.opIndex;
//	
//	additionalAtrribute = std::move(op.additionalAtrribute);
//	attriHelper = std::move(op.attriHelper);
//}

Op& Op::operator = (const Op& op) {
	if (this != &op) {
		inputNames = op.inputNames;
		outputName = op.outputName;
		opName = op.opName;
		opType = op.opType;
		opIndex = op.opIndex;

		additionalAtrribute = op.additionalAtrribute;
		attriHelper = new NodeAttrHelper(*op.attriHelper); 
	}
	return *this;
}
Op::Op(const Op& op) {
	inputNames = op.inputNames;
	outputName = op.outputName;
	opName = op.opName;
	opType =  op.opType;
	opIndex = op.opIndex;

	additionalAtrribute = op.additionalAtrribute;
	attriHelper = new NodeAttrHelper(*op.attriHelper);
}


Op::Op(const std::vector<std::string>& input, const std::string& output, const std::string& name, const std::string& type, const unsigned int index) {
	inputNames = input;
	outputName = output;
	opName = name;
	opType = type;
	opIndex = index;
}

Op::Op(const onnx::NodeProto& node, unsigned int index) {
	attriHelper = new NodeAttrHelper(node);
	//attriHelper = std::make_unique<NodeAttrHelper>(node);


	inputNames.resize(node.input_size());
	// https://github.com/onnx/onnx/blob/main/docs/Operators.md
	opType = node.op_type();

	

	for (int i = 0; i < node.input_size(); i++) {
		inputNames[i] = node.input(i);
	}
	outputName = node.output(0);

	opName = node.name();
	opIndex = index;
}
// get input and output tensor info of op node
// void Op::AppendIOInfo(std::map<std::string, TensorInfo>& inputMap, std::map<std::string, TensorInfo>& outputMap, std::map<std::string, InitializerTensorInfo>& initializerMap) {
// 	// input and output tensor might appear in input output and initializer
// 	// unsigned int numInput = inputNames.size();
// 	// inputInfo.resize(numInput);
// 	// for (int i = 0; i < numInput; i++) {
// 	// 	if (inputMap.count(inputNames[i]))
// 	// 		inputInfo[i] = inputMap[inputNames[i]];
// 	// 	else if (initializerMap.count(inputNames[i]))
// 	// 		inputInfo[i] = static_cast<TensorInfo>(initializerMap[inputNames[i]]);
// 	// 	else
// 	// 		assert(false);
// 	// }

// 	// if (outputMap.count(outputName))
// 	// 	outputInfo = outputMap[outputName];
// 	// else if (initializerMap.count(outputName))
// 	// 	outputInfo = static_cast<TensorInfo>(initializerMap[outputName]);
// 	// else
// 	// 	assert(false);


// }

template <typename T>
inline void CopyValToVectorChar(const T val, std::vector<char>& returnVal) {
	const unsigned int length = sizeof(T);
	returnVal.resize(length);
	memcpy(returnVal.data(), &val, length);
}

template <typename T>
inline void CopyVecToVectorChar(const std::vector<T>& valVec, std::vector<char>& returnVal) {
	const unsigned int length = sizeof(T) * valVec.size();
	returnVal.resize(length);
	memcpy(returnVal.data(), valVec.data(), length);
}

inline void CopyTensorToVectorChar(const onnx::TensorProto& tensor, std::vector<char>& returnVal) {
	const char* ptr = nullptr;
	unsigned int byteSize = 1;
	

	for (int i = 0; i < tensor.dims_size(); i++) {
		byteSize *= tensor.dims(i);
	}
	if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
		ptr = tensor.float_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.float_data().data());
		byteSize *= 4;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16) { // according to onnx.proto3
		ptr = tensor.int32_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.int32_data().data());
		byteSize *= 2;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
		ptr = tensor.double_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.double_data().data());
		byteSize *= 8;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT8) {
		ptr = tensor.int32_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.int32_data().data());
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT16) {
		ptr = tensor.int64_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.int32_data().data());
		byteSize *= 2;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT32) {
		ptr = tensor.uint64_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.uint64_data().data());
		byteSize *= 4;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT64) {
		ptr = tensor.uint64_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.uint64_data().data());
		byteSize *= 8;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT8) {
		ptr = tensor.int32_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.int32_data().data());
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT16) {
		ptr = tensor.int64_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.int32_data().data());
		byteSize *= 2;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
		ptr = tensor.int32_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.int32_data().data());
		byteSize *= 4;
	}
	else if (tensor.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
		ptr = tensor.int64_data().empty()
			? reinterpret_cast<const char*>(tensor.raw_data().data())
			: reinterpret_cast<const char*>(tensor.int64_data().data());
		byteSize *= 8;
	}

	returnVal.resize(byteSize);

	memcpy(returnVal.data(), ptr, byteSize);
}
// pimpl
bool Op::GetAttribute(const std::string& attriName, AttributeType attriType, std::vector<char>& returnVal) {

	switch (attriType) {
	case  AttributeType::UNDEFINED:
	case  AttributeType::STRINGS:
	case  AttributeType::GRAPH:
	case  AttributeType::GRAPHS:
	case  AttributeType::TENSORS:
	case  AttributeType::SPARSE_TENSORS:
	case  AttributeType::TYPE_PROTO:
	case  AttributeType::TYPE_PROTOS:
		assert(false);
		return false;
	case  AttributeType::STRING:
	{
		std::string stringVal;
		bool isOk = attriHelper->get(attriName, stringVal);
		if (!isOk)
			return false;
		returnVal.resize(stringVal.size());
		memcpy(returnVal.data(), stringVal.c_str(), stringVal.size());
		return true;
	}	
	case  AttributeType::FLOAT:
	{
		float floatVal;
		bool isOk = attriHelper->get(attriName, floatVal);
		if (!isOk)
			return false;
		CopyValToVectorChar(floatVal, returnVal);
		return true;
	}
	case  AttributeType::FLOATS:
	{
		std::vector<float> floatsVal;
		bool isOk = attriHelper->get(attriName, floatsVal);
		if (!isOk)
			return false;
		CopyVecToVectorChar(floatsVal, returnVal);
		return true;
	}
	case  AttributeType::INT:
	{
		int intVal;
		bool isOk = attriHelper->get(attriName, intVal);
		if (!isOk)
			return false;
		CopyValToVectorChar(intVal, returnVal);
		return true;
	}
	case  AttributeType::INTS:
	{
		std::vector<int> intsVal;
		bool isOk = attriHelper->get(attriName, intsVal);
		if (!isOk)
			return false;
		CopyVecToVectorChar(intsVal, returnVal);
		return true;
	}
	case  AttributeType::TENSOR:
	{	
		onnx::TensorProto tensorVal;
		bool isOk = attriHelper->get(attriName, tensorVal);
		if (!isOk) {
			// search in additionalAtrribute
			if (additionalAtrribute.count(attriName)) {
				tensorVal = additionalAtrribute[attriName];
			}
			else
				return false;
		}
		CopyTensorToVectorChar(tensorVal, returnVal);
		return true;
	}
	/*case AttributeType::SPARSE_TENSOR:
		auto warppedSparseTensorVal = attriHelper->get<attriType, onnx::SparseTensorProto>(attriName);*/
	default:
		assert(false);
		return false;
	}
}


std::unordered_map<unsigned int, TensorType> g_protoTensorType2DmlType = {
	{onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED,		TensorType::UKNOWN},
	{onnx::TensorProto_DataType::TensorProto_DataType_FLOAT,			TensorType::FLOAT},
	{onnx::TensorProto_DataType::TensorProto_DataType_UINT8,			TensorType::UINT8},
	{onnx::TensorProto_DataType::TensorProto_DataType_INT8,				TensorType::INT8},
	{onnx::TensorProto_DataType::TensorProto_DataType_UINT16,			TensorType::UINT16},
	{onnx::TensorProto_DataType::TensorProto_DataType_INT16,			TensorType::INT16},
	{onnx::TensorProto_DataType::TensorProto_DataType_INT32,			TensorType::INT32},
	{onnx::TensorProto_DataType::TensorProto_DataType_INT64,			TensorType::INT64},
	{onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16,			TensorType::FLOAT16},
	{onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE,			TensorType::DOUBLE},
	{onnx::TensorProto_DataType::TensorProto_DataType_UINT32,			TensorType::UINT32},
	{onnx::TensorProto_DataType::TensorProto_DataType_UINT64,			TensorType::UINT64},
};

ONNXPARSER_API TensorType ONNX_PARSER::OnnxTensorType2DmlTensorType(unsigned int onnxTensorType){
	return g_protoTensorType2DmlType[onnxTensorType];
}

OnnxParser::OnnxParser(const std::wstring& path_to_onnx) {//google::protobuf::io::FileInputStream* fileStream
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
		//return PERROR::O_NOTFOUND;
		throw std::exception("file not found");
	}

	if (0 > file_descriptor) {
		//return PERROR::O_NOTFOUND;
		throw std::exception("file not found");
	}

	model = new onnx::ModelProto();

	google::protobuf::io::FileInputStream stream(file_descriptor);
	stream.SetCloseOnDelete(true);
	model->ParseFromZeroCopyStream(&stream);

	ParseGraphInitializers(); // need to be done first
	ParseGraphNodes();
	ParseInputs();
	ParseOutputs();
	//ParseGraphNodes();
}
OnnxParser::~OnnxParser() {
	delete(model);
}

int64_t OnnxParser::GetIrVersion() const {
	return model->ir_version();
}
std::string OnnxParser::GetProducerName() const {
	return model->producer_name();
}

int64_t OnnxParser::GetOpsetVersion() const {
	return model->opset_import().Get(0).version();
}

const std::map<std::string, TensorInfo>& OnnxParser::GetInputs()const {
	return inputMap;
}
const std::map<std::string, TensorInfo>& OnnxParser::GetOutputs() const {
	return outputMap;
}
//std::map<std::string, Op> OnnxParser::GetGraphNodes() const {
//	return nodeMap;
//}

const std::vector<BindingInfo>& OnnxParser::GetBindings() const {
	return bindings;
}
const std::map<std::string, Op>& OnnxParser::GetGraphNodes() const {
	return nodeMap;
}


const std::map<std::string, InitializerTensorInfo>& OnnxParser::GetGraphInitializers() const {
	return initializerMap;
}
// pass a pointer to weight pointer, copy data, and return data in byte size
const std::vector<char>& OnnxParser::GetWeights() const {
	
	/*Weights.resize(weightValues.size());
	memcpy(Weights.data(), weightValues.data(), weightValues.size());*/
	return weightValues;
}

void OnnxParser::ParseInputs() {
	const auto& graph = model->graph();

	for (int i = 0; i < graph.input_size(); i++) {
		const auto& input = graph.input(i);
		const auto& shape = input.type().tensor_type().shape();

		if (initializerMap.count(input.name())) { // some onnx file takes the initializer as input, which need to be excluded
			continue;
		}
		TensorType tensorType = g_protoTensorType2DmlType[input.type().tensor_type().elem_type()];


		auto tf = TensorInfo(input.name(), shape.dim_size(), tensorType);

		for (int n = 0; n < shape.dim_size(); n++) {
			tf.SetShape(n, shape.dim(n).dim_value());
		}

		inputMap[input.name()] = tf;
	}
}
void OnnxParser::ParseOutputs() {
	const auto& graph = model->graph();

	for (int i = 0; i < graph.output_size(); i++) {
		const auto& output = graph.output(i);
		const auto& shape = output.type().tensor_type().shape();

		TensorType tensorType = g_protoTensorType2DmlType[output.type().tensor_type().elem_type()];

		auto tf = TensorInfo(output.name(), shape.dim_size(), tensorType);

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
void Op::AppendAdditionAttribute(const onnx::TensorProto& attriTensor, const  std::string& name) {
	additionalAtrribute[name] = attriTensor;
}
//void Op::SetOutputInfo(const onnx::TensorProto& attriTensor, const  std::string& name) {
//
//}

void OnnxParser::ParseGraphNodes() {
	const auto& graph = model->graph();

	for (int i = 0; i < graph.node_size(); i++) {
		const auto& node = graph.node(i);



		Op op(node, i);

		//op.SetOutputInfo(initializerMetaData[op.outputName]); // needed for some operators, which need output info

		// TODO: ugly patch
		const unsigned int opsetVer = static_cast<unsigned int>(model->opset_import().Get(0).version());
		if (node.op_type() == "Slice" && opsetVer >= 10) { // need attribute from initializer
			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[1]], "starts");
			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[2]], "ends");
			if (op.inputNames.size() >= 4)
				op.AppendAdditionAttribute(initializerMetaData[op.inputNames[3]], "axes");
			if (op.inputNames.size() >= 5)
				op.AppendAdditionAttribute(initializerMetaData[op.inputNames[4]], "steps");
		}
		else if (node.op_type() == "Upsample" && opsetVer == 9) {
			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[1]], "scales");
		}
		else if (node.op_type() == "Pad" && opsetVer >= 11) {
			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[1]], "pads");
			if (op.inputNames.size() >= 3)
				op.AppendAdditionAttribute(initializerMetaData[op.inputNames[2]], "constant_value");
		}
		// else if (node.op_type() == "Cast"){ // need cast type 
		// 	op.AppendAdditionAttribute(initializerMetaData[op.inputNames[1]], "target_type");
		// }

		nodeMap[op.outputName] = op;// std::move(op);
	}
}

//void OnnxParser::GetGraphNodes(std::map<std::string, Op>& nodeMap)  {
//	const auto& graph = model->graph();
//
//	for (int i = 0; i < graph.node_size(); i++) {
//		const auto& node = graph.node(i);
//
//
//
//		Op op(node, i);
//		
//		//op.SetOutputInfo(initializerMetaData[op.outputName]); // needed for some operators, which need output info
//		
//
//		// TODO: ugly patch
//		const unsigned int opsetVer = static_cast<unsigned int>(model->opset_import().Get(0).version());
//		if (node.op_type() == "Slice" && opsetVer >= 10) { // need attribute from initializer
//			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[1]], "starts");
//			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[2]], "ends");
//			if (op.inputNames.size() >= 4)
//				op.AppendAdditionAttribute(initializerMetaData[op.inputNames[3]], "axes");
//			if (op.inputNames.size() >= 5)
//				op.AppendAdditionAttribute(initializerMetaData[op.inputNames[4]], "steps");
//		}
//		else if (node.op_type() == "Upsample" && opsetVer == 9) {
//			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[1]], "scales");
//		}
//		else if (node.op_type() == "Pad" && opsetVer >= 11) {
//			op.AppendAdditionAttribute(initializerMetaData[op.inputNames[1]], "pads");
//			if (op.inputNames.size() >= 3)
//				op.AppendAdditionAttribute(initializerMetaData[op.inputNames[2]], "constant_value");
//			
//		}
//		// op.AppendIOInfo(inputMap, outputMap, initializerMap);
//		
//		nodeMap[op.outputName] = std::move(op);
//	}
//}
void OnnxParser::ParseGraphInitializers() {
	unsigned int stride = 0;
	weightValues.resize(10000);
	bindings.reserve(10000);
	unsigned int index = 0;

	const auto& graph = model->graph();

	

	for (int i = 0; i < graph.initializer_size(); i++) {
		const auto& initializer = graph.initializer(i);

		if (g_protoTensorType2DmlType.count(initializer.data_type()) == 0)
			throw std::exception("Unsupported data type");
		TensorType tensorType = g_protoTensorType2DmlType[initializer.data_type()];


		const char* ptr = nullptr;
		unsigned int typeBytes;
		// reference: onnx-runtime/onnx_converter.cc
		if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
			ptr = initializer.float_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.float_data().data());
			typeBytes = 4;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16) { // according to onnx.proto3
			ptr = initializer.int32_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.int32_data().data());
			typeBytes = 2;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
			ptr = initializer.double_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.double_data().data());
			typeBytes = 8;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT8) {
			ptr = initializer.int32_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.int32_data().data());
			typeBytes = 1;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT16) {
			ptr = initializer.int64_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.int32_data().data());
			typeBytes = 2;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT32) {
			ptr = initializer.uint64_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.uint64_data().data());
			typeBytes = 4;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_UINT64) {
			ptr = initializer.uint64_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.uint64_data().data());
			typeBytes = 8;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT8) {
			ptr = initializer.int32_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.int32_data().data());
			typeBytes = 1;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT16) {
			ptr = initializer.int64_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.int32_data().data());
			typeBytes = 2;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
			ptr = initializer.int32_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.int32_data().data());
			typeBytes = 4;
		}
		else if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
			ptr = initializer.int64_data().empty()
				? reinterpret_cast<const char*>(initializer.raw_data().data())
				: reinterpret_cast<const char*>(initializer.int64_data().data());
			typeBytes = 8;
		}
		else {
			assert(0);
		}

		auto ComputeWeightByteSize = [&]() {
			unsigned int arraySize = 1;
			for (int n = 0; n < initializer.dims_size(); n++) {
				arraySize *= initializer.dims(n);
			}
			return arraySize * typeBytes;
		};

		const unsigned int weightBytes = ComputeWeightByteSize();
		if (stride + weightBytes > weightValues.size())
			weightValues.resize((stride + weightBytes) * 2);

		memcpy(weightValues.data() + stride, ptr, weightBytes);

		// 
		
		
		auto tf = InitializerTensorInfo(initializer.name(), initializer.dims_size(), tensorType, index);
		for (int n = 0; n < initializer.dims_size(); n++) {
			tf.SetShape(n, initializer.dims(n));
		}
		initializerMetaData[initializer.name()] = initializer;// stored for graph node parsing
		initializerMap[initializer.name()] = tf;
		bindings.push_back(BindingInfo(stride, weightBytes));

		
		stride += weightBytes;
		index += 1;
	}

	// interpret constant node as initializer
	for (int i = 0; i < graph.node_size(); i++) {
		const auto& node = graph.node(i);
		if (node.op_type() == "Constant"){
			auto attriHelper = NodeAttrHelper(node);
			onnx::TensorProto const_data_tensor;
			attriHelper.get("value", const_data_tensor);
			std::vector<char> const_data;
			CopyTensorToVectorChar(const_data_tensor, const_data);
			

			const unsigned int weightBytes = const_data.size();
			if (stride + weightBytes > weightValues.size())
				weightValues.resize((stride + weightBytes) * 2);

			memcpy(weightValues.data() + stride, const_data.data(), weightBytes);

			if (g_protoTensorType2DmlType.count(const_data_tensor.data_type()) == 0)
				throw std::exception("Unsupported data type");
			TensorType tensorType = g_protoTensorType2DmlType[const_data_tensor.data_type()];
			auto tensorName = node.output(0);
			auto tf = InitializerTensorInfo(tensorName, const_data_tensor.dims_size(), tensorType, index);
			for (int n = 0; n < const_data_tensor.dims_size(); n++) {
				tf.SetShape(n, const_data_tensor.dims(n));
			}

			initializerMap[tensorName] = tf;
			bindings.push_back(BindingInfo(stride, weightBytes));
			
			stride += weightBytes;
			index += 1;

		}
	}

	weightValues.resize(stride);
}

ONNXPARSER_API PERROR ONNX_PARSER::ParseFromFile(const std::wstring& path_to_onnx, std::map<std::string, TensorInfo>& inputMap, std::map<std::string, TensorInfo>& outputMap, std::map<std::string, Op>& graphNodes, std::map<std::string, InitializerTensorInfo>& graphInitializers, std::vector<BindingInfo>& bindings, std::vector<char>& weights, unsigned int& opsetVersion)
{
	


	//OnnxParser* parser = new OnnxParser(&stream);
	OnnxParser* parser = new OnnxParser(path_to_onnx);

	/*auto& inputs = parser->GetInputs();
	bindings.push_back(BindingInfo());*/
	//outputMap = parser->GetOutputs();
	////graphNodes = parser->GetGraphNodes();
	//parser->GetGraphNodes(graphNodes);
	//graphInitializers = parser->GetGraphInitializers();
	//bindings = parser->GetBindings();
	//parser->GetWeights(weights);
	//// weightBytes = parser->GetWeights(weights);

	opsetVersion = static_cast<unsigned int>(parser->GetOpsetVersion());

	delete(parser);

	return PERROR::O_OK;
}
//
//ONNXPARSER_API PERROR ONNX_PARSER::TestFunction(const std::wstring& path_to_onnx, std::map<std::string, TensorInfo>& inputMap, std::map<std::string, TensorInfo>& outputMap, std::map<std::string, Op>& graphNodes, std::map<std::string, InitializerTensorInfo>& graphInitializers, std::vector<BindingInfo>& bindings, std::vector<char>& weights, unsigned int& opsetVersion) {//std::map<std::string, TensorInfo>* inputMap
//	GOOGLE_PROTOBUF_VERIFY_VERSION;
//	int file_descriptor;
//	_wsopen_s(
//		&file_descriptor,
//		path_to_onnx.c_str(),
//		O_RDONLY | _O_SEQUENTIAL | _O_BINARY,
//		_SH_DENYWR,
//		_S_IREAD | _S_IWRITE);
//	errno_t err = 0;
//	_get_errno(&err);
//	if (err == ENOENT) {
//		return PERROR::O_NOTFOUND;
//	}
//
//	if (0 > file_descriptor) {
//		return PERROR::O_NOTFOUND;
//	}
//
//	google::protobuf::io::FileInputStream stream(file_descriptor);
//	stream.SetCloseOnDelete(true);
//
//	OnnxParser* parser = new OnnxParser(&stream);
//
//	//inputMap = parser->GetInputs();
//
//	delete(parser);
//	return PERROR::O_OK;
//
//}

	
