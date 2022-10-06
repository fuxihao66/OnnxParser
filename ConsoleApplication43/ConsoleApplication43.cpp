// ConsoleApplication43.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <optional>
#include "onnx.proto3.pb.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"


#include "node_attr_helper.h"
//static Status LoadOrtModelBytes(const PathString& model_uri,
//	gsl::span<const uint8_t>& bytes,
//	std::vector<uint8_t>& bytes_data_holder) {
//	size_t num_bytes = 0;
//	ORT_RETURN_IF_ERROR(Env::Default().GetFileLength(model_uri.c_str(), num_bytes));
//
//	bytes_data_holder.resize(num_bytes);
//
//	std::ifstream bytes_stream(model_uri, std::ifstream::in | std::ifstream::binary);
//	bytes_stream.read(reinterpret_cast<char*>(bytes_data_holder.data()), num_bytes);
//
//	if (!bytes_stream) {
//		return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
//			"Load model from ", ToUTF8String(model_uri), " failed. Only ",
//			bytes_stream.gcount(), "/", num_bytes, " bytes were able to be read.");
//	}
//
//	bytes = gsl::span<const uint8_t>(bytes_data_holder.data(), num_bytes);
//
//	return Status::OK();
//}

template<unsigned int T>
struct TensorInfo {
	std::array<unsigned int, T> Dims;

};

int main() {
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	int file_descriptor;


	onnx::ModelProto model;
	
	std::wstring wide_path(L"D:/candy-9.onnx");
	//std::wstring wide_path(L"D:/UNetLWGated.onnx");
	_set_errno(0);  // clear errno

	_wsopen_s(
		&file_descriptor,
		wide_path.c_str(),
		O_RDONLY | _O_SEQUENTIAL | _O_BINARY,
		_SH_DENYWR,
		_S_IREAD | _S_IWRITE);
	errno_t err = 0;
	_get_errno(&err);
	if (err == ENOENT) {
		std::cout <<  "Model file not found!";
	}

	if (0 > file_descriptor) {
		std::cout << "Model file not found!";

	}
	google::protobuf::io::FileInputStream stream(file_descriptor);
	stream.SetCloseOnDelete(true);

	model.ParseFromZeroCopyStream(&stream);

	/*std::ifstream modelStream(model_file, std::ios::in | std::ios::binary);
	model.ParseFromIstream(&modelStream);*/

	std::cout
		<< model.ir_version() << "\n"
		<< model.producer_name() << std::endl;

	std::cout << "opset domain is "  << model.opset_import().Get(0).domain() << std::endl;
	std::cout << "opset version is "  << model.opset_import().Get(0).version() << std::endl;

	const auto& graph = model.graph();

	std::cout << "---- inputs ----" << std::endl;
	for (int i = 0; i < graph.input_size(); i++) {
		const auto& input = graph.input(i);
		std::cout << input.name() << "\t";
		const auto& tensor_type = input.type().tensor_type();
		const auto& shape = tensor_type.shape();
		std::cout << tensor_type.elem_type() << "[";
		for (int n = 0; n < shape.dim_size(); n++) {
			if (n != 0) std::cout << ",";
			std::cout << shape.dim(n).dim_value();
		}
		std::cout << "]" << std::endl;
	}

	/*std::cout << "---- outputs ----" << std::endl;
	for (int i = 0; i < graph.output_size(); i++) {
		const auto& output = graph.output(i);
		std::cout << output.name() << "\t";
		const auto& tensor_type = output.type().tensor_type();
		const auto& shape = tensor_type.shape();
		std::cout << tensor_type.elem_type() << "[";
		for (int n = 0; n < shape.dim_size(); n++) {
			if (n != 0) std::cout << ",";
			std::cout << shape.dim(n).dim_value();
		}
		std::cout << "]" << std::endl;
	}*/

	// opset is 9, how to get?

	std::cout << "---- nodes ----" << std::endl;
	for (int i = 0; i < graph.node_size(); i++) {
		const auto& node = graph.node(i);

		onnxruntime::NodeAttrHelper helper(node); // parse attribute (eg: pad)

		

		if (node.op_type() == "InstanceNormalization") {
			std::cout << node.op_type() << std::endl;

			float returnVal = helper.get("epsilon", 1e-3f);

			std::cout << "epsilon is " << returnVal << std::endl;
		}
		else if (node.op_type() == "Constant") { // TODO: not done
			std::cout << node.op_type() << std::endl;
			// opset version 1 or 9 (9 support all data type, while 1 only support float)
			auto returnVal = helper.get("value");
			// anonymous tensor, only use its data
			if (returnVal.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
				const float* ptr = returnVal.float_data().empty()
					? reinterpret_cast<const float*>(returnVal.raw_data().data())
					: returnVal.float_data().data();

				//const float* ptr = reinterpret_cast<const float*>(initializer.raw_data().data());

				std::cout << ptr[0] << std::endl;
			}
			else if (returnVal.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
				const int* ptr = returnVal.int32_data().empty()
					? reinterpret_cast<const int*>(returnVal.raw_data().data())
					: returnVal.int32_data().data();

				//const float* ptr = reinterpret_cast<const float*>(initializer.raw_data().data());

				std::cout << ptr[0] << std::endl;
			}
			else if (returnVal.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
				const long long* ptr = returnVal.int32_data().empty()
					? reinterpret_cast<const long long*>(returnVal.raw_data().data())
					: reinterpret_cast<const long long*>(returnVal.int32_data().data());

				//const float* ptr = reinterpret_cast<const float*>(initializer.raw_data().data());

				std::cout << ptr[0] << std::endl;
			}

			// 
			// 
			// 
			// opset version 11
			//auto returnVal = helper.get("value", std::vector<float>{});
			// auto returnVal = helper.get("sparse_value", std::vector<float>{});
			// 
			// 
			// opset version 12 13
			// auto returnVal = helper.get("value", std::vector<float>{});
			// auto returnVal = helper.get("sparse_value", std::vector<float>{});
			//auto returnVal = helper.get("value_float", 10.f);
			//auto returnVal = helper.get("value_floats", std::vector<float>{});
			// auto returnVal = helper.get("value_int", 10.f);
			//auto returnVal = helper.get("value_ints", std::vector<float>{});
			//std::cout << returnVal;
			/*for (int i = 0; i < returnVal.size(); i++) {
				std::cout << returnVal[i] << " ";
			}
			std::cout << std::endl;*/
		}
		else if (node.op_type() == "Cast") {
			std::cout << node.op_type() << std::endl;
			auto returnVal = helper.get("to", 10); // tensor proto type
			std::cout << returnVal << std::endl;
		}
		else if (node.op_type() == "Conv") {
			std::cout << node.op_type() << std::endl;
			auto returnVal0 = helper.get("auto_pad", "NOTSET");
			auto returnVal1 = helper.get("dilations", std::vector<int>{1, 1});
			auto returnVal2 = helper.get("group", 10);
			auto returnVal3 = helper.get("kernel_shape", std::vector<int>{5, 5});
			auto returnVal4 = helper.get("pads", std::vector<int>{});
			//auto returnVal4 = helper.get("pads", std::vector<int>{0, 0, 0, 0});
			auto returnVal5 = helper.get("strides", std::vector<int>{1, 1});
			std::cout << "dilations is " << returnVal1[0] << " " << returnVal1[1] << std::endl;
			std::cout << "kernel shape is " << returnVal3[0] << " " << returnVal3[1] << std::endl;
			std::cout << "pads  is ";
			for (int i = 0; i < returnVal4.size(); i++)
				std::cout << returnVal4[i] << " ";
			std::cout << std::endl;
			std::cout << "strides is " << returnVal5[0] << " " << returnVal5[1] << std::endl;

		}
		else if (node.op_type() == "Gather") {
			std::cout << node.op_type() << std::endl;
			auto returnVal = helper.get("axis", 0);
			std::cout << returnVal << std::endl;
		}
		else if (node.op_type() == "Pad") { 
			auto returnVal0 = helper.get("mode", "constant");

			//// version  1 
			//std::cout << node.op_type() << std::endl;
			//auto returnVal1 = helper.get("paddings", std::vector<int>{});
			//auto returnVal2 = helper.get("value", 0.f);
			//for (int i = 0; i < returnVal1.size(); i++) {
			//	std::cout << returnVal1[i] << " "; 
			//}
			//std::cout << std::endl;
			//std::cout <<  returnVal2 << std::endl;

			// version  2 
			std::cout << node.op_type() << std::endl;
			auto returnVal1 = helper.get("pads", std::vector<int>{});
			auto returnVal2 = helper.get("value", 0.f);
			for (int i = 0; i < returnVal1.size(); i++) {
				std::cout << returnVal1[i] << " ";
			}
			std::cout << std::endl;
			std::cout << returnVal2 << std::endl;

			// otherwise none
		}
		else if (node.op_type() == "Upsample") {
			// version 

			std::cout << node.op_type() << std::endl;
			auto returnVal0 = helper.get("mode", "nearest");

			//version == 1
			/*auto returnVal1 = helper.get("height_scale", 0.9f);
			auto returnVal2 = helper.get("width_scale", 0.9f);
			std::cout << returnVal0 << " " << returnVal1 << " " << returnVal2 << std::endl;*/

			//version == 7
			/*auto returnVal = helper.get("scales", std::vector<float>{});
			for (int i = 0; i < returnVal.size(); i++)
				std::cout << returnVal[i] << " ";
			std::cout << std::endl;*/

			// version ==9 none
		}
		else if (node.op_type() == "Slice") { 
			// version == 1

			std::cout << node.op_type() << std::endl;
			auto axes = helper.get("axes", std::vector<int>{});
			auto ends = helper.get("ends", std::vector<int>{});
			auto starts = helper.get("starts", std::vector<int>{});
			for (int i = 0; i < axes.size(); i++)
				std::cout << axes[i] << " ";
			std::cout << std::endl;

			for (int i = 0; i < ends.size(); i++)
				std::cout << ends[i] << " ";
			std::cout << std::endl;

			for (int i = 0; i < starts.size(); i++)
				std::cout << starts[i] << " ";
			std::cout << std::endl;

			// version >= 10 none
		}
		else if (node.op_type() == "Unsqueeze") { 

			// version 
			//if (version <= 11) {
				std::cout << node.op_type() << std::endl;
				auto returnVal = helper.get("axes", std::vector<int>{}); // vector
				for (int i = 0; i < returnVal.size() ; i++)
					std::cout << returnVal[i] << " ";
				std::cout << std::endl;
			//}
				
			// version >=13 none
		}


		std::cout << node.name() << "\tinputs[";
		for (int n = 0; n < node.input_size(); n++) {
			if (n != 0) std::cout << ",";
			std::cout << node.input(n);
		}
		std::cout << "]\toutputs[";
		for (int n = 0; n < node.output_size(); n++) {
			if (n != 0) std::cout << ",";
			std::cout << node.output(n);
		}
		std::cout << "]\n";
	}

	std::cout << "---- initializers ----" << std::endl;
	for (int i = 0; i < graph.initializer_size(); i++) {
		const auto& initializer = graph.initializer(i);
		std::cout << initializer.name() << "\t";
		std::cout << initializer.data_type() << ":" << initializer.dims_size() << "[";
		for (int n = 0; n < initializer.dims_size(); n++) {
			if (n != 0) std::cout << ",";
			std::cout << initializer.dims(n);
		}
		std::cout << "] " ;
		std::cout <<std::endl;


		//if (initializer.data_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
		//	const float* ptr = initializer.float_data().empty()
		//		? reinterpret_cast<const float*>(initializer.raw_data().data())
		//		: initializer.float_data().data();

		//	//const float* ptr = reinterpret_cast<const float*>(initializer.raw_data().data());

		//	std::cout << ptr[0];
		//}
		//std::cout << std::endl;
	}
	
	

	return 0;
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
