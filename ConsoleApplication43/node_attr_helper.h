//
// Created by daquexian on 8/3/18.
//

#pragma once

#include "onnx.proto3.pb.h"
#include <vector>
#include <string>

namespace onnxruntime {

/**
 * Wrapping onnx::NodeProto for retrieving attribute values
 */
class NodeAttrHelper {
 public:
  explicit NodeAttrHelper(onnx::NodeProto proto);

  float get(const std::string& key,
            const float def_val) const;
  int get(const std::string& key,
          const int def_val) const;
  onnx::TensorProto get(const std::string& key) const;

  void IterateKey() const;

  std::vector<float> get(const std::string& key,
                         const std::vector<float>& def_val) const;
  std::vector<int> get(const std::string& key,
                       const std::vector<int>& def_val) const;
  std::string get(const std::string& key,
                  const std::string& def_val) const;

  bool has_attr(const std::string& key) const;

 private:
  onnx::NodeProto node_;
};

}  // namespace onnxruntime
