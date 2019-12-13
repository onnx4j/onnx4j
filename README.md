# onnx4j
## 简介
Onnx4j是一个面向ONNX（开放式神经网络交换协议）的Java表达项目，其并不提供具体的运算实现。

ONNX官方提供了其规范的proto定义，我们可以通过Google ProtoBuffer生成用于Java的相关定义类。onnx4j在ONNX官方的proto定义基础上，通过使用OOP的相关手段，如：接口与实现、类与继承等，使其转换为一个更为结构化和更清晰的表达方式，为所有建基于ONNX规范的Java程序开发提供更友好的开发方式。

对于onnx4j的使用，我们可以参考基于onnx4j派生的，专注于神经网络forward操作的框架Forwarder：https://github.com/onnx4j/forwarder
