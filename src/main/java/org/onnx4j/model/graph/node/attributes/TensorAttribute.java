package org.onnx4j.model.graph.node.attributes;

import org.onnx4j.Tensor;
import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class TensorAttribute extends Attribute<Tensor> {

	public <T> TensorAttribute(AttributeProto attrProto, Tensor.Options tensorOptions) {
		super(Tensor.toTensor(attrProto.getT(), tensorOptions), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {
		this.getValue().close();
	}

}
