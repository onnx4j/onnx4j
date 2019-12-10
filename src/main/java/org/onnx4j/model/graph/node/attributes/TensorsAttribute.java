package org.onnx4j.model.graph.node.attributes;

import java.util.List;

import org.onnx4j.Tensor;
import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class TensorsAttribute extends Attribute<List<Tensor>> {

	public <T> TensorsAttribute(AttributeProto attrProto, Tensor.Options tensorOptions) {
		super(Tensor.toTensors(attrProto.getTensorsList(), tensorOptions), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {
		for (Tensor t : this.getValue()) {
			t.close();
		}
	}

}
