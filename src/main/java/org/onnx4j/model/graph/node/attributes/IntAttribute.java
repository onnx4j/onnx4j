package org.onnx4j.model.graph.node.attributes;

import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class IntAttribute extends Attribute<Long> {

	public <T> IntAttribute(AttributeProto attrProto) {
		super(attrProto.getI(), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {}

}
