package org.onnx4j.model.graph.node.attributes;

import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class NullAttribute extends Attribute<Object> {

	public <T> NullAttribute(AttributeProto attrProto) {
		super(null, attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {}

}
