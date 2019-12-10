package org.onnx4j.model.graph.node.attributes;

import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class StringAttribute extends Attribute<String> {

	public <T> StringAttribute(AttributeProto attrProto) {
		super(attrProto.getS().toStringUtf8(), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {}

}
