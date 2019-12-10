package org.onnx4j.model.graph.node.attributes;

import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class FloatAttribute extends Attribute<Float> {

	public <T> FloatAttribute(AttributeProto attrProto) {
		super(attrProto.getF(), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {}

}
