package org.onnx4j.model.graph.node.attributes;

import java.util.List;

import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class FloatsAttribute extends Attribute<List<Float>> {

	public <T> FloatsAttribute(AttributeProto attrProto) {
		super(attrProto.getFloatsList(), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {}

}
