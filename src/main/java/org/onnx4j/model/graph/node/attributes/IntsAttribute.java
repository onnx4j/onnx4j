package org.onnx4j.model.graph.node.attributes;

import java.util.List;

import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class IntsAttribute extends Attribute<List<Long>> {

	public <T> IntsAttribute(AttributeProto attrProto) {
		super(attrProto.getIntsList(), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {}

}
