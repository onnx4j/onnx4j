package org.onnx4j.model.graph;

import org.onnx4j.onnx.NamedOnnxObject;
import org.onnx4j.onnx.prototypes.OnnxProto3.ValueInfoProto;
import org.onnx4j.tensor.ValueInfo;

public abstract class Exchange extends NamedOnnxObject {
	
	protected ValueInfo valueInfo;

	public Exchange(ValueInfoProto valueInfoProto) {
		super(valueInfoProto.getName(), valueInfoProto.getDocString());
		this.valueInfo = ValueInfo.toValueInfo(valueInfoProto);
	}

	public ValueInfo getValueInfo() {
		return valueInfo;
	}
	
}
